"""A2 immutable-prefix replay and single-observation permutation."""
import collections
import copy
import json

from .common import digest
from .metrics import lexical_fact_hits
from .rollout import (FORMATS, PROMPT_VERSION, continue_record, observation_text, parse_action,
                      question_prompt)

# Replayable sources: every agent format whose prompt is a single user turn (ReSeek needs a system turn; U2 only).
PROMPT_VARIANT_OF = {f["version"]: v for v, f in FORMATS.items() if v != "reseek"}


def token_hash(tokens):
    return digest((",".join(map(str, tokens))).encode())


def observation(documents, variant="bounded"):
    return observation_text(documents, variant)


def visible_signature(documents):
    # Scores and hidden ranks do not enter the model's observation.
    value = [[d["document_id"], d["contents"]] for d in documents]
    return digest(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def validate_prefix(source, turn, agent, retrieval, generator=None):
    variant = PROMPT_VARIANT_OF.get(source.get("prompt_version"))
    if variant is None or source.get("condition") != "E0_clean":
        raise ValueError("Unsupported A1 prompt version or source condition")
    prefixes = source["prefixes"]
    if not 1 <= turn <= len(prefixes) or turn > agent["max_search_turns"]:
        raise ValueError("Target turn is not reachable in the source trajectory")
    text = source["initial_prompt"]
    tokens = None
    if generator is not None:
        expected = generator.initial_prompt(question_prompt(source["question"], agent["max_search_turns"], variant))
        if text != expected:
            raise ValueError("Source initial prompt differs from the frozen A1 prompt/tokenizer")
        tokens = generator.encode(text)
    generated = 0
    for index in range(turn):
        prefix, step = prefixes[index], source["steps"][index]
        raw = prefix["prefix_token_ids"]
        if not raw or any(type(t) is not int or t < 0 for t in raw):
            raise ValueError("Invalid raw prefix token IDs")
        if prefix["prefix_token_sha256"] != token_hash(raw):
            raise ValueError("Source prefix token checksum mismatch")
        if prefix["sample_id"] != source["sample_id"] or prefix["search_turn"] != index + 1:
            raise ValueError("Source prefix ID/turn mismatch")
        kind, query = parse_action(step["text"])
        if kind != "search" or query != prefix["search_query"]:
            raise ValueError("Source prefix query does not match its emitted action")
        generated += len(step["token_ids"])
        text += step["text"]
        if prefix["generated_tokens_before_observation"] != generated or prefix["prefix_text"] != text:
            raise ValueError("Source prefix text/token-cost history mismatch")
        if tokens is not None:
            if step["input_tokens"] != len(tokens):
                raise ValueError("Source input token count mismatch")
            tokens += step["token_ids"]
            if tokens != raw:
                raise ValueError("Raw token prefix cannot be reconstructed with this tokenizer")
        visible = prefix["visible_documents"]
        if len(visible) != retrieval["visible_top_k"]:
            raise ValueError("Source visible document count mismatch")
        if visible != prefix["candidate_documents"][:retrieval["visible_top_k"]]:
            raise ValueError("Source clean documents are not its original top-k")
        ids = [d["document_id"] for d in visible]
        if ids != prefix["original_document_ids"] or ids != prefix["observed_document_ids"]:
            raise ValueError("Source document IDs were altered")
        if prefix["observation_text"] != observation(visible, variant):
            raise ValueError("Source observation format/content mismatch")
        if index < turn - 1:
            text += prefix["observation_text"]
            if tokens is not None:
                tokens += generator.encode(prefix["observation_text"])
    return prefixes[turn - 1]


class RetrieverDriftError(RuntimeError):
    pass


class AuditedRetriever:
    """Normal live retrieval, without caching or hiding same-query drift."""
    def __init__(self, retriever, visible_top_k):
        self.retriever, self.visible_top_k = retriever, visible_top_k
        self.expected = {}
        self.calls = []

    def seed(self, query, documents):
        signature = visible_signature(documents[:self.visible_top_k])
        if query in self.expected and self.expected[query] != signature:
            raise RetrieverDriftError("A1 contains inconsistent same-query visible documents: " + query)
        self.expected[query] = signature

    def search(self, query, top_k):
        documents = self.retriever.search(query, top_k)
        if len(documents) < self.visible_top_k:
            raise RetrieverDriftError("Insufficient visible documents for query: " + query)
        signature = visible_signature(documents[:self.visible_top_k])
        previous = self.expected.get(query)
        matched = previous is None or previous == signature
        self.calls.append({"query": query, "visible_sha256": signature,
                           "had_reference": previous is not None,
                           "matched_reference": matched})
        if not matched:
            raise RetrieverDriftError("Live retriever differs from A1/prior visible results: " + query)
        self.expected[query] = signature
        return documents


def permute_documents(documents):
    if len(documents) < 2:
        raise ValueError("Permutation needs at least two documents")
    order = list(range(1, len(documents))) + [0]
    permuted = [copy.deepcopy(documents[i]) for i in order]
    signature = lambda items: collections.Counter(json.dumps(d, sort_keys=True, ensure_ascii=False) for d in items)
    if signature(permuted) != signature(documents):
        raise ValueError("Permutation changed the document multiset")
    return permuted, order


def replay_branch(source, turn, branch, generator, retriever, agent, retrieval):
    prefix = validate_prefix(source, turn, agent, retrieval, generator)
    if branch not in ("clean_a", "clean_b", "permuted_a"):
        raise ValueError("Unknown replay branch")
    visible = copy.deepcopy(prefix["visible_documents"])
    order = list(range(len(visible)))
    if branch == "permuted_a":
        visible, order = permute_documents(visible)
    return replay_observation(source, turn, branch, visible,
                              "E1_permutation" if branch == "permuted_a" else "E0_clean",
                              {"type": "permutation" if branch == "permuted_a" else "none",
                               "order": order, "document_multiset_preserved": True},
                              generator, retriever, agent, retrieval)


def replay_observation(source, turn, branch, documents, condition, metadata,
                       generator, retriever, agent, retrieval):
    """Shared continuation; callers validate the intervention before injection."""
    prefix = validate_prefix(source, turn, agent, retrieval, generator)
    visible = copy.deepcopy(documents)
    if len(visible) != retrieval["visible_top_k"]:
        raise ValueError("Intervention must preserve visible document count")
    obs = observation(visible)
    record = {key: copy.deepcopy(source[key]) for key in
              ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts", "initial_prompt")}
    record.update({"condition": condition,
                   "prompt_version": PROMPT_VERSION, "branch": branch, "target_turn": turn,
                   "status": None, "final_answer": "", "search_calls": turn,
                   "generated_tokens": prefix["generated_tokens_before_observation"],
                   "model_input_tokens": sum(s["input_tokens"] for s in source["steps"][:turn]),
                   "steps": copy.deepcopy(source["steps"][:turn]),
                   "prefixes": copy.deepcopy(source["prefixes"][:turn]),
                   "intervention_metadata": {**copy.deepcopy(metadata),
                                             "target_observation_changed": obs != prefix["observation_text"]}})
    target = record["prefixes"][-1]
    target.update({"visible_documents": visible, "observed_document_ids": [d["document_id"] for d in visible],
                   "observation_text": obs,
                   "visible_lexical_fact_ids": lexical_fact_hits(visible, source["gold_supporting_facts"])})
    tokens = list(prefix["prefix_token_ids"]) + generator.encode(obs)
    before = record["generated_tokens"]
    record = continue_record(record, generator, retriever, agent, retrieval,
                             prefix["prefix_text"] + obs, tokens)
    record["search_calls_after_target"] = record["search_calls"] - turn
    record["generated_tokens_after_target"] = record["generated_tokens"] - before
    return record


def replay_intervention(source, turn, branch, visible, intervention_metadata,
                        generator, retriever, agent, retrieval, condition="E0_clean"):
    """Compatibility with the API described in the factory f4a4bb5 report.

    Reconstructed from the supplied signature, not imported factory source.
    Intervention eligibility remains the caller's responsibility.
    """
    return replay_observation(source, turn, branch, visible, condition,
                              intervention_metadata, generator, retriever, agent, retrieval)


def continuation_signature(row):
    turn = row["target_turn"]
    content = {"tokens": [s["token_ids"] for s in row["steps"][turn:]],
               "texts": [s["text"] for s in row["steps"][turn:]],
               "queries": [p["search_query"] for p in row["prefixes"][turn:]],
               "observations": [p.get("observation_text") for p in row["prefixes"][turn:]],
               "status": row["status"], "answer": row["final_answer"]}
    return digest(json.dumps(content, sort_keys=True, ensure_ascii=False).encode())


def compare_branches(branches, turn):
    clean_a, clean_b, permuted = (branches[k] for k in ("clean_a", "clean_b", "permuted_a"))
    a, b, p = (r["prefixes"][turn - 1] for r in (clean_a, clean_b, permuted))
    same_prefix = a["prefix_token_ids"] == b["prefix_token_ids"] == p["prefix_token_ids"]
    untouched_history = all(r["steps"][:turn] == clean_a["steps"][:turn]
                            and r["prefixes"][:turn - 1] == clean_a["prefixes"][:turn - 1]
                            for r in (clean_b, permuted))
    unchanged_clean = a["observation_text"] == b["observation_text"]
    multiset = lambda docs: collections.Counter(json.dumps(d, ensure_ascii=False, sort_keys=True) for d in docs)
    preserved = multiset(a["visible_documents"]) == multiset(p["visible_documents"])
    if not (same_prefix and untouched_history and unchanged_clean and preserved):
        raise ValueError("Paired replay invariant failed")
    control_exact = continuation_signature(clean_a) == continuation_signature(clean_b)
    def first_query(row):
        later = row["prefixes"][turn:]
        return later[0]["search_query"] if later else None
    return {"same_raw_prefix": same_prefix, "pre_target_history_unchanged": untouched_history,
            "clean_observation_identical": unchanged_clean, "document_multiset_preserved": preserved,
            "permutation_observation_changed": a["observation_text"] != p["observation_text"],
            "clean_control_exact": control_exact,
            "clean_control_answer_equal": clean_a["final_answer"] == clean_b["final_answer"],
            "clean_vs_permuted_exact": continuation_signature(clean_a) == continuation_signature(permuted),
            "next_query_string_equal": first_query(clean_a) == first_query(permuted),
            "next_query_available_in_both": first_query(clean_a) is not None and first_query(permuted) is not None,
            "degradation": int(clean_a["em"] == 1 and permuted["em"] == 0),
            "improvement": int(clean_a["em"] == 0 and permuted["em"] == 1),
            "clean_em": clean_a["em"], "permuted_em": permuted["em"],
            "delta_search_calls": permuted["search_calls"] - clean_a["search_calls"]}

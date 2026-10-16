"""PCR v1: semantic-null placebo families and same-prefix branch arms.

Every arm forks the SAME immutable raw prefix at a target search turn. Placebo
arms change only how the target observation is presented (order or format),
never its document multiset or body text. Content and instruction arms are
always run together with the placebos so effects can be placebo-corrected.
"""
import collections
import copy
import hashlib
import itertools
import json

from .metrics import lexical_fact_hits, normalize
from .replay import continuation_signature, observation, validate_prefix
from .rollout import continue_record
from .selective import PROMPTS

VERSION = "PCR_placebo_controlled_replay_v1"
CONTROL_ARMS = ("clean_a", "clean_b")
PERMUTATION_ARMS = ("perm_1", "perm_2", "perm_3", "perm_4")
FORMAT_ARMS = ("fmt_1", "fmt_2")
PLACEBO_ARMS = PERMUTATION_ARMS + FORMAT_ARMS
RECOVERY_ARMS = ("review", "neutral", "budget2", "perm1_review", "perm1_neutral", "perm1_budget2")
DELETION_ARMS = ("drop_support", "drop_nonsupport")
ALL_ARMS = CONTROL_ARMS + PLACEBO_ARMS + RECOVERY_ARMS + DELETION_ARMS
ARM_GROUPS = {"core": CONTROL_ARMS + PLACEBO_ARMS,
              "recovery": CONTROL_ARMS + PLACEBO_ARMS + RECOVERY_ARMS,
              "deletion": CONTROL_ARMS + PLACEBO_ARMS + DELETION_ARMS,
              "all": ALL_ARMS}
REVIEW_PROMPT = PROMPTS["review"]  # identical text to SI1 review, for cross-checks
NEUTRAL_SEED = ("\n\nNote: this line is a neutral layout separator and carries no task "
                "information or instruction. ")
NEUTRAL_FILLER = (" ok", " .", " -", " note", " line")
REPLACEMENT_RATIO = (0.5, 2.0)


def _split(document):
    content = document["contents"]
    title = content.split("\n", 1)[0]
    body = content.split("\n", 1)[1] if "\n" in content else ""
    return title, body


def render(documents, variant="canonical"):
    """Observation text. Titles/bodies are byte-identical across variants."""
    if variant == "canonical":
        return observation(documents)
    parts = []
    for rank, document in enumerate(documents, 1):
        title, body = _split(document)
        if variant == "fmt_1":
            parts.append("Doc {} (Title: {})\n{}\n\n".format(rank, title, body))
        elif variant == "fmt_2":
            parts.append("[{}] {}\n{}\n".format(rank, title, body))
        else:
            raise ValueError("Unknown observation format variant: " + variant)
    return "\n\n<information>{}</information>\n\n".format("".join(parts))


def multiset(documents):
    return collections.Counter(json.dumps([d["document_id"], d["contents"]], ensure_ascii=False)
                               for d in documents)


def permutations_for(sample_id, turn, size, count=4):
    """Deterministic, outcome-blind choice among the non-identity permutations."""
    options = [p for p in itertools.permutations(range(size)) if list(p) != list(range(size))]
    if len(options) < count:
        raise ValueError("Not enough distinct non-identity permutations")
    key = lambda p: hashlib.sha256("{}|{}|{}".format(sample_id, turn, p).encode()).hexdigest()
    return [list(p) for p in sorted(options, key=key)[:count]]


def neutral_instruction(encode, target_tokens):
    """Semantically empty text with exactly the review prompt's token count."""
    count = lambda t: len(encode(t + "\n\n"))
    text, step = NEUTRAL_SEED, 0
    while count(text) < target_tokens:
        text += NEUTRAL_FILLER[step % len(NEUTRAL_FILLER)]
        step += 1
    while count(text) > target_tokens and len(text) > 2:
        text = text[:-1]
    if count(text) != target_tokens:
        raise ValueError("Cannot match neutral instruction token count to the review prompt")
    return text + "\n\n"


def contains_answer(document, gold):
    answer = normalize(gold)
    return bool(answer) and (" " + answer + " ") in (" " + normalize(document["contents"]) + " ")


def deletion_plan(prefix, source, turn):
    """Outcome-blind support/non-support slot replacement with one shared real candidate.

    Lexical title+sentence proxy for supporting facts; recorded, never upgraded
    to a semantic certification. Both arms use the SAME replacement candidate.
    """
    visible, facts = prefix["visible_documents"], source["gold_supporting_facts"]
    hits = [lexical_fact_hits([d], facts) for d in visible]
    gold = source["gold_answer"]
    support = [i for i, h in enumerate(hits) if h]
    nonsupport = [i for i, h in enumerate(hits) if not h and not contains_answer(visible[i], gold)]
    earlier = set(lexical_fact_hits([d for p in source["prefixes"][:turn - 1]
                                     for d in p["visible_documents"]], facts))
    plan = {"support_slots": support, "nonsupport_slots": nonsupport,
            "visible_fact_ids": sorted({i for h in hits for i in h}),
            "earlier_fact_ids": sorted(earlier), "candidate": None, "arms": {}}
    ids = {d["document_id"] for d in visible}
    contents = {d["contents"] for d in visible}
    for candidate in prefix["candidate_documents"][len(visible):]:
        if (candidate["document_id"] in ids or candidate["contents"] in contents
                or lexical_fact_hits([candidate], facts) or contains_answer(candidate, gold)):
            continue
        plan["candidate"] = candidate
        break
    for arm, slots in (("drop_support", support), ("drop_nonsupport", nonsupport)):
        entry = {"eligible": False, "reason": None}
        if not slots:
            entry["reason"] = "no_" + arm.split("_")[1] + "_slot"
        elif plan["candidate"] is None:
            entry["reason"] = "no_clean_candidate"
        else:
            slot = slots[0]
            ratio = len(plan["candidate"]["contents"]) / max(1, len(visible[slot]["contents"]))
            if not REPLACEMENT_RATIO[0] <= ratio <= REPLACEMENT_RATIO[1]:
                entry["reason"] = "length_ratio_outside_frozen_range"
            else:
                after = copy.deepcopy(visible)
                after[slot] = copy.deepcopy(plan["candidate"])
                lost = set(plan["visible_fact_ids"]) - set(lexical_fact_hits(after, facts))
                entry = {"eligible": True, "reason": None, "slot": slot,
                         "removed_document_id": visible[slot]["document_id"],
                         "replacement_document_id": plan["candidate"]["document_id"],
                         "replacement_rank": plan["candidate"].get("rank"),
                         "length_ratio": ratio, "lost_fact_ids": sorted(lost),
                         "lost_fact_not_seen_earlier": sorted(lost - earlier),
                         "removed_contains_answer": contains_answer(visible[slot], gold),
                         "documents": after}
        plan["arms"][arm] = entry
    plan.pop("candidate")
    return plan


def arm_spec(arm, prefix, source, turn, encode, deletion):
    """Return (documents, format variant, instruction, extra budget, metadata) or None."""
    visible = copy.deepcopy(prefix["visible_documents"])
    perms = permutations_for(source["sample_id"], turn, len(visible))
    permuted = lambda k: [copy.deepcopy(visible[i]) for i in perms[k]]
    review_tokens = len(encode(REVIEW_PROMPT))
    instructions = {"review": REVIEW_PROMPT, "neutral": neutral_instruction(encode, review_tokens)}
    if arm in CONTROL_ARMS:
        return visible, "canonical", "", 0, {"family": "E0_control"}
    if arm in PERMUTATION_ARMS:
        k = PERMUTATION_ARMS.index(arm)
        return permuted(k), "canonical", "", 0, {"family": "N1_permutation", "order": perms[k]}
    if arm in FORMAT_ARMS:
        return visible, arm, "", 0, {"family": "N2_format_null", "variant": arm}
    if arm in RECOVERY_ARMS:
        base = arm.split("_", 1)[1] if arm.startswith("perm1_") else arm
        documents = permuted(0) if arm.startswith("perm1_") else visible
        meta = {"family": "recovery", "base_observation": "perm_1" if arm.startswith("perm1_") else "clean",
                "operation": base, "order": perms[0] if arm.startswith("perm1_") else list(range(len(visible)))}
        if base == "budget2":
            return documents, "canonical", "", 2, dict(meta, extra_search_budget=2)
        text = instructions[base]
        return documents, "canonical", text, 0, dict(meta, instruction=text,
                                                      instruction_tokens=len(encode(text)),
                                                      review_tokens=review_tokens)
    if arm in DELETION_ARMS:
        entry = deletion["arms"][arm]
        if not entry["eligible"]:
            return None
        meta = {k: v for k, v in entry.items() if k != "documents"}
        return copy.deepcopy(entry["documents"]), "canonical", "", 0, dict(meta, family="content_deletion")
    raise ValueError("Unknown PCR arm: " + arm)


def run_branch(source, turn, arm, generator, retriever, agent, retrieval, deletion):
    prefix = validate_prefix(source, turn, agent, retrieval, generator)
    spec = arm_spec(arm, prefix, source, turn, generator.encode, deletion)
    if spec is None:
        return None
    documents, variant, instruction, extra, meta = spec
    if len(documents) != retrieval["visible_top_k"]:
        raise ValueError("Arm changed the visible document count")
    if arm not in DELETION_ARMS and multiset(documents) != multiset(prefix["visible_documents"]):
        raise ValueError("Placebo/recovery arm changed the document multiset")
    obs = render(documents, variant)
    record = {key: copy.deepcopy(source[key]) for key in
              ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts",
               "initial_prompt", "prompt_version")}
    record.update(condition=VERSION + ":" + arm, target_turn=turn, status=None, final_answer="",
                  search_calls=turn, generated_tokens=prefix["generated_tokens_before_observation"],
                  model_input_tokens=sum(s["input_tokens"] for s in source["steps"][:turn]),
                  steps=copy.deepcopy(source["steps"][:turn]), prefixes=copy.deepcopy(source["prefixes"][:turn]))
    target = record["prefixes"][-1]
    target.update(visible_documents=copy.deepcopy(documents),
                  observed_document_ids=[d["document_id"] for d in documents], observation_text=obs,
                  visible_lexical_fact_ids=lexical_fact_hits(documents, source["gold_supporting_facts"]))
    tokens = list(prefix["prefix_token_ids"]) + generator.encode(obs) + (generator.encode(instruction) if instruction else [])
    run_agent = dict(agent, max_search_turns=agent["max_search_turns"] + extra)
    before = record["generated_tokens"]
    record = continue_record(record, generator, retriever, run_agent, retrieval,
                             prefix["prefix_text"] + obs + instruction, tokens)
    if record["status"] == "infrastructure_error":
        raise RuntimeError("Infrastructure failure in PCR arm {}: {}".format(arm, record.get("error")))
    return {"arm": arm, "metadata": meta,
            "observation_changed": obs != prefix["observation_text"],
            "observation_sha256": hashlib.sha256(obs.encode()).hexdigest(),
            "post_observation_input_tokens": len(tokens),
            "continuation_sha256": continuation_signature(record),
            "status": record["status"], "final_answer": record["final_answer"],
            "em": record["em"], "f1": record["f1"], "search_calls": record["search_calls"],
            "search_calls_after_target": record["search_calls"] - turn,
            "generated_tokens_after_target": record["generated_tokens"] - before,
            "model_input_tokens": record["model_input_tokens"],
            "post_target_texts": [s["text"] for s in record["steps"][turn:]],
            "post_target_queries": [p["search_query"] for p in record["prefixes"][turn:]],
            "post_target_document_ids": [p["observed_document_ids"] for p in record["prefixes"][turn:]],
            "post_target_fact_ids": [p["visible_lexical_fact_ids"] for p in record["prefixes"][turn:]]}


def run_state(source, turn, arms, generator, retriever, agent, retrieval):
    """All arms of one (question, turn) state; E0/E0 must reproduce the source exactly."""
    prefix = validate_prefix(source, turn, agent, retrieval, generator)
    deletion = deletion_plan(prefix, source, turn)
    expected = continuation_signature({**source, "target_turn": turn})
    results = {}
    for arm in arms:
        results[arm] = run_branch(source, turn, arm, generator, retriever, agent, retrieval, deletion)
        if arm in CONTROL_ARMS and results[arm]["continuation_sha256"] != expected:
            raise ValueError("E0 control does not reproduce source continuation: {} turn {} {}".format(
                source["sample_id"], turn, arm))
    return {"version": VERSION, "sample_id": source["sample_id"], "dataset": source["dataset"],
            "target_turn": turn, "question": source["question"], "gold_answer": source["gold_answer"],
            "source_status": source["status"], "source_em": source["em"], "source_f1": source["f1"],
            "source_final_answer": source["final_answer"], "source_search_calls": source["search_calls"],
            "target_query": prefix["search_query"],
            "target_visible_fact_ids": prefix["visible_lexical_fact_ids"],
            "trajectory_fact_ids": sorted({i for p in source["prefixes"] for i in p["visible_lexical_fact_ids"]}),
            "candidate_fact_ids": sorted({i for p in source["prefixes"] for i in p.get("candidate_lexical_fact_ids", [])}),
            "known_fact_ids": [i for i, f in enumerate(source["gold_supporting_facts"]) if f.get("sentence")],
            "deletion_plan": {k: v for k, v in deletion.items() if k != "arms"},
            "deletion_eligibility": {a: {k: v for k, v in e.items() if k != "documents"}
                                     for a, e in deletion["arms"].items()},
            "controls_exact": True, "arms": results}

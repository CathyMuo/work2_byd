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
import random

from .metrics import gold_list, lexical_fact_hits, normalize
from .replay import PROMPT_VARIANT_OF, continuation_signature, observation, validate_prefix
from .rollout import FORMATS, continue_record
from .selective import PROMPTS

VERSION = "PCR_placebo_controlled_replay_v1"
CONTROL_ARMS = ("clean_a", "clean_b")
# U1L amendment (2026-10-30, before any research_7b arm outcome was analysed): a run frozen with
# control_mismatch="record" keeps a state whose E0 replay differs from its source, marks it
# controls_exact=False with a divergence receipt, and every analysis drops it. An agent whose
# mismatch rate exceeds this bound is not measurable.
CONTROL_MISMATCH_MAX = 0.02
PERMUTATION_ARMS = ("perm_1", "perm_2", "perm_3", "perm_4")
FORMAT_ARMS = ("fmt_1", "fmt_2")
PLACEBO_ARMS = PERMUTATION_ARMS + FORMAT_ARMS
RECOVERY_ARMS = ("review", "neutral", "budget2", "perm1_review", "perm1_neutral", "perm1_budget2")
DELETION_ARMS = ("drop_support", "drop_nonsupport")
# AC1 (2026-10-19): a second, more specific reflection method with its own token-matched placebo.
REFLECT_ARMS = ("reflect", "neutral_reflect", "perm1_reflect", "perm1_neutral_reflect")
# AS1 (2026-11-02, research/AS1_U1N_PROTOCOL.md): assay-sensitivity arms on the perm_1 observation. alert flags the current
# line of reasoning as likely wrong without saying why; hint states the gold answer (an oracle, not a method). Each has its
# own token-matched filler.
ASSAY_ARMS = ("perm1_alert", "perm1_neutral_alert", "perm1_hint", "perm1_neutral_hint")
# AS2 (2026-11-04, research/AS2_PROTOCOL.md). Experiment 1: four more content-free texts, each token-matched to the review
# prompt, on the perm_1 observation. Experiment 2: graded positive controls between alert (no information) and hint (the
# answer): pointer says which shown document contains the answer, evidence adds one answer-bearing candidate passage; each
# has its own matched filler and is None where the state is not eligible. Experiment 3: sampled continuations (k per arm)
# of the H1 arms from the same token prefix.
FILLER_VARIANT_ARMS = ("perm1_fill_prose", "perm1_fill_protocol", "perm1_fill_shuffled", "perm1_fill_punct")
GRADED_ARMS = ("perm1_pointer", "perm1_neutral_pointer", "perm1_evidence", "perm1_neutral_evidence")
SAMPLED_ARMS = ("s_perm1", "s_perm2", "s_perm1_review", "s_perm1_neutral", "s_perm1_reflect", "s_perm1_neutral_reflect")
SAMPLED_BASE = {"s_perm1": "perm_1", "s_perm2": "perm_2"}
ALL_ARMS = CONTROL_ARMS + PLACEBO_ARMS + RECOVERY_ARMS + DELETION_ARMS
ARM_GROUPS = {"core": CONTROL_ARMS + PLACEBO_ARMS,
              "recovery": CONTROL_ARMS + PLACEBO_ARMS + RECOVERY_ARMS,
              "deletion": CONTROL_ARMS + PLACEBO_ARMS + DELETION_ARMS,
              "all": ALL_ARMS,
              "audit": ALL_ARMS + REFLECT_ARMS,
              # U1 (2026-10-22): only what H1 needs (selected-fragile design + both methods and their placebos).
              "h1": CONTROL_ARMS + ("perm_1", "perm_2", "perm1_review", "perm1_neutral", "perm1_budget2",
                                    "perm1_reflect", "perm1_neutral_reflect"),
              # AS1 / U1N (2026-11-02): positive controls on H1 states; the prompts on the unperturbed observation.
              "assay": CONTROL_ARMS + ("perm_1", "perm1_review", "perm1_neutral") + ASSAY_ARMS,
              "natural": CONTROL_ARMS + ("perm_1", "review", "neutral", "reflect", "neutral_reflect"),
              # AS2 (2026-11-04): filler variants and graded positive controls on AS1 states; sampled H1 arms.
              "as2": CONTROL_ARMS + ("perm_1", "perm1_review", "perm1_neutral") + FILLER_VARIANT_ARMS + GRADED_ARMS,
              "as2s": CONTROL_ARMS + ("perm_1",) + SAMPLED_ARMS}
REVIEW_PROMPT = PROMPTS["review"]  # identical text to SI1 review, for cross-checks
REFLECT_PROMPT = ("\n\nBefore continuing, reflect: state which part of the original question is still unresolved, "
                  "check whether the information already shown actually resolves it, and if something is missing, "
                  "issue one more specific search within the remaining original budget; otherwise give the answer. "
                  "Use the original <search> or <answer> protocol.\n\n")
ALERT_PROMPT = ("\n\nBefore continuing, note: an external check indicates that the current line of reasoning is likely to "
                "lead to a wrong answer. Re-examine the original question and the information already shown, and search again "
                "or answer within the remaining original budget. Use the original <search> or <answer> protocol.\n\n")
HINT_TEMPLATE = ("\n\nBefore continuing, note: a verified reference states that the answer to the original question is "
                 "\"{}\". Use the original <search> or <answer> protocol.\n\n")
NEUTRAL_SEED = ("\n\nNote: this line is a neutral layout separator and carries no task "
                "information or instruction. ")
NEUTRAL_FILLER = (" ok", " .", " -", " note", " line")
REPLACEMENT_RATIO = (0.5, 2.0)
# AS2 texts. The prose filler is ordinary narrative with no named entities, numbers or dates, so it cannot answer a QA question;
# a state where it nevertheless contains a gold alias is flagged and dropped from that contrast.
PROSE_FILLER = ("The morning was quiet and the light came in slowly over the rooftops. A few people walked along the path "
                "beside the river, some carrying bags, some simply looking at the water. The shops were only beginning to open, "
                "and someone was setting out chairs in front of a small cafe while a delivery van waited at the corner. "
                "Later the wind picked up a little, and the leaves on the trees moved in long, soft waves. Children went past "
                "on bicycles, calling to each other, and an older man stopped to read a notice on a wooden board. By noon the "
                "streets were busier, the air was warmer, and the sound of conversation filled the open square. Nothing in "
                "particular happened, and the afternoon passed in the same unhurried way as the morning had.")
PROTOCOL_SENTENCE = " Use the original <search> or <answer> protocol."
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
POINTER_TEMPLATE = ("\n\nBefore continuing, note: a verified reference indicates that the {} of the documents shown above contains "
                    "the answer to the original question. Use the original <search> or <answer> protocol.\n\n")
EVIDENCE_TEMPLATE = ("\n\nBefore continuing, note: one more search result for the original question follows.\n{}\n"
                     "Use the original <search> or <answer> protocol.\n\n")


def _split(document):
    content = document["contents"]
    title = content.split("\n", 1)[0]
    body = content.split("\n", 1)[1] if "\n" in content else ""
    return title, body


def render(documents, variant="canonical", style="bounded"):
    """Observation text in the agent's style. Titles/bodies are byte-identical across variants."""
    if variant == "canonical":
        return observation(documents, style)
    parts = []
    for rank, document in enumerate(documents, 1):
        title, body = _split(document)
        if variant == "fmt_1":
            parts.append("Doc {} (Title: {})\n{}\n\n".format(rank, title, body))
        elif variant == "fmt_2":
            parts.append("[{}] {}\n{}\n".format(rank, title, body))
        else:
            raise ValueError("Unknown observation format variant: " + variant)
    return "\n\n<{0}>{1}</{0}>\n\n".format(FORMATS[style]["tag"], "".join(parts))


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


def fill_to(encode, target_tokens, head, units, tail):
    """head + units (cycled, last one cut at a character) + tail with exactly target_tokens tokens."""
    count = lambda body: len(encode(head + body + tail))
    if count("") > target_tokens:
        raise ValueError("Filler frame alone exceeds the target token count")
    body, step = "", 0
    while count(body) < target_tokens:
        body += units[step % len(units)]
        step += 1
    while count(body) > target_tokens and body:
        body = body[:-1]
    if count(body) != target_tokens:
        raise ValueError("Cannot match filler token count")
    return head + body + tail


def state_rng(*parts):
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def filler_variant(arm, encode, target_tokens, sample_id, turn):
    """AS2 experiment 1: the four content-free texts, each with exactly the review prompt's token count."""
    if arm == "perm1_fill_prose":
        return fill_to(encode, target_tokens, "\n\n", [" " + w for w in PROSE_FILLER.split()], "\n\n")
    if arm == "perm1_fill_protocol":
        return fill_to(encode, target_tokens, NEUTRAL_SEED, NEUTRAL_FILLER, PROTOCOL_SENTENCE + "\n\n")
    if arm == "perm1_fill_shuffled":
        words = REVIEW_PROMPT.split()
        state_rng(sample_id, turn, "shuffle").shuffle(words)
        return fill_to(encode, target_tokens, "\n\n", [" " + w for w in words], "\n\n")
    if arm == "perm1_fill_punct":
        return fill_to(encode, target_tokens, "\n\n", [" ."], "\n\n")
    raise ValueError("Unknown filler variant: " + arm)


def sample_seed(sample_id, turn, arm, index):
    return int(hashlib.sha256("{}|{}|{}|{}".format(sample_id, turn, arm, index).encode()).hexdigest()[:8], 16) % (2 ** 31)


def contains_answer(document, gold):
    """True if any gold alias appears as a whole normalized phrase in the document."""
    text = " " + normalize(document["contents"]) + " "
    return any(normalize(g) and (" " + normalize(g) + " ") in text for g in gold_list(gold))


def pointer_prompt(shown, gold):
    """AS2 pointer: which shown document (by position in the displayed order) contains the answer; None if none does."""
    k = next((i for i, d in enumerate(shown) if contains_answer(d, gold)), None)
    if k is None or k >= len(ORDINALS):
        return None, {"eligible": False}
    return POINTER_TEMPLATE.format(ORDINALS[k]), {"eligible": True, "position": k + 1, "document_id": shown[k]["document_id"]}


def evidence_prompt(prefix, gold):
    """AS2 evidence: the highest-ranked retrieval candidate beyond the visible ones that contains the answer and does not
    repeat a visible document; None if there is none."""
    visible = prefix["visible_documents"]
    ids, contents = {d["document_id"] for d in visible}, {d["contents"] for d in visible}
    for rank, d in enumerate(prefix["candidate_documents"][len(visible):], len(visible) + 1):
        if d["document_id"] not in ids and d["contents"] not in contents and contains_answer(d, gold):
            return EVIDENCE_TEMPLATE.format(d["contents"]), {"eligible": True, "document_id": d["document_id"], "candidate_rank": rank}
    return None, {"eligible": False}


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
    if arm in SAMPLED_ARMS:  # AS2 experiment 3: the same branch as its greedy arm; only decoding differs (run_branch)
        spec = arm_spec(SAMPLED_BASE.get(arm, arm[2:]), prefix, source, turn, encode, deletion)
        return spec[:4] + (dict(spec[4], sampled_from=SAMPLED_BASE.get(arm, arm[2:])),)
    visible = copy.deepcopy(prefix["visible_documents"])
    perms = permutations_for(source["sample_id"], turn, len(visible))
    permuted = lambda k: [copy.deepcopy(visible[i]) for i in perms[k]]
    review_tokens = len(encode(REVIEW_PROMPT))
    # Each content prompt and the filler matched to it; "neutral" is the review-matched filler (H1 naming).
    prompts = {"review": REVIEW_PROMPT, "reflect": REFLECT_PROMPT, "alert": ALERT_PROMPT,
               "hint": HINT_TEMPLATE.format(gold_list(source["gold_answer"])[0])}
    method_of = {"neutral": "review", "neutral_reflect": "reflect", "neutral_alert": "alert", "neutral_hint": "hint"}
    if arm in CONTROL_ARMS:
        return visible, "canonical", "", 0, {"family": "E0_control"}
    if arm in PERMUTATION_ARMS:
        k = PERMUTATION_ARMS.index(arm)
        return permuted(k), "canonical", "", 0, {"family": "N1_permutation", "order": perms[k]}
    if arm in FORMAT_ARMS:
        return visible, arm, "", 0, {"family": "N2_format_null", "variant": arm}
    if arm in FILLER_VARIANT_ARMS:
        text = filler_variant(arm, encode, review_tokens, source["sample_id"], turn)
        return permuted(0), "canonical", text, 0, {
            "family": "filler_variant", "base_observation": "perm_1", "operation": arm.split("_", 1)[1], "order": perms[0],
            "instruction": text, "instruction_tokens": len(encode(text)), "matched_tokens": review_tokens,
            "review_tokens": review_tokens, "leaks_answer": contains_answer({"contents": text}, source["gold_answer"])}
    graded = {}
    if arm in GRADED_ARMS:  # computed only for these arms, so every earlier arm's text and metadata are unchanged
        graded["pointer"] = pointer_prompt(permuted(0), source["gold_answer"])
        graded["evidence"] = evidence_prompt(prefix, source["gold_answer"])
        prompts.update({k: v[0] for k, v in graded.items()})
        method_of.update(neutral_pointer="pointer", neutral_evidence="evidence")
    if arm in RECOVERY_ARMS or arm in REFLECT_ARMS or arm in ASSAY_ARMS or arm in GRADED_ARMS:
        base = arm.split("_", 1)[1] if arm.startswith("perm1_") else arm
        documents = permuted(0) if arm.startswith("perm1_") else visible
        meta = {"family": "recovery", "base_observation": "perm_1" if arm.startswith("perm1_") else "clean",
                "operation": base, "order": perms[0] if arm.startswith("perm1_") else list(range(len(visible)))}
        if base == "budget2":
            return documents, "canonical", "", 2, dict(meta, extra_search_budget=2)
        method = method_of.get(base, base)
        if method in graded:
            if prompts[method] is None:
                return None  # not eligible: the arm is not run (recorded as None, like ineligible deletion arms)
            meta["eligibility"] = graded[method][1]
        matched = len(encode(prompts[method]))
        text = neutral_instruction(encode, matched) if base in method_of else prompts[method]
        return documents, "canonical", text, 0, dict(meta, instruction=text,
                                                      instruction_tokens=len(encode(text)),
                                                      matched_tokens=matched, review_tokens=review_tokens)
    if arm in DELETION_ARMS:
        entry = deletion["arms"][arm]
        if not entry["eligible"]:
            return None
        meta = {k: v for k, v in entry.items() if k != "documents"}
        return copy.deepcopy(entry["documents"]), "canonical", "", 0, dict(meta, family="content_deletion")
    raise ValueError("Unknown PCR arm: " + arm)


def run_branch(source, turn, arm, generator, retriever, agent, retrieval, deletion, sample=None):
    """sample = (temperature, top_p, seed) samples the continuation (AS2 experiment 3); None keeps greedy decoding."""
    prefix = validate_prefix(source, turn, agent, retrieval, generator)
    spec = arm_spec(arm, prefix, source, turn, generator.encode, deletion)
    if spec is None:
        return None
    documents, variant, instruction, extra, meta = spec
    if len(documents) != retrieval["visible_top_k"]:
        raise ValueError("Arm changed the visible document count")
    if arm not in DELETION_ARMS and multiset(documents) != multiset(prefix["visible_documents"]):
        raise ValueError("Placebo/recovery arm changed the document multiset")
    obs = render(documents, variant, PROMPT_VARIANT_OF[source["prompt_version"]])
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
    if sample is not None:
        run_agent.update(temperature=sample[0], top_p=sample[1], seed=sample[2])
    before = record["generated_tokens"]
    record = continue_record(record, generator, retriever, run_agent, retrieval,
                             prefix["prefix_text"] + obs + instruction, tokens)
    if record["status"] == "infrastructure_error":
        raise RuntimeError("Infrastructure failure in PCR arm {}: {}".format(arm, record.get("error")))
    out = {"arm": arm, "metadata": meta,
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
    if sample is not None:
        out["sampling"] = {"temperature": sample[0], "top_p": sample[1], "seed": sample[2]}
    return out


def run_sampled(source, turn, arm, generator, retriever, agent, retrieval, deletion, sampling):
    """k sampled continuations of one arm from the same token prefix, each with its own outcome-blind seed."""
    if not sampling:
        raise ValueError("Sampled arm {} needs a sampling configuration".format(arm))
    return {"arm": arm, "samples": [run_branch(source, turn, arm, generator, retriever, agent, retrieval, deletion,
                                               (sampling["temperature"], sampling["top_p"], sample_seed(source["sample_id"], turn, arm, i)))
                                    for i in range(sampling["k"])]}


def divergence(source, turn, branch, chars=160):
    """Where an E0 replay first leaves its source continuation (text, step count, query, documents or tokens)."""
    src = [s["text"] for s in source["steps"][turn:]]
    rep = branch["post_target_texts"]
    out = {"source_steps": len(src), "replay_steps": len(rep), "source_em": source["em"], "replay_em": branch["em"],
           "source_answer": source["final_answer"], "replay_answer": branch["final_answer"]}
    for i, (a, b) in enumerate(zip(src, rep)):
        if a != b:
            k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return dict(out, kind="text", step=turn + i, char=k, source_text=a[max(0, k - chars):k + chars],
                        replay_text=b[max(0, k - chars):k + chars])
    if len(src) != len(rep):
        return dict(out, kind="step_count", step=turn + min(len(src), len(rep)))
    if [p["search_query"] for p in source["prefixes"][turn:]] != branch["post_target_queries"]:
        return dict(out, kind="query")
    if [p["observed_document_ids"] for p in source["prefixes"][turn:]] != branch["post_target_document_ids"]:
        return dict(out, kind="documents")
    return dict(out, kind="tokens_or_status")


def run_state(source, turn, arms, generator, retriever, agent, retrieval, on_mismatch="stop", sampling=None):
    """All arms of one (question, turn) state; E0/E0 must reproduce the source exactly.

    on_mismatch="stop" raises (pre-registered default); "record" still runs every arm but marks the state
    controls_exact=False with a divergence receipt so that analyses exclude it (see CONTROL_MISMATCH_MAX)."""
    if on_mismatch not in ("stop", "record"):
        raise ValueError("on_mismatch must be stop or record")
    prefix = validate_prefix(source, turn, agent, retrieval, generator)
    deletion = deletion_plan(prefix, source, turn)
    expected = continuation_signature({**source, "target_turn": turn})
    results, mismatch = {}, {}
    for arm in arms:
        if arm in SAMPLED_ARMS:
            results[arm] = run_sampled(source, turn, arm, generator, retriever, agent, retrieval, deletion, sampling)
            continue
        results[arm] = run_branch(source, turn, arm, generator, retriever, agent, retrieval, deletion)
        if arm in CONTROL_ARMS and results[arm]["continuation_sha256"] != expected:
            if on_mismatch == "stop":
                raise ValueError("E0 control does not reproduce source continuation: {} turn {} {}".format(
                    source["sample_id"], turn, arm))
            mismatch[arm] = divergence(source, turn, results[arm])
    controls = [results[a] for a in CONTROL_ARMS if a in results]
    receipt = {"control_mismatch": {"arms": mismatch, "controls_agree": len({c["continuation_sha256"] for c in controls}) == 1}} \
        if mismatch else {}
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
            "controls_exact": not mismatch, **receipt, "arms": results}


def usable_states(rows, manifest=None):
    """States with exact E0 controls; with a merged manifest, the dropped count must match its receipt."""
    keep = [r for r in rows if r.get("controls_exact", True)]
    if manifest is not None and len(rows) - len(keep) != manifest.get("control_mismatch_states", 0):
        raise ValueError("Control-mismatch states disagree with the merged manifest")
    return keep

"""Restore actual perturbed history AFTER an observed search result, without relabeling E0."""
import collections
import copy
import json

from .common import digest
from .replay import observation, token_hash, validate_prefix
from .rollout import PROMPT_VERSION, continue_record, parse_action

PROMPTS = {
    "original_a": "", "original_b": "",
    "general_review": "\n\nBefore continuing, briefly review the original question and the information already shown. Check your reasoning and decide whether to answer or search within the remaining original budget. Use the original <search> or <answer> protocol.\n\n",
    "constraint_review": "\n\nBefore continuing, identify the exact entity, relation, time and comparison constraints required by the original question. Check which already-shown evidence supports those constraints, and avoid transferring an answer for a different question. Decide whether to answer or search within the remaining original budget. Use the original <search> or <answer> protocol.\n\n",
}
PROTOCOL = {"version": "work2_E_post_observation_v1", "prompts": PROMPTS,
            "anchor": "after max(parent intervention turn, reviewed earliest sufficient turn)",
            "controls": "all selected originals twice and historical agreement BEFORE any review arms",
            "budget": "inherit A1; no reset, 4 total successful searches, 512 tokens per generation segment",
            "selection": "outcome-selected development engineering, not deployable triggering or confirmation",
            "stable_input_length_ratio": [0.75, 1.25], "max_failed_cases": 4, "max_stable_proposals": 3}


def value_hash(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def restore_after_observation(row, clean, parent_turn, turn, agent, retrieval, generator=None):
    if row.get("condition") not in ("E1_permutation", "E2_distractor", "E3_bridge_removal", "E4_redundancy") or row.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("E needs a real perturbed branch; do not relabel it E0")
    if row["target_turn"] != parent_turn or not parent_turn <= turn <= len(row["prefixes"]) or turn > agent["max_search_turns"]:
        raise ValueError("Resume must be reachable and at/after the parent intervention")
    for key in ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts", "initial_prompt"):
        if row[key] != clean[key]:
            raise ValueError("Perturbed/source identity changed: " + key)
    validate_prefix(clean, parent_turn, agent, retrieval, generator)
    if row["steps"][:parent_turn] != clean["steps"][:parent_turn] or row["prefixes"][:parent_turn-1] != clean["prefixes"][:parent_turn-1]:
        raise ValueError("History before parent intervention was altered")
    original, changed = clean["prefixes"][parent_turn-1], row["prefixes"][parent_turn-1]
    for key in ("prefix_token_ids", "prefix_token_sha256", "prefix_text", "search_query", "candidate_documents", "original_document_ids"):
        if original[key] != changed[key]:
            raise ValueError("Parent intervention changed its saved prefix/candidates: " + key)
    if row["condition"] == "E1_permutation":
        multiset = lambda docs: collections.Counter(json.dumps(d, sort_keys=True, ensure_ascii=False) for d in docs)
        if multiset(original["visible_documents"]) != multiset(changed["visible_documents"]):
            raise ValueError("Parent E1 changed document multiset")
    elif sum(a != b for a, b in zip(original["visible_documents"], changed["visible_documents"])) != 1:
        raise ValueError("Parent E2/E3/E4 did not change exactly one slot")
    if original["observation_text"] == changed["observation_text"]:
        raise ValueError("Parent intervention observation unchanged")
    text, tokens = row["initial_prompt"], generator.encode(row["initial_prompt"]) if generator else None
    generated = 0
    for index in range(turn):
        step, prefix = row["steps"][index], row["prefixes"][index]
        raw = prefix["prefix_token_ids"]
        if not raw or any(type(t) is not int or t < 0 for t in raw) or token_hash(raw) != prefix["prefix_token_sha256"]:
            raise ValueError("Perturbed history raw token checksum failed")
        if prefix["sample_id"] != row["sample_id"] or prefix["search_turn"] != index+1:
            raise ValueError("Perturbed history ID/turn failed")
        kind, query = parse_action(step["text"])
        if kind != "search" or query != prefix["search_query"]:
            raise ValueError("Perturbed query/action mismatch")
        text += step["text"]
        generated += len(step["token_ids"])
        if text != prefix["prefix_text"] or generated != prefix["generated_tokens_before_observation"]:
            raise ValueError("Perturbed text/token-cost reconstruction failed")
        if tokens is not None:
            if step["input_tokens"] != len(tokens):
                raise ValueError("Perturbed model input count failed")
            tokens += step["token_ids"]  # Never re-tokenize generated text.
            if tokens != raw:
                raise ValueError("Perturbed raw prefix does not reconstruct with the frozen tokenizer")
        docs = prefix["visible_documents"]
        if len(docs) != retrieval["visible_top_k"] or prefix["observation_text"] != observation(docs) or prefix["observed_document_ids"] != [d["document_id"] for d in docs]:
            raise ValueError("Perturbed visible observation mismatch")
        if index != parent_turn-1 and (docs != prefix["candidate_documents"][:retrieval["visible_top_k"]] or prefix["original_document_ids"] != prefix["observed_document_ids"]):
            raise ValueError("More than the parent observation was intervened")
        text += prefix["observation_text"]
        if tokens is not None:
            tokens += generator.encode(prefix["observation_text"])
    if len(row["steps"]) <= turn:
        raise ValueError("No original generation after this observation; choose a real post-evidence opportunity")
    if tokens is not None and len(tokens) != row["steps"][turn]["input_tokens"]:
        raise ValueError("Restored post-observation length differs from original next generation")
    if tokens is not None and len(row["prefixes"]) > turn:
        later = row["prefixes"][turn]["prefix_token_ids"]
        if later != tokens + row["steps"][turn]["token_ids"]:
            raise ValueError("Restored input disagrees with the subsequent saved raw prefix")
    return text, tokens, {"resume_after_turn": turn, "successful_searches_already_spent": turn,
        "remaining_searches": agent["max_search_turns"]-turn,
        "prefix_before_observation_sha256": row["prefixes"][turn-1]["prefix_token_sha256"],
        "observation_sha256": digest(row["prefixes"][turn-1]["observation_text"].encode()),
        "post_observation_token_sha256": token_hash(tokens) if tokens is not None else None,
        "post_observation_input_tokens": row["steps"][turn]["input_tokens"],
        "post_observation_tokens_basis": "saved raw pre-observation IDs + same frozen tokenizer observation encoding; original next input count checked; next saved prefix checked where available"}


def continue_from_observation(row, clean, parent_turn, turn, name, generator, retriever, agent, retrieval):
    if name not in PROMPTS:
        raise ValueError("Unknown E arm")
    text, tokens, anchor = restore_after_observation(row, clean, parent_turn, turn, agent, retrieval, generator)
    instruction = PROMPTS[name]
    added = generator.encode(instruction) if instruction else []
    record = {key: copy.deepcopy(row[key]) for key in ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts", "initial_prompt", "condition", "prompt_version", "intervention_metadata")}
    generated = sum(len(s["token_ids"]) for s in row["steps"][:turn])
    prior_inputs = sum(s["input_tokens"] for s in row["steps"][:turn])
    record.update(branch=name, target_turn=turn, parent_target_turn=parent_turn,
        status=None, final_answer="", search_calls=turn, generated_tokens=generated, model_input_tokens=prior_inputs,
        steps=copy.deepcopy(row["steps"][:turn]), prefixes=copy.deepcopy(row["prefixes"][:turn]),
        e_metadata={"anchor": anchor, "instruction": instruction, "instruction_token_ids": added,
                    "instruction_tokens": len(added), "input_after_instruction_sha256": token_hash(tokens+added),
                    "maximum_future_generation_segments": agent["max_search_turns"]-turn+1})
    record = continue_record(record, generator, retriever, agent, retrieval, text+instruction, tokens+added)
    record.update(search_calls_after_anchor=record["search_calls"]-turn,
                  generated_tokens_after_anchor=record["generated_tokens"]-generated,
                  model_input_tokens_after_anchor=record["model_input_tokens"]-prior_inputs)
    return record

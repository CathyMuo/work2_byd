"""Bounded clean Search-R1-style rollouts with immutable pre-observation prefixes."""
import re
import time

from .common import digest
from .metrics import answer_scores, lexical_fact_hits
from .retrieval import format_documents

PROMPT_VERSION = "work2_search_r1_bounded_v1"


def question_prompt(question, max_search_turns):
    return ("Answer the given question. You must conduct reasoning inside <think> and </think> "
            "first every time you get new information. After reasoning, if you find you lack "
            "some knowledge, you can call a search engine by <search> query </search> and it "
            "will return the top searched results between <information> and </information>. "
            "You may make at most {} search calls. If you find no further external knowledge "
            "needed, provide the answer inside <answer> and </answer>, without detailed "
            "illustrations. For example, <answer> Beijing </answer>. Question: {}\n"
            .format(max_search_turns, question))


def parse_action(text):
    matches = list(re.finditer(r"<(search|answer)>(.*?)</\1>", text, re.DOTALL))
    if len(matches) != 1 or not matches[0].group(2).strip():
        return None, None
    # Only a single complete action is allowed in a generation segment.
    match = matches[0]
    if re.search(r"<(search|answer)>", text[:match.start()] + text[match.end():]):
        return None, None
    return match.group(1), match.group(2).strip()


def run_sample(sample, generator, retriever, agent, retrieval):
    start = time.monotonic()
    record = {**sample, "condition": "E0_clean", "prompt_version": PROMPT_VERSION,
              "status": None, "final_answer": "", "search_calls": 0,
              "generated_tokens": 0, "model_input_tokens": 0, "steps": [], "prefixes": []}
    text = generator.initial_prompt(question_prompt(sample["question"], agent["max_search_turns"]))
    tokens = generator.encode(text)
    record["initial_prompt"] = text
    return continue_record(record, generator, retriever, agent, retrieval, text, tokens, start)


def continue_record(record, generator, retriever, agent, retrieval, text, tokens, start=None):
    """Shared A1/A2 continuation; preserves the already-spent global search budget."""
    start = time.monotonic() if start is None else start
    tokens = list(tokens)
    try:
        # Four successful calls permit a fifth generation to answer; a fifth call is refused.
        for _ in range(agent["max_search_turns"] - record["search_calls"] + 1):
            if len(tokens) >= agent["max_context_tokens"]:
                record["status"] = "context_limit"
                break
            allowed = min(agent["max_generation_tokens_per_turn"], agent["max_context_tokens"] - len(tokens))
            generated = generator.generate(tokens, allowed, agent["seed"])
            step = {"input_tokens": len(tokens), **generated}
            record["model_input_tokens"] += len(tokens)
            record["generated_tokens"] += len(generated["token_ids"])
            record["steps"].append(step)
            text += generated["text"]
            tokens += generated["token_ids"]
            kind, value = parse_action(generated["text"])
            if kind == "answer":
                record["final_answer"], record["status"] = value, "answered"
                break
            if kind != "search":
                record["status"] = ("generation_limit" if generated["finish_reason"] == "length" else "malformed_action")
                break
            if record["search_calls"] >= agent["max_search_turns"]:
                record["status"] = "search_budget_exhausted"
                break
            turn = record["search_calls"] + 1
            prefix = {"sample_id": record["sample_id"], "search_turn": turn,
                      "search_query": value, "prefix_text": text, "prefix_token_ids": list(tokens),
                      "prefix_token_sha256": digest(
                          (",".join(map(str, tokens))).encode()),
                      "generated_tokens_before_observation": record["generated_tokens"]}
            # Store the prefix even if the local service fails.
            record["prefixes"].append(prefix)
            record["search_calls"] += 1
            candidates = retriever.search(value, retrieval["candidate_top_k"])
            visible = candidates[:retrieval["visible_top_k"]]
            if len(visible) < retrieval["visible_top_k"]:
                raise ValueError("Retriever returned fewer than visible_top_k documents")
            observation = "\n\n<information>{}</information>\n\n".format(format_documents(visible))
            prefix.update({"candidate_documents": candidates, "visible_documents": visible,
                           "original_document_ids": [d["document_id"] for d in visible],
                           "observed_document_ids": [d["document_id"] for d in visible],
                           "observation_text": observation,
                           "visible_lexical_fact_ids": lexical_fact_hits(visible, record["gold_supporting_facts"]),
                           "candidate_lexical_fact_ids": lexical_fact_hits(candidates, record["gold_supporting_facts"])})
            text += observation
            tokens += generator.encode(observation)
        record["status"] = record["status"] or "search_budget_exhausted"
    except Exception as exc:
        record["status"] = "infrastructure_error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["elapsed_seconds"] = time.monotonic() - start
    record["final_prompt_token_count"] = len(tokens)
    record["repeated_queries"] = len(record["prefixes"]) - len(set(
        " ".join(p["search_query"].casefold().split()) for p in record["prefixes"]))
    if record["status"] == "infrastructure_error":
        record.update({"em": None, "f1": None})
    elif record["status"] == "answered":
        record.update(answer_scores(record["final_answer"], record["gold_answer"]))
    else:
        record.update({"em": 0, "f1": 0.0})
    return record

"""Bounded clean Search-R1-style rollouts with immutable pre-observation prefixes."""
import re
import time

from .common import digest
from .metrics import answer_scores, lexical_fact_hits
from .retrieval import format_documents

PROMPT_VERSION = "work2_search_r1_bounded_v1"
# FC1 (2026-10-20): verbatim Search-R1 base template (third_party/Search-R1 scripts/data_process/
# qa_search_test_merge.py make_prefix), used only for the harness-fidelity check.
OFFICIAL_PROMPT_VERSION = "search_r1_official_base_v1"
OFFICIAL_TEMPLATE = ("Answer the given question. You must conduct reasoning inside <think> and </think> first every time "
                     "you get new information. After reasoning, if you find you lack some knowledge, you can call a search "
                     "engine by <search> query </search> and it will return the top searched results between <information> "
                     "and </information>. You can search as many times as your want. If you find no further external "
                     "knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed "
                     "illustrations. For example, <answer> Beijing </answer>. Question: {}\n")

# U2 (2026-10-23): ReSeek (TencentBAC/ReSeek utils/preprocess_reseek_dataset.py) system message and user prefix,
# extracted verbatim with ast (indentation inside the triple-quoted string included); user = prefix.rstrip("\n") + question.
RESEEK_PROMPT_VERSION = "reseek_official_v1"
RESEEK_SYSTEM = "You are a helpful and harmless assistant."
RESEEK_USER_PREFIX = (
    "Answer the given question step by step.\n    Instructions:\n"
    "    1. First, conduct reasoning inside <think> and </think> tags whenever you receive new information.\n"
    "    2. If you need external knowledge, you can search using <search>query</search>.\n"
    "    3. When you receive search results, evaluate their usefulness and put your judgment inside <judge>Yes</judge> or <judge>No</judge> tags.\n"
    "    4. Based on your judgment, follow these strict rules:\n"
    "    a. If the information is useful AND you now have sufficient information to provide a complete final answer, proceed directly to step 5.\n"
    "    b. If the information is useful BUT you still need more details, you MUST search again with <search>...</search>.\n"
    "    c. If the information is not useful, you MUST search again with <search>...</search>. You MUST NOT provide an answer in this case.\n"
    "    5. Provide your final answer in <answer>...</answer> tags. The <answer> tag marks the end of the task. After providing the <answer>, you MUST stop and generate no further text.\n"
    "\n    Question: ")
SYSTEM_OF = {"reseek": RESEEK_SYSTEM}

# U1x (2026-10-27): agents from other training recipes, prompts read verbatim with ast from their repos.
# AutoRefine (syr-cn/AutoRefine preprocess/data_process/utils.py, template_type "autorefine");
# StepSearch (Zillwang/StepSearch scripts/data_process/{nq,hotpot}_search.py, the effective second prefix = paper Table 4).
# "{question}" is replaced literally (no str.format), so braces inside questions are safe.
AUTOREFINE_PROMPT_VERSION = "autorefine_official_v1"
STEPSEARCH_PROMPT_VERSION = "stepsearch_official_v1"
AUTOREFINE_TEMPLATE = (
    'You are a helpful assistant excel at answering questions with multi-turn search engine calling. To answer questions, you must first reason through the available information using <think> and </think>. If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with the three most relevant documents enclosed in <documents> and </documents>. After each search, you need to summarize and refine the existing documents in <refine> and </refine>. You may send multiple search requests if needed. Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>. Question: {question}\n')
STEPSEARCH_TEMPLATE = (
    '## Background\n'
    "You are a deep AI research assistant. I will give you a single-hop or multi-hop question. You don't have to answer the question now, but you should first think about your research plan or what to search for next. You can use search to fill in knowledge gaps. \n"
    '## Response format: Your output format should be one of the following two formats: \n'
    '<think>your thinking process</think>\n'
    '<answer>your answer after getting enough information</answer>\n'
    'or\n'
    '<think>your thinking process</think>\n'
    'use <search>search keywords</search> to search for information. For example, <think> plan to search: (Q1) (Q2) (Q3) ... </think> <search> (Q1) question </search> <think> reasoning ... </think> <answer> Beijing </answer>.\n'
    'The search engine will return the results contained in <information> and </information>. \n'
    'Please follow the loop of think, search, information, think, search, information, and answer until the original question is finally solved. \n'
    'Note: The retrieval results may not contain the answer or contain noise. You need to tell whether there is a golden answer. If not, you need to correct the search query and search again. Question:{question}\n')
# Observation rendering per prompt variant. Search-R1 / AutoRefine / StepSearch strip the document block;
# bounded / official / reseek keep the historical unstripped block (every earlier run used it).
FORMATS = {
    "bounded": {"version": PROMPT_VERSION, "tag": "information", "pad": "", "strip": False, "docs": "searchr1"},
    "official": {"version": OFFICIAL_PROMPT_VERSION, "tag": "information", "pad": "", "strip": False, "docs": "searchr1"},
    "reseek": {"version": RESEEK_PROMPT_VERSION, "tag": "information", "pad": "", "strip": False, "docs": "searchr1"},
    "autorefine": {"version": AUTOREFINE_PROMPT_VERSION, "tag": "documents", "pad": "", "strip": True, "docs": "searchr1"},
    "stepsearch": {"version": STEPSEARCH_PROMPT_VERSION, "tag": "information", "pad": " ", "strip": True, "docs": "stepsearch"},
}
VARIANT_OF_VERSION = {f["version"]: v for v, f in FORMATS.items()}


def observation_text(documents, variant="bounded"):
    f = FORMATS[variant]
    body = format_documents(documents, f["docs"])
    body = body.strip() if f["strip"] else body
    return "\n\n<{0}>{1}{2}{1}</{0}>\n\n".format(f["tag"], f["pad"], body)


def record_variant(record):
    return VARIANT_OF_VERSION.get(record.get("prompt_version"), "bounded")


def question_prompt(question, max_search_turns, variant="bounded"):
    if variant == "official":
        return OFFICIAL_TEMPLATE.format(question)
    if variant == "reseek":
        return RESEEK_USER_PREFIX.rstrip("\n") + question
    if variant == "autorefine":
        return AUTOREFINE_TEMPLATE.replace("{question}", question)
    if variant == "stepsearch":
        return STEPSEARCH_TEMPLATE.replace("{question}", question)
    if variant != "bounded":
        raise ValueError("Unknown prompt variant: " + str(variant))
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
    variant = agent.get("prompt_variant", "bounded")
    record = {**sample, "condition": "E0_clean",
              "prompt_version": FORMATS[variant]["version"],
              "status": None, "final_answer": "", "search_calls": 0,
              "generated_tokens": 0, "model_input_tokens": 0, "steps": [], "prefixes": []}
    text = generator.initial_prompt(question_prompt(sample["question"], agent["max_search_turns"], variant))
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
            observation = observation_text(visible, record_variant(record))
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

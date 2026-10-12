"""SI1: observable natural states and one intervention relative to a frozen policy.

No gold, candidate-only content, future trace or branch outcomes enter state features.
The old perturbed E protocol is deliberately not used or relaxed.
"""
import collections
import copy
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

from .common import file_hash, load_json, local_path, write_json
from .datasets import canonicalize, read_rows
from .metrics import answer_scores
from .replay import token_hash, validate_prefix
from .rollout import continue_record, parse_action

VERSION = "SI1_natural_prefix_gain_harm_v1"
ARMS = ("keep", "budget", "review", "answer")
PROMPTS = {
    "keep": "",
    "budget": "\n\nSearch budget: used={used}; remaining={remaining}; total={total}. Continue using the original <search> or <answer> protocol.\n\n",
    "review": "\n\nBefore continuing, briefly review the original question and the information already shown. Check your reasoning and decide whether to answer or search within the remaining original budget. Use the original <search> or <answer> protocol.\n\n",
    "answer": "\n\nUsing only the original question and the information already shown, reason briefly and provide your best final answer inside <answer> and </answer>. Do not request another search.\n\n",
}
FEATURE_VERSION = "prefix_numeric_and_signed_hash64_v1"
EXACT_FIELDS = ("status", "final_answer", "em", "f1", "search_calls", "generated_tokens",
                "model_input_tokens", "steps", "prefixes", "final_prompt_token_count", "repeated_queries")


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def question_key(text):
    return " ".join(text.casefold().split())


def jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def exclusions(paths):
    """Require actual historical canonical samples; IDs AND question text are blocked."""
    blocked_ids, blocked_questions, receipts = set(), set(), []
    for path in paths:
        rows = read_rows(path)
        if not rows:
            raise ValueError("Historical sample file is empty: " + str(path))
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or not isinstance(row.get("question"), str):
                raise ValueError("Exclusion must be canonical sample JSON/JSONL, not summary/manifest: " + str(path))
            if row.get("dataset") not in ("hotpotqa", "2wiki"):
                raise ValueError("Unknown historical dataset")
            blocked_ids.add(row["sample_id"])
            blocked_questions.add(question_key(row["question"]))
        receipts.append({"path": str(Path(path).resolve()), "sha256": file_hash(path), "rows": len(rows)})
    return blocked_ids, blocked_questions, receipts


def prepare(local, destination, excluded, train_per_dataset=80, validation_per_dataset=40, seed=20261012):
    destination = Path(destination)
    if destination.exists() or not excluded or min(train_per_dataset, validation_per_dataset) < 1:
        raise ValueError("Need a new destination, explicit historical samples, and positive split sizes")
    ids, questions, receipts = exclusions(excluded)
    train, validation, sources = [], [], []
    selected_questions = set()
    for dataset, key in (("hotpotqa", "hotpotqa_file"), ("2wiki", "two_wiki_file")):
        path = local_path(local[key])
        rows = [canonicalize(row, dataset) for row in read_rows(path)]
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate source ID: " + dataset)
        if len({question_key(r["question"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate source question: " + dataset)
        eligible = [r for r in rows if r["sample_id"] not in ids and question_key(r["question"]) not in questions]
        eligible.sort(key=lambda r: r["sample_id"])
        random.Random(str(seed) + ":" + dataset).shuffle(eligible)
        # Deterministically skip cross-dataset duplicate questions, before seeing outcomes.
        chosen = []
        for row in eligible:
            q = question_key(row["question"])
            if q in selected_questions:
                continue
            chosen.append(row)
            selected_questions.add(q)
            if len(chosen) == train_per_dataset + validation_per_dataset:
                break
        if len(chosen) != train_per_dataset + validation_per_dataset:
            raise ValueError("Not enough unobserved unique questions in " + dataset)
        train.extend(dict(r, split="train") for r in chosen[:train_per_dataset])
        validation.extend(dict(r, split="validation") for r in chosen[train_per_dataset:])
        sources.append({"dataset": dataset, "sha256": file_hash(path), "rows": len(rows),
                        "eligible_before_cross_dataset_dedup": len(eligible)})
    samples = train + validation
    destination.mkdir(parents=True)
    jsonl(destination / "samples.jsonl", samples)
    smoke = sum(([r for r in train if r["dataset"] == d][:10] for d in ("hotpotqa", "2wiki")), [])
    jsonl(destination / "smoke.jsonl", smoke)
    manifest = {"version": VERSION, "seed": seed, "exclusions": receipts, "sources": sources,
                "historical_unique_ids": len(ids), "historical_unique_questions": len(questions),
                "train_per_dataset": train_per_dataset, "validation_per_dataset": validation_per_dataset,
                "sample_ids": [r["sample_id"] for r in samples],
                "splits": {r["sample_id"]: r["split"] for r in samples},
                "samples_sha256": file_hash(destination / "samples.jsonl"),
                "smoke_sha256": file_hash(destination / "smoke.jsonl"),
                "smoke_ids": [r["sample_id"] for r in smoke],
                "exclusion_scope": "Only the supplied historical files are proven excluded; caller must supply ALL observed cohorts."}
    write_json(destination / "selection_manifest.json", manifest)
    return manifest


def load_samples(folder, smoke=False):
    folder = Path(folder)
    m = load_json(folder / "selection_manifest.json")
    name = "smoke" if smoke else "samples"
    if m["version"] != VERSION or file_hash(folder / (name + ".jsonl")) != m[name + "_sha256"]:
        raise ValueError("Frozen sample hash/version mismatch")
    rows = read_rows(folder / (name + ".jsonl"))
    expected = m["smoke_ids"] if smoke else m["sample_ids"]
    if [r["sample_id"] for r in rows] != expected or len(set(expected)) != len(expected):
        raise ValueError("Frozen coverage/order mismatch")
    if any(r["split"] != m["splits"][r["sample_id"]] for r in rows):
        raise ValueError("Frozen split mismatch")
    return rows, m


def observable(source, agent, retrieval, generator=None):
    """Construct an allowlisted state before reading ANY continuation outcome."""
    if not source["prefixes"]:
        return {"version": VERSION, "sample_id": source["sample_id"], "dataset": source["dataset"],
                "question": source["question"], "reachable": False, "history_text": "",
                "visible_documents": [], "queries": [], "spent_calls": 0,
                "remaining_calls": agent["max_search_turns"], "context_tokens": 0}
    prefix = validate_prefix(source, 1, agent, retrieval, generator)
    text = prefix["prefix_text"] + prefix["observation_text"]
    actionable = len(source["steps"]) > 1
    if not actionable and source["status"] != "context_limit":
        raise ValueError("Missing generation opportunity without an explained context limit")
    # Only observed documents; never use unpresented top-20, gold or suffix.
    return {"version": VERSION, "sample_id": source["sample_id"], "dataset": source["dataset"],
            "question": source["question"], "reachable": actionable,
            "history_text": text,
            "visible_documents": [{"document_id": d["document_id"], "contents": d["contents"]}
                                  for d in prefix["visible_documents"]],
            "queries": [prefix["search_query"]], "spent_calls": 1,
            "remaining_calls": agent["max_search_turns"] - 1,
            "context_tokens": (len(prefix["prefix_token_ids"]) + len(generator.encode(prefix["observation_text"])) if generator else
                               (source["steps"][1]["input_tokens"] if actionable else source["final_prompt_token_count"])),
            "pre_observation_sha256": prefix["prefix_token_sha256"],
            "observation_sha256": sha(prefix["observation_text"])}


def exact(source, replay):
    return all(source.get(k) == replay.get(k) for k in EXACT_FIELDS)


def continue_arm(source, arm, generator, retriever, agent, retrieval):
    if arm not in ARMS:
        raise ValueError("Unknown SI1 arm")
    if not source["prefixes"] or (len(source["steps"]) == 1 and source["status"] == "context_limit"):
        result = copy.deepcopy(source)
        result["si1"] = {"arm": arm, "reachable": False, "incremental_instruction_tokens": 0,
                         "search_calls_after_anchor": 0, "generated_tokens_after_anchor": 0,
                         "model_input_tokens_after_anchor": 0, "elapsed_after_anchor_seconds": 0}
        return result
    prefix = validate_prefix(source, 1, agent, retrieval, generator)
    text = prefix["prefix_text"] + prefix["observation_text"]
    tokens = list(prefix["prefix_token_ids"]) + generator.encode(prefix["observation_text"])
    if len(source["steps"]) <= 1:
        raise ValueError("Source has no post-first-observation generation opportunity")
    if len(tokens) != source["steps"][1]["input_tokens"]:
        raise ValueError("Post-observation input identity mismatch")
    if len(source["prefixes"]) > 1 and tokens + source["steps"][1]["token_ids"] != source["prefixes"][1]["prefix_token_ids"]:
        raise ValueError("Source next raw prefix identity mismatch")
    base_generated = len(source["steps"][0]["token_ids"])
    base_input = source["steps"][0]["input_tokens"]
    record = {k: copy.deepcopy(source[k]) for k in ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts", "initial_prompt", "condition", "prompt_version")}
    record.update(status=None, final_answer="", search_calls=1, generated_tokens=base_generated,
                  model_input_tokens=base_input, steps=copy.deepcopy(source["steps"][:1]),
                  prefixes=copy.deepcopy(source["prefixes"][:1]))
    instruction = PROMPTS[arm].format(used=1, remaining=agent["max_search_turns"]-1, total=agent["max_search_turns"])
    added = generator.encode(instruction)
    anchor_hash = token_hash(tokens)
    tokens += added
    started = time.monotonic()
    if arm != "answer":
        record = continue_record(record, generator, retriever, agent, retrieval, text + instruction, tokens)
    else:
        # Answer-only changes the action space; a requested search is failed, never executed.
        if len(tokens) >= agent["max_context_tokens"]:
            record.update(status="context_limit", em=0, f1=0.0, final_prompt_token_count=len(tokens), repeated_queries=0)
        else:
            allowed = min(agent["max_generation_tokens_per_turn"], agent["max_context_tokens"]-len(tokens))
            out = generator.generate(tokens, allowed, agent["seed"])
            record["steps"].append(dict(out, input_tokens=len(tokens)))
            record["generated_tokens"] += len(out["token_ids"])
            record["model_input_tokens"] += len(tokens)
            kind, value = parse_action(out["text"])
            record["final_prompt_token_count"] = len(tokens) + len(out["token_ids"])
            record["repeated_queries"] = 0
            if kind == "answer":
                record.update(status="answered", final_answer=value, **answer_scores(value, source["gold_answer"]))
            else:
                record.update(status="answer_only_search_refused" if kind == "search" else
                              ("generation_limit" if out["finish_reason"] == "length" else "malformed_action"), em=0, f1=0.0)
    if record["status"] == "infrastructure_error":
        raise RuntimeError("Invalid infrastructure continuation: " + str(record.get("error")))
    record["si1"] = {"arm": arm, "reachable": True, "post_observation_token_sha256": anchor_hash,
                     "instruction": instruction, "incremental_instruction_tokens": len(added),
                     "original_total_search_budget": agent["max_search_turns"], "spent_at_anchor": 1,
                     "remaining_at_anchor": agent["max_search_turns"]-1,
                     "action_space": "answer_only" if arm == "answer" else "search_or_answer",
                     "search_calls_after_anchor": record["search_calls"]-1,
                     "generated_tokens_after_anchor": record["generated_tokens"]-base_generated,
                     "model_input_tokens_after_anchor": record["model_input_tokens"]-base_input,
                     "elapsed_after_anchor_seconds": time.monotonic()-started}
    return record


def features(state):
    words = lambda t: re.findall(r"\w+", t.casefold())
    q = words(state["question"])
    docs = state["visible_documents"]
    content = "\n".join(d["contents"] for d in docs)
    dw = words(content)
    qw = words(" ".join(state["queries"]))
    overlap = lambda a,b: len(set(a) & set(b))/max(1, len(set(a)))
    x = [1.0, float(state["reachable"]), state["spent_calls"]/4,
         state["remaining_calls"]/4, min(state["context_tokens"]/8192, 1),
         min(len(q)/60,1), min(len(dw)/2000,1), len(docs)/3,
         overlap(q,dw), overlap(q,qw), overlap(qw,dw),
         len({d["document_id"] for d in docs})/max(1,len(docs))]
    for markers in ({"who"},{"where"},{"when"},{"which"},{"how","many","much"},
                    {"earlier","later","older","younger","both","same","different"},
                    {"before","after","year","first","last"}):
        x.append(float(bool(set(q) & markers)))
    buckets = [0.0]*64
    # Content/question/query vocab are observable. No outcome-dependent feature extraction.
    for label, seq in (("q",q),("d",dw),("s",qw)):
        counts = collections.Counter(seq)
        for word, count in counts.items():
            h = hashlib.sha256((label+":"+word).encode()).digest()
            buckets[int.from_bytes(h[:2],"big") % 64] += (1 if h[2]&1 else -1)*math.log1p(count)
    norm = math.sqrt(sum(v*v for v in buckets)) or 1
    return x + [v/norm for v in buckets]


def fit(cases, ridge=5.0):
    """Multi-output ridge predicts gain/harm separately; it is an exploratory selector."""
    if not cases or any(c["split"] != "train" for c in cases):
        raise ValueError("Selector fitting requires ONLY nonempty train rows")
    cases = [c for c in cases if c["state"]["reachable"]]
    if not cases:
        raise ValueError("No reachable training states")
    d = len(features(cases[0]["state"]))
    nout = 9  # gain/harm/relative-call-cost for each of three interventions
    matrix = [[0.0]*(d+nout) for _ in range(d)]
    for c in cases:
        x = features(c["state"])
        base = c["arms"]["keep"]
        y = []
        for arm in ARMS[1:]:
            row = c["arms"][arm]
            y += [float(base["em"]==0 and row["em"]==1), float(base["em"]==1 and row["em"]==0),
                  (row["search_calls"]-base["search_calls"])/4]
        for i, v in enumerate(x):
            for j,w in enumerate(x):
                matrix[i][j] += v*w
            for j,w in enumerate(y):
                matrix[i][d+j] += v*w
    for i in range(d):
        matrix[i][i] += ridge if i else 0.01
    # Pivoted Gaussian elimination once for all nine targets; no new library dependency.
    for i in range(d):
        pivot = max(range(i,d),key=lambda k:abs(matrix[k][i]))
        matrix[i],matrix[pivot]=matrix[pivot],matrix[i]
        scale=matrix[i][i]
        if abs(scale)<1e-12: raise ValueError("Singular selector system")
        matrix[i]=[v/scale for v in matrix[i]]
        for k in range(d):
            if k==i:continue
            f=matrix[k][i]
            if f:matrix[k]=[a-f*b for a,b in zip(matrix[k],matrix[i])]
    return {"version":VERSION,"feature_version":FEATURE_VERSION,"ridge":ridge,
            "weights":[r[d:] for r in matrix],"train_ids":[c["sample_id"] for c in cases],
            "train_question_keys":[question_key(c["state"]["question"]) for c in cases],
            "trained_rows":len(cases),"target_order":"per arm: gain, harm, relative calls / 4",
            "warning":"Clipped ridge outputs are exploratory scores, not calibrated probabilities or safety guarantees."}


def choose(model, state, risk_weight=2.0, margin=0.02, call_penalty=0.01):
    if model["version"]!=VERSION or model["feature_version"]!=FEATURE_VERSION:
        raise ValueError("Selector version mismatch")
    if not state["reachable"]:
        return "keep", {}
    x=features(state)
    if len(x)!=len(model["weights"]): raise ValueError("Selector dimension mismatch")
    pred=[sum(v*w[j] for v,w in zip(x,model["weights"])) for j in range(9)]
    scores={}
    for j,arm in enumerate(ARMS[1:]):
        gain=max(0,min(1,pred[3*j]));harm=max(0,min(1,pred[3*j+1]))
        scores[arm]={"gain_score":gain,"harm_score":harm,"delta_calls_score":4*pred[3*j+2],
                     "utility":gain-risk_weight*harm-call_penalty*4*pred[3*j+2]}
    arm=max(ARMS[1:],key=lambda a:scores[a]["utility"])
    return (arm if scores[arm]["utility"]>margin else "keep"),scores


def arm_summary(rows, base):
    n=len(rows)
    if not n:raise ValueError("Empty evaluation")
    return {"n":n,"em":sum(r["em"] for r in rows)/n,"f1":sum(r["f1"] for r in rows)/n,
            "gain":sum(b["em"]==0 and r["em"]==1 for r,b in zip(rows,base)),
            "harm":sum(b["em"]==1 and r["em"]==0 for r,b in zip(rows,base)),
            "net_em":sum(r["em"]-b["em"] for r,b in zip(rows,base))/n,
            "baseline_correct_n":sum(b["em"]==1 for b in base),
            "conditional_harm":(sum(b["em"]==1 and r["em"]==0 for r,b in zip(rows,base))/sum(b["em"]==1 for b in base)
                                if any(b["em"]==1 for b in base) else None),
            **{k:sum(r[k] for r in rows)/n for k in ("search_calls","generated_tokens","model_input_tokens")},
            "mean_elapsed_after_anchor_seconds":sum(r.get("si1",{}).get("elapsed_after_anchor_seconds",0) for r in rows)/n,
            "statuses":dict(collections.Counter(r["status"] for r in rows))}


def bootstrap(deltas, seed=20261012, repeats=2000):
    if not deltas:return [None,None]
    rng=random.Random(seed);n=len(deltas)
    estimates=sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n for _ in range(repeats))
    return [estimates[int(.025*repeats)],estimates[int(.975*repeats)]]


def summarize(cases, decisions=None):
    if not cases or len({c["sample_id"] for c in cases})!=len(cases):
        raise ValueError("Empty or duplicate question cases")
    base=[c["arms"]["keep"] for c in cases]
    arms={a:arm_summary([c["arms"][a] for c in cases],base) for a in ARMS}
    best=max(ARMS,key=lambda a:arms[a]["em"])
    oracle=sum(max(c["arms"][a]["em"] for a in ARMS) for c in cases)/len(cases)
    result={"version":VERSION,"n":len(cases),"reachable":sum(c["state"]["reachable"] for c in cases),
            "arms":arms,"best_fixed_em_arm":best,"finite_action_oracle_em":oracle,
            "oracle_headroom_vs_best_fixed_em":oracle-arms[best]["em"],
            "oracle_note":"Outcome-selected finite-action diagnostic, never a deployable result or theoretical upper bound."}
    if decisions is not None:
        ids=[c["sample_id"] for c in cases]
        if set(decisions)!=set(ids):raise ValueError("Decision coverage differs from evaluation questions")
        selected=[c["arms"][decisions[c["sample_id"]]] for c in cases]
        result["selector"]=arm_summary(selected,base)
        result["selector"]["selection_counts"]=dict(collections.Counter(decisions.values()))
        result["selector"]["net_em_ci95_vs_keep"]=bootstrap([r["em"]-b["em"] for r,b in zip(selected,base)])
        result["selector"]["net_em_vs_best_fixed"]=sum(r["em"]-c["arms"][best]["em"] for r,c in zip(selected,cases))/len(cases)
        result["selector"]["comparison_note"]="Best fixed EM arm selected on this evaluation set is a conservative diagnostic competitor; final baseline must be frozen from train."
    return result

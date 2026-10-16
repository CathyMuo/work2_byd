"""CT1: on-policy interventional ground truth for turn-level credit in search agents.

For each question: sample one scored trajectory tau, then estimate by batched
re-execution from the SAME token prefixes
  V_j   = V(s_j)            state after observation j (s_0 = initial prompt)
  N_j   = V(s_j with observation j replaced by a semantic-null observation)
  CB    = V(s_0) when EVERY retrieval returns the null observation
so A_t = V_t - V_{t-1} and E_t = V_t - N_t (analysed in credit_analysis).
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import file_hash, git_output, load_json, local_path, write_json
from .datasets import read_rows
from .metrics import answer_scores, lexical_fact_hits
from .pcr import gpu_list, rank_key, stop
from .placebo import contains_answer
from .replay import observation, visible_signature
from .rollout import PROMPT_VERSION, parse_action, question_prompt
from .run_baseline import asset_identity, validate_design
from .selective import sha

VERSION = "CT1_interventional_credit_pilot_v1"


def seed_for(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def load_protocol(path):
    p = load_json(path)
    if p.get("protocol") != VERSION:
        raise ValueError("Unsupported CT1 protocol")
    validate_design(dict(p, agent=dict(p["agent"], temperature=0)))
    s = p["sampling"]
    if not 0 < s["temperature"] <= 2 or not 0 < s["top_p"] <= 1 or s["k"] < 2 or s["k"] % 2:
        raise ValueError("Invalid sampling settings (k must be even for split-half reliability)")
    return p


class LiveSearcher:
    """Batched live retrieval with same-query visible-result drift checks."""
    def __init__(self, retriever, retrieval):
        self.retriever, self.retrieval = retriever, retrieval
        self.signatures, self.calls, self.drift = {}, 0, 0

    def search_many(self, queries):
        unique = list(dict.fromkeys(queries))
        if not unique:
            return {}
        groups = self.retriever.search_batch(unique, self.retrieval["candidate_top_k"])
        out = {}
        for query, docs in zip(unique, groups):
            if len(docs) < self.retrieval["visible_top_k"]:
                raise ValueError("Retriever returned fewer than visible_top_k documents")
            signature = visible_signature(docs[:self.retrieval["visible_top_k"]])
            if self.signatures.setdefault(query, signature) != signature:
                self.drift += 1
                raise RuntimeError("Live retriever drift for query: " + query)
            out[query] = docs
        self.calls += len(unique)
        return out


def new_rollout(sample, tokens, calls, seed, mode="live", null_docs=None, track=False, segments=None):
    return {"sample": sample, "tokens": list(tokens), "search_calls": calls, "seed": seed, "mode": mode,
            "null_docs": null_docs, "track": track, "segments": list(segments or []),
            "steps_left": None, "status": None, "final_answer": "", "generated": 0, "queries": [],
            "turns": [], "texts": []}


def batched_continue(rollouts, generator, searcher, agent, retrieval, temperature, top_p):
    """Lockstep version of rollout.continue_record for many independent branches.

    Same action parsing, global search budget (spent calls carried over) and
    terminal statuses; generation is one batched sampling call per step.
    """
    for r in rollouts:
        r["steps_left"] = agent["max_search_turns"] - r["search_calls"] + 1
    step = 0
    while True:
        active = [r for r in rollouts if r["status"] is None]
        for r in active:
            if r["steps_left"] <= 0:
                r["status"] = "search_budget_exhausted"
            elif len(r["tokens"]) >= agent["max_context_tokens"]:
                r["status"] = "context_limit"
        active = [r for r in active if r["status"] is None]
        if not active:
            break
        allowed = [min(agent["max_generation_tokens_per_turn"], agent["max_context_tokens"] - len(r["tokens"])) for r in active]
        outs = generator.sample_batch([r["tokens"] for r in active], allowed,
                                      [(r["seed"] + 7919 * step) % (2 ** 31 - 1) for r in active], temperature, top_p)
        step += 1
        requests = []
        for r, out in zip(active, outs):
            r["steps_left"] -= 1
            start = len(r["tokens"])
            r["tokens"] += out["token_ids"]
            r["generated"] += len(out["token_ids"])
            if r["track"]:
                r["segments"].append({"kind": "gen", "start": start, "end": len(r["tokens"]), "after_turn": r["search_calls"]})
                r["texts"].append(out["text"])
            kind, value = parse_action(out["text"])
            if kind == "answer":
                r["status"], r["final_answer"] = "answered", value
            elif kind != "search":
                r["status"] = "generation_limit" if out["finish_reason"] == "length" else "malformed_action"
            elif r["search_calls"] >= agent["max_search_turns"]:
                r["status"] = "search_budget_exhausted"
            else:
                r["queries"].append(value)
                if r["mode"] == "live":
                    requests.append((r, value))
                else:
                    requests.append((r, None))
        docs = searcher.search_many([q for _, q in requests if q is not None])
        for r, query in requests:
            candidates = docs[query] if query is not None else r["null_docs"]
            visible = candidates[:retrieval["visible_top_k"]]
            r["search_calls"] += 1
            obs = generator.encode(observation(visible))
            start = len(r["tokens"])
            r["tokens"] += obs
            if r["track"]:
                facts = r["sample"]["gold_supporting_facts"]
                r["segments"].append({"kind": "obs", "start": start, "end": len(r["tokens"]), "turn": r["search_calls"]})
                r["turns"].append({"turn": r["search_calls"], "query": r["queries"][-1],
                                   "visible_ids": [d["document_id"] for d in visible],
                                   "fact_ids": lexical_fact_hits(visible, facts),
                                   "answer_hit": any(contains_answer(d, r["sample"]["gold_answer"]) for d in visible)})
    for r in rollouts:
        r.update(answer_scores(r["final_answer"], r["sample"]["gold_answer"]) if r["status"] == "answered" else {"em": 0, "f1": 0.0})
    return rollouts


def null_documents(searcher, decoy_question, sample, retrieval):
    docs = searcher.search_many([decoy_question])[decoy_question]
    keep = [d for d in docs if not contains_answer(d, sample["gold_answer"])
            and not lexical_fact_hits([d], sample["gold_supporting_facts"])]
    return keep[:retrieval["visible_top_k"]] if len(keep) >= retrieval["visible_top_k"] else None


def obs_end(trajectory, j):
    if j == 0:
        return next(s["end"] for s in trajectory["segments"] if s["kind"] == "prompt")
    return next(s["end"] for s in trajectory["segments"] if s["kind"] == "obs" and s["turn"] == j)


def query_end(trajectory, j):
    """End of the generation that issued search j (prefix before observation j)."""
    return next(s["start"] for s in trajectory["segments"] if s["kind"] == "obs" and s["turn"] == j)


def run_question(sample, decoy_question, generator, searcher, protocol):
    agent, retrieval, s = protocol["agent"], protocol["retrieval"], protocol["sampling"]
    k, temp, top_p = s["k"], s["temperature"], s["top_p"]
    started = time.monotonic()
    prompt = generator.encode(generator.initial_prompt(question_prompt(sample["question"], agent["max_search_turns"])))
    seed = seed_for(protocol["seed"], sample["sample_id"], "trajectory")
    traj = new_rollout(sample, prompt, 0, seed, track=True, segments=[{"kind": "prompt", "start": 0, "end": len(prompt)}])
    again = new_rollout(sample, prompt, 0, seed)
    batched_continue([traj], generator, searcher, agent, retrieval, temp, top_p)
    batched_continue([again], generator, searcher, agent, retrieval, temp, top_p)
    null = null_documents(searcher, decoy_question, sample, retrieval)
    T = traj["search_calls"]
    branches = []
    for j in range(T + 1):
        for i in range(k):
            branches.append(("V", j, i, new_rollout(sample, traj["tokens"][:obs_end(traj, j)], j, seed_for(protocol["seed"], sample["sample_id"], "V", j, i))))
    if null is not None:
        null_obs = generator.encode(observation(null))
        for j in range(1, T + 1):
            for i in range(k):
                branches.append(("N", j, i, new_rollout(sample, traj["tokens"][:query_end(traj, j)] + null_obs, j,
                                                        seed_for(protocol["seed"], sample["sample_id"], "N", j, i))))
        for i in range(k):
            branches.append(("C", 0, i, new_rollout(sample, prompt, 0, seed_for(protocol["seed"], sample["sample_id"], "C", i),
                                                    mode="null", null_docs=null)))
    batched_continue([b[3] for b in branches], generator, searcher, agent, retrieval, temp, top_p)
    values = {"V": [[None] * k for _ in range(T + 1)], "N": [[None] * k for _ in range(T)], "C": [None] * k}
    f1 = {"V": [[None] * k for _ in range(T + 1)], "N": [[None] * k for _ in range(T)], "C": [None] * k}
    statuses = {}
    for kind, j, i, r in branches:
        target = values[kind] if kind == "C" else values[kind][j if kind == "V" else j - 1]
        ftarget = f1[kind] if kind == "C" else f1[kind][j if kind == "V" else j - 1]
        target[i], ftarget[i] = r["em"], r["f1"]
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    if null is None:
        values["N"], f1["N"], values["C"], f1["C"] = None, None, None, None
    return {"version": VERSION, "sample_id": sample["sample_id"], "dataset": sample["dataset"],
            "question": sample["question"], "gold_answer": sample["gold_answer"],
            "gold_supporting_facts": sample["gold_supporting_facts"],
            "null_eligible": null is not None, "null_document_ids": [d["document_id"] for d in null] if null else None,
            "trajectory": {"status": traj["status"], "final_answer": traj["final_answer"], "em": traj["em"], "f1": traj["f1"],
                           "search_calls": T, "token_ids": traj["tokens"], "segments": traj["segments"],
                           "texts": traj["texts"], "turns": traj["turns"], "seed": seed,
                           "reproduced": again["tokens"] == traj["tokens"]},
            "values": values, "f1": f1, "branch_statuses": statuses, "branch_rollouts": len(branches),
            "branch_generated_tokens": sum(b[3]["generated"] for b in branches),
            "seconds": time.monotonic() - started}


# ---------------- factory orchestration (same pattern as work2.pcr) ----------------

def prepare(args):
    protocol = load_protocol(args.protocol)
    if args.output.exists():
        raise ValueError("Plan directory exists; keep frozen selections")
    rows = read_rows(args.sources)
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source IDs")
    chosen = []
    for dataset in ("hotpotqa", "2wiki"):
        pool = sorted((r for r in rows if r["dataset"] == dataset), key=lambda r: rank_key(protocol["seed"], r["sample_id"]))
        if len(pool) < args.per_dataset:
            raise ValueError("Not enough {} questions".format(dataset))
        chosen += pool[:args.per_dataset]
    questions = [{k: r[k] for k in ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts")} for r in chosen]
    n = len(questions)
    for i, q in enumerate(questions):
        q["decoy_sample_id"] = questions[(i + n // 2) % n]["sample_id"]
        q["decoy_question"] = questions[(i + n // 2) % n]["question"]
    smoke = [q["sample_id"] for d in ("hotpotqa", "2wiki") for q in [x for x in questions if x["dataset"] == d][:4]]
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", {
        "version": VERSION, "protocol": protocol, "protocol_sha256": sha(protocol),
        "sources_path": str(args.sources.resolve()), "sources_sha256": file_hash(args.sources),
        "questions": questions, "smoke_question_ids": smoke,
        "selection": "per dataset sha256(seed:sample_id) order, first N; outcome-blind; decoy = question i+n/2",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print("Frozen {} questions (smoke {}).".format(n, len(smoke)))


def frozen(run_dir):
    run = load_json(run_dir / "run_plan.json")
    if file_hash(run["plan_path"]) != run["plan_sha256"]:
        raise ValueError("Plan changed after freezing")
    plan = load_json(run["plan_path"])
    if sha(plan["protocol"]) != plan["protocol_sha256"]:
        raise ValueError("Protocol changed")
    return run, plan


def shard(run, plan, index):
    ids = run["question_ids"][index::len(run["gpus"])]
    by_id = {q["sample_id"]: q for q in plan["questions"]}
    return [by_id[i] for i in ids]


def worker(run_dir, index):
    run, plan = frozen(run_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != run["gpus"][index]:
        raise ValueError("Worker GPU binding mismatch")
    if git_output("rev-parse", "HEAD") != run["repository_commit"]:
        raise ValueError("Code changed after run freezing")
    if file_hash(run["local_file"]) != run["local_file_sha256"]:
        raise ValueError("Local resource configuration changed")
    local, protocol = load_json(run["local_file"]), plan["protocol"]
    questions = shard(run, plan, index)
    folder = run_dir / "workers" / ("worker_" + str(index)); folder.mkdir(parents=True, exist_ok=True)
    out = folder / "records.jsonl"
    done = read_rows(out) if out.exists() else []
    if [r["sample_id"] for r in done] != [q["sample_id"] for q in questions[:len(done)]]:
        raise ValueError("Existing worker output does not match frozen order")
    manifest = {"version": VERSION, "worker_index": index, "gpu": run["gpus"][index], "status": "loading",
                "expected": len(questions), "recorded": len(done), "resumed_from": len(done)}
    write_json(folder / "run_manifest.json", manifest)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from .backends import VLLMGenerator
    from .retrieval import SearchR1Retriever
    generator = VLLMGenerator(local_path(local["agent_checkpoint"]), protocol["agent"], 1, run["gpu_memory_utilization"])
    searcher = LiveSearcher(SearchR1Retriever(local["retrieval_url"], run["request_timeout"]), protocol["retrieval"])
    started = time.monotonic()
    try:
        with out.open("a", encoding="utf-8") as stream:
            for q in questions[len(done):]:
                row = run_question(q, q["decoy_question"], generator, searcher, protocol)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
                manifest.update(status="running", recorded=manifest["recorded"] + 1)
                write_json(folder / "run_manifest.json", manifest)
                print("worker {} {}/{} {} T={} em={} V={} {:.0f}s".format(index, manifest["recorded"], len(questions), q["sample_id"],
                      row["trajectory"]["search_calls"], row["trajectory"]["em"],
                      [round(sum(v) / len(v), 2) for v in row["values"]["V"]], row["seconds"]), flush=True)
        manifest.update(status="completed")
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        manifest["seconds_this_session"] = time.monotonic() - started
        manifest["retrieval_unique_calls_this_session"] = searcher.calls
        manifest["retrieval_drift"] = searcher.drift
        manifest["packages"] = {}
        for name in ("torch", "transformers", "vllm"):
            try: manifest["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: manifest["packages"][name] = None
        write_json(folder / "run_manifest.json", manifest)


def merge(run_dir):
    run, plan = frozen(run_dir)
    rows = []
    for index in range(len(run["gpus"])):
        folder = run_dir / "workers" / ("worker_" + str(index))
        m = load_json(folder / "run_manifest.json")
        got = read_rows(folder / "records.jsonl")
        expected = [q["sample_id"] for q in shard(run, plan, index)]
        if m["status"] != "completed" or [r["sample_id"] for r in got] != expected or m.get("retrieval_drift"):
            raise ValueError("Worker {} incomplete, out of order or drifted".format(index))
        rows += got
    order = {q: i for i, q in enumerate(run["question_ids"])}
    rows.sort(key=lambda r: order[r["sample_id"]])
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    m = load_json(run_dir / "run_manifest.json")
    m.update(status="completed", recorded=len(rows), records_sha256=file_hash(run_dir / "records.jsonl"),
             trajectory_reproduced=sum(r["trajectory"]["reproduced"] for r in rows),
             null_eligible=sum(r["null_eligible"] for r in rows),
             branch_rollouts=sum(r["branch_rollouts"] for r in rows))
    write_json(run_dir / "run_manifest.json", m)
    print("Merged {} questions; {} branch rollouts; trajectory reproduced {}/{}; null eligible {}.".format(
        len(rows), m["branch_rollouts"], m["trajectory_reproduced"], len(rows), m["null_eligible"]))
    return m


def spawn(run_dir, gpus, module, extra, label):
    procs, logs, started, last = [], [], time.monotonic(), 0
    try:
        for index, gpu in enumerate(gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
            log = (run_dir / "logs" / "{}_worker_{}.log".format(label, index)).open("a"); logs.append(log)
            procs.append(subprocess.Popen([sys.executable, "-m", module, "worker", "--run-dir", str(run_dir.resolve()),
                                           "--index", str(index)] + extra, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True))
        while True:
            codes = [p.poll() for p in procs]
            if any(c not in (None, 0) for c in codes):
                raise RuntimeError("A {} worker failed (exit codes {}); see logs/; rerun the same command to resume".format(label, codes))
            if all(c == 0 for c in codes):
                break
            if time.monotonic() - last > 60:
                print("{} {:.0f}s; workers running: {}".format(label, time.monotonic() - started, sum(c is None for c in codes)), flush=True)
                last = time.monotonic()
            time.sleep(1)
    except BaseException:
        for p in procs: stop(p)
        raise
    finally:
        for log in logs: log.close()
    return time.monotonic() - started


def run(args):
    plan = load_json(args.plan_dir / "plan.json")
    protocol = load_protocol(args.protocol)
    if sha(protocol) != plan["protocol_sha256"]:
        raise ValueError("Protocol differs from the frozen plan")
    gpus, local = gpu_list(args.gpus), load_json(args.local)
    if not (local_path(local["agent_checkpoint"]) / "config.json").is_file():
        raise ValueError("Missing local checkpoint config.json")
    from .retrieval import SearchR1Retriever
    probe = SearchR1Retriever(local["retrieval_url"], args.request_timeout).search_batch(["CT1 service check", "second query"], protocol["retrieval"]["candidate_top_k"])
    if len(probe) != 2 or len(probe[0]) < protocol["retrieval"]["visible_top_k"]:
        raise ValueError("Retriever batch interface check failed")
    ids = plan["smoke_question_ids"] if args.smoke else [q["sample_id"] for q in plan["questions"]]
    print("{} questions on GPUs {}; k={} temperature={}; batch retrieval OK.".format(len(ids), gpus, protocol["sampling"]["k"], protocol["sampling"]["temperature"]))
    if args.check_only:
        print("Check passed; no model loaded."); return
    if args.run_dir.exists():
        frozen_run = load_json(args.run_dir / "run_plan.json")
        if frozen_run["question_ids"] != ids or frozen_run["gpus"] != gpus or frozen_run["repository_commit"] != git_output("rev-parse", "HEAD"):
            raise ValueError("Existing run dir differs (questions/GPUs/commit); use a new --run-dir")
        print("Resuming existing run directory.")
    else:
        args.run_dir.mkdir(parents=True); (args.run_dir / "logs").mkdir()
        write_json(args.run_dir / "run_plan.json", {
            "version": VERSION, "plan_path": str((args.plan_dir / "plan.json").resolve()),
            "plan_sha256": file_hash(args.plan_dir / "plan.json"), "question_ids": ids, "smoke": args.smoke, "gpus": gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization, "request_timeout": args.request_timeout,
            "local_file": str(args.local.resolve()), "local_file_sha256": file_hash(args.local),
            "repository_commit": git_output("rev-parse", "HEAD"), "working_tree_dirty": bool(git_output("status", "--porcelain")),
            "assets": {k: asset_identity(local_path(local[k]), k in ("agent_checkpoint", "retriever_model"))
                       for k in ("agent_checkpoint", "retriever_model", "corpus_file", "index_file")},
            "retrieval_url": local["retrieval_url"], "prompt_version": PROMPT_VERSION})
        write_json(args.run_dir / "run_manifest.json", {"version": VERSION, "status": "running", "smoke": args.smoke,
                                                         "planned": len(ids), "research_GO": False, "RL_GO": False})
    seconds = spawn(args.run_dir, gpus, "work2.credit_testbed", [], "mc")
    m = load_json(args.run_dir / "run_manifest.json")
    write_json(args.run_dir / "run_manifest.json", dict(m, mc_wall_seconds=m.get("mc_wall_seconds", 0) + seconds))
    merge(args.run_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--sources", type=Path, required=True)
    q.add_argument("--protocol", type=Path, default=Path("configs/credit_v1.json"))
    q.add_argument("--per-dataset", type=int, default=150); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("run"); q.add_argument("--plan-dir", type=Path, required=True); q.add_argument("--run-dir", type=Path, required=True)
    q.add_argument("--protocol", type=Path, default=Path("configs/credit_v1.json")); q.add_argument("--local", type=Path, default=Path("configs/local.json"))
    q.add_argument("--gpus", default="0,1,2,3"); q.add_argument("--gpu-memory-utilization", type=float, default=.8)
    q.add_argument("--request-timeout", type=float, default=300); q.add_argument("--smoke", action="store_true"); q.add_argument("--check-only", action="store_true")
    q = sub.add_parser("worker"); q.add_argument("--run-dir", type=Path, required=True); q.add_argument("--index", type=int, required=True)
    q = sub.add_parser("merge"); q.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == "prepare": prepare(a)
        elif a.command == "run": run(a)
        elif a.command == "worker": worker(a.run_dir, a.index)
        else: merge(a.run_dir)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

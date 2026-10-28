"""PCR v1 factory CLI: freeze states, run multi-GPU same-prefix arms, merge, audit.

prepare  -> outcome-blind selection of (question, target turn) states from frozen E0 sources
run      -> one TP=1 worker per GPU; every state runs E0/E0 + placebo + content/recovery arms
merge    -> ordered, coverage-checked states.jsonl + manifest (also done by run)
"""
import argparse
import collections
import datetime
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import file_hash, git_output, load_json, local_path, write_json
from .datasets import read_rows
from .placebo import ARM_GROUPS, VERSION, run_state
from .replay import PROMPT_VARIANT_OF, AuditedRetriever
from .run_baseline import asset_identity, validate_design
from .selective import sha


def load_protocol(path):
    p = load_json(path)
    validate_design(p)
    if p.get("protocol") != VERSION or p.get("arm_group") not in ARM_GROUPS:
        raise ValueError("Unsupported PCR protocol or arm group")
    # Greedy is fixed; the search budget is 4 for every agent except formats trained without a turn cap (U1L ReSearch: 8).
    if p["agent"]["temperature"] != 0 or p["agent"]["max_search_turns"] not in (4, 8):
        raise ValueError("PCR v1 requires greedy decoding and a 4- (or, for uncapped formats, 8-) search budget")
    if not p["target_turns"] or any(t not in (1, 2, 3) for t in p["target_turns"]):
        raise ValueError("Target turns must be a subset of 1..3")
    return p


def source_rows(path):
    rows = read_rows(path)
    ids = [r["sample_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Source rollouts empty or duplicated")
    for r in rows:
        if r.get("condition") != "E0_clean" or r.get("prompt_version") not in PROMPT_VARIANT_OF:
            raise ValueError("Sources must be frozen E0_clean rollouts of " + " / ".join(PROMPT_VARIANT_OF))
        if r["status"] == "infrastructure_error" or not r.get("dataset"):
            raise ValueError("Invalid source row: " + r["sample_id"])
    if len({r["prompt_version"] for r in rows}) != 1:
        raise ValueError("Sources mix prompt versions")
    return rows


DATASET_ORDER = ("hotpotqa", "2wiki", "nq", "musique", "bamboogle")


def datasets_of(rows):
    present = {r["dataset"] for r in rows}
    return [d for d in DATASET_ORDER if d in present] + sorted(present - set(DATASET_ORDER))


def rank_key(seed, sample_id):
    return hashlib.sha256("{}:{}".format(seed, sample_id).encode()).hexdigest()


def prepare(args):
    protocol = load_protocol(args.protocol)
    rows = source_rows(args.sources)
    if args.output.exists():
        raise ValueError("Plan directory exists; keep frozen selections")
    questions, unreachable, short = [], collections.Counter(), {}
    datasets = datasets_of(rows)
    for dataset in datasets:
        pool = sorted((r for r in rows if r["dataset"] == dataset), key=lambda r: rank_key(protocol["seed"], r["sample_id"]))
        # Outcome-blind: order by hash only; a question needs at least the first search to fork.
        pool = [r for r in pool if r["prefixes"]]
        limit = args.per_dataset if args.per_dataset > 0 else len(pool)
        if len(pool) < limit:
            if datasets == ["hotpotqa", "2wiki"]:   # original PCR / AC1 behaviour
                raise ValueError("Only {} searchable {} questions, requested {}".format(len(pool), dataset, limit))
            short[dataset] = len(pool)            # small sets (e.g. Bamboogle) are used in full and recorded
        questions.extend(pool[:limit])
    states = []
    for r in questions:
        for turn in protocol["target_turns"]:
            if len(r["prefixes"]) >= turn:
                states.append({"sample_id": r["sample_id"], "dataset": r["dataset"], "target_turn": turn})
            else:
                unreachable[str(turn)] += 1
    per_smoke = 5 if len(datasets) <= 2 else 2
    smoke_ids = [r["sample_id"] for d in datasets for r in [q for q in questions if q["dataset"] == d][:per_smoke]]
    args.output.mkdir(parents=True)
    plan = {"version": VERSION, "protocol": protocol, "protocol_sha256": sha(protocol),
            "sources_path": str(args.sources.resolve()), "sources_sha256": file_hash(args.sources),
            "question_ids": [r["sample_id"] for r in questions], "smoke_question_ids": smoke_ids,
            "states": states, "unreachable_by_turn": dict(unreachable), "datasets": datasets, "short_datasets": short,
            "selection": "per dataset: sha256(seed:sample_id) order among questions with >=1 search; no outcome filtering",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(args.output / "plan.json", plan)
    print("Frozen {} questions / {} states (unreachable by turn: {}).".format(len(questions), len(states), dict(unreachable)))


def gpu_list(text):
    ids = text.split(",")
    if not ids or len(set(ids)) != len(ids) or any(not s.isdigit() for s in ids):
        raise ValueError("Use distinct physical GPU indices, e.g. 0,1,2,3")
    return ids


def frozen_inputs(run_dir):
    run = load_json(run_dir / "run_plan.json")
    plan = load_json(run["plan_path"])
    if file_hash(run["plan_path"]) != run["plan_sha256"] or file_hash(plan["sources_path"]) != plan["sources_sha256"]:
        raise ValueError("Plan or sources changed after freezing")
    if sha(plan["protocol"]) != plan["protocol_sha256"]:
        raise ValueError("Protocol changed")
    return run, plan


def worker_states(run, plan, index):
    ids = run["question_ids"][index::len(run["gpus"])]
    keep = set(ids)
    order = {q: i for i, q in enumerate(ids)}
    states = [s for s in plan["states"] if s["sample_id"] in keep]
    return sorted(states, key=lambda s: (order[s["sample_id"]], s["target_turn"]))


def worker(run_dir, index):
    run, plan = frozen_inputs(run_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != run["gpus"][index]:
        raise ValueError("Worker GPU binding mismatch")
    if git_output("rev-parse", "HEAD") != run["repository_commit"]:
        raise ValueError("Code changed after run freezing")
    local = load_json(run["local_file"])
    if file_hash(run["local_file"]) != run["local_file_sha256"]:
        raise ValueError("Local resource configuration changed")
    protocol = plan["protocol"]
    agent, retrieval, arms = protocol["agent"], protocol["retrieval"], list(ARM_GROUPS[protocol["arm_group"]])
    states = worker_states(run, plan, index)
    sources = {r["sample_id"]: r for r in source_rows(plan["sources_path"]) if r["sample_id"] in {s["sample_id"] for s in states}}
    folder = run_dir / "workers" / ("worker_" + str(index))
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "states.jsonl"
    done = read_rows(out) if out.exists() else []
    if [(r["sample_id"], r["target_turn"]) for r in done] != [(s["sample_id"], s["target_turn"]) for s in states[:len(done)]]:
        raise ValueError("Existing worker output does not match frozen state order; use a new run dir")
    manifest = {"version": VERSION, "worker_index": index, "gpu": run["gpus"][index], "status": "loading",
                "expected_states": len(states), "recorded_states": len(done), "resumed_from": len(done)}
    write_json(folder / "run_manifest.json", manifest)
    from .retrieval import SearchR1Retriever
    from .backends import HFGenerator, VLLMGenerator
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    checkpoint = local_path(local["agent_checkpoint"])
    generator = (VLLMGenerator(checkpoint, agent, 1, run["gpu_memory_utilization"]) if run["backend"] == "vllm"
                 else HFGenerator(checkpoint))
    retriever = AuditedRetriever(SearchR1Retriever(local["retrieval_url"], run["request_timeout"]), retrieval["visible_top_k"])
    for source in sources.values():
        for p in source["prefixes"]:
            retriever.seed(p["search_query"], p["visible_documents"])
    started = time.monotonic()
    try:
        with out.open("a", encoding="utf-8") as stream:
            for state in states[len(done):]:
                row = run_state(sources[state["sample_id"]], state["target_turn"], arms, generator, retriever, agent, retrieval)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
                manifest.update(status="running", recorded_states=manifest["recorded_states"] + 1)
                write_json(folder / "run_manifest.json", manifest)
                print("worker {} {}/{} {} t{} clean={} placebo={}".format(
                    index, manifest["recorded_states"], len(states), state["sample_id"], state["target_turn"],
                    row["arms"]["clean_a"]["em"], [row["arms"][a]["em"] for a in row["arms"] if a.startswith(("perm_", "fmt_"))]), flush=True)
        manifest.update(status="completed", seconds=time.monotonic() - started)
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        manifest["packages"] = {}
        for name in ("torch", "transformers", "vllm"):
            try: manifest["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: manifest["packages"][name] = None
        drift = sum(not c["matched_reference"] for c in retriever.calls)
        with (folder / "retrieval_audit.jsonl").open("a", encoding="utf-8") as stream:
            for call in retriever.calls:
                stream.write(json.dumps(call, ensure_ascii=False) + "\n")
        manifest["retrieval_drift_this_session"] = drift
        write_json(folder / "run_manifest.json", manifest)


def stop(proc):
    try:
        if os.name == "posix": os.killpg(proc.pid, signal.SIGTERM)
        else: proc.terminate()
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix": os.killpg(proc.pid, signal.SIGKILL)
            else: proc.kill()
        except ProcessLookupError: pass


def merge(run_dir):
    run, plan = frozen_inputs(run_dir)
    expected = []
    rows = []
    for index in range(len(run["gpus"])):
        folder = run_dir / "workers" / ("worker_" + str(index))
        m = load_json(folder / "run_manifest.json")
        states = worker_states(run, plan, index)
        got = read_rows(folder / "states.jsonl")
        if m["status"] != "completed" or len(got) != len(states):
            raise ValueError("Worker {} incomplete: {}".format(index, m["status"]))
        for call in read_rows(folder / "retrieval_audit.jsonl"):
            if not call["matched_reference"]:
                raise ValueError("Retrieval drift in worker {}".format(index))
        rows.extend(got)
        expected.extend(states)
    by_key = {(r["sample_id"], r["target_turn"]): r for r in rows}
    if len(by_key) != len(rows) or set(by_key) != {(s["sample_id"], s["target_turn"]) for s in expected}:
        raise ValueError("Merged coverage missing/duplicated")
    order = {q: i for i, q in enumerate(run["question_ids"])}
    ordered = sorted(rows, key=lambda r: (order[r["sample_id"]], r["target_turn"]))
    arms = set(ARM_GROUPS[plan["protocol"]["arm_group"]])
    for r in ordered:
        if not r["controls_exact"] or set(r["arms"]) != arms:
            raise ValueError("State missing arms or control receipt: " + r["sample_id"])
        if r["arms"]["clean_a"]["continuation_sha256"] != r["arms"]["clean_b"]["continuation_sha256"]:
            raise ValueError("E0/E0 mismatch: " + r["sample_id"])
    with (run_dir / "states.jsonl").open("w", encoding="utf-8") as stream:
        for r in ordered:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = dict(load_json(run_dir / "run_manifest.json"), status="completed", recorded_states=len(ordered),
                    states_sha256=file_hash(run_dir / "states.jsonl"), all_controls_exact=True, retrieval_drift=0)
    write_json(run_dir / "run_manifest.json", manifest)
    print("Merged {} states; all E0/E0 controls exact; retrieval drift 0.".format(len(ordered)))
    return manifest


def run(args):
    plan = load_json(args.plan_dir / "plan.json")
    protocol = load_protocol(args.protocol)
    if sha(protocol) != plan["protocol_sha256"] or file_hash(plan["sources_path"]) != plan["sources_sha256"]:
        raise ValueError("Protocol/sources differ from the frozen plan")
    gpus = gpu_list(args.gpus)
    local = load_json(args.local)
    checkpoint = local_path(local["agent_checkpoint"])
    if not (checkpoint / "config.json").is_file():
        raise ValueError("Missing local checkpoint config.json")
    from .retrieval import SearchR1Retriever
    probe = SearchR1Retriever(local["retrieval_url"], args.request_timeout).search("PCR service check", protocol["retrieval"]["candidate_top_k"])
    if len(probe) < protocol["retrieval"]["visible_top_k"]:
        raise ValueError("Insufficient retrieval results")
    ids = plan["smoke_question_ids"] if args.smoke else plan["question_ids"]
    n_states = sum(s["sample_id"] in set(ids) for s in plan["states"])
    print("{} questions / {} states on GPUs {}; arm group {} ({} arms/state).".format(
        len(ids), n_states, gpus, protocol["arm_group"], len(ARM_GROUPS[protocol["arm_group"]])))
    if args.check_only:
        print("Check passed; no model loaded."); return
    resume = args.run_dir.exists()
    if resume and not args.resume:
        raise ValueError("Run directory exists; pass --resume to continue the SAME frozen run, or use a new directory")
    if resume:
        frozen = load_json(args.run_dir / "run_plan.json")
        if frozen["question_ids"] != ids or frozen["gpus"] != gpus or frozen["repository_commit"] != git_output("rev-parse", "HEAD"):
            raise ValueError("Resume requires identical questions, GPU mapping and code commit")
    else:
        args.run_dir.mkdir(parents=True); (args.run_dir / "logs").mkdir()
        assets = {k: asset_identity(local_path(local[k]), k in ("agent_checkpoint", "retriever_model"))
                  for k in ("agent_checkpoint", "retriever_model", "corpus_file", "index_file")}
        write_json(args.run_dir / "run_plan.json", {
            "version": VERSION, "plan_path": str((args.plan_dir / "plan.json").resolve()),
            "plan_sha256": file_hash(args.plan_dir / "plan.json"), "question_ids": ids, "smoke": args.smoke,
            "gpus": gpus, "backend": args.backend, "gpu_memory_utilization": args.gpu_memory_utilization,
            "request_timeout": args.request_timeout, "local_file": str(args.local.resolve()),
            "local_file_sha256": file_hash(args.local), "repository_commit": git_output("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git_output("status", "--porcelain")), "assets": assets,
            "declared_asset_versions": {k: local.get(k) for k in ("checkpoint_version", "corpus_version", "index_version", "actual_index_type")},
            "retrieval_url": local["retrieval_url"]})
        write_json(args.run_dir / "run_manifest.json", {"version": VERSION, "status": "running", "smoke": args.smoke,
                                                         "planned_states": n_states, "research_GO": False, "RL_GO": False})
    started = time.monotonic(); procs, logs = [], []
    try:
        for index, gpu in enumerate(gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
            log = (args.run_dir / "logs" / "worker_{}.log".format(index)).open("a"); logs.append(log)
            procs.append(subprocess.Popen([sys.executable, "-m", "work2.pcr", "worker", "--run-dir", str(args.run_dir.resolve()),
                                           "--index", str(index)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True))
        last = 0
        while True:
            codes = [p.poll() for p in procs]
            if any(c not in (None, 0) for c in codes):
                raise RuntimeError("A PCR worker failed (exit codes {}); see logs/; resume with --resume after fixing".format(codes))
            if all(c == 0 for c in codes): break
            if time.monotonic() - last > 60:
                parts = []
                for index in range(len(gpus)):
                    try:
                        m = load_json(args.run_dir / "workers" / "worker_{}".format(index) / "run_manifest.json")
                        parts.append("GPU{}:{} {}/{}".format(gpus[index], m["status"], m["recorded_states"], m["expected_states"]))
                    except (FileNotFoundError, json.JSONDecodeError):
                        parts.append("GPU{}:starting".format(gpus[index]))
                print("{:.0f}s; {}".format(time.monotonic() - started, "; ".join(parts)), flush=True)
                last = time.monotonic()
            time.sleep(1)
    except BaseException as exc:
        for p in procs: stop(p)
        m = load_json(args.run_dir / "run_manifest.json")
        write_json(args.run_dir / "run_manifest.json", dict(m, status="failed", error=str(exc)))
        raise
    finally:
        for log in logs: log.close()
    m = load_json(args.run_dir / "run_manifest.json")
    write_json(args.run_dir / "run_manifest.json", dict(m, wall_seconds=m.get("wall_seconds", 0) + time.monotonic() - started))
    merge(args.run_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--sources", type=Path, required=True, help="frozen E0_clean rollouts JSONL (e.g. SI1 dev sources.jsonl)")
    q.add_argument("--protocol", type=Path, default=Path("configs/pcr_v1.json"))
    q.add_argument("--per-dataset", type=int, default=300, help="0 = all searchable questions")
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("run")
    q.add_argument("--plan-dir", type=Path, required=True); q.add_argument("--run-dir", type=Path, required=True)
    q.add_argument("--protocol", type=Path, default=Path("configs/pcr_v1.json"))
    q.add_argument("--local", type=Path, default=Path("configs/local.json"))
    q.add_argument("--gpus", default="0,1,2,3"); q.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    q.add_argument("--gpu-memory-utilization", type=float, default=.75); q.add_argument("--request-timeout", type=float, default=120)
    q.add_argument("--smoke", action="store_true"); q.add_argument("--check-only", action="store_true"); q.add_argument("--resume", action="store_true")
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

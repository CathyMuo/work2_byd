"""Four independent TP=1 workers with a global controls-before-treatment barrier."""
import argparse
import collections
import copy
import datetime
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
from .metrics import answer_scores
from .replay import AuditedRetriever
from .retrieval import SearchR1Retriever
from .rollout import run_sample
from .run_baseline import asset_identity
from .run_selective import load_protocol, new_file, validated_cases
from .selective import ARMS, PROMPTS, VERSION, continue_arm, exact, jsonl, load_samples, observable, sha, summarize

DP_VERSION = "SI1_four_worker_global_barrier_v1"
FILES = ("sources.jsonl", "states.jsonl", "controls.jsonl", "cases.jsonl")


def devices(text):
    ids = text.split(",")
    if len(ids) != 4 or len(set(ids)) != 4 or any(not s.isdigit() for s in ids):
        raise ValueError("Exactly four distinct physical GPU indices are required, e.g. 0,1,2,3")
    return ids


def check_gpus(ids):
    available = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
    available = {s.strip() for s in available.splitlines()}
    if not set(ids).issubset(available):
        raise ValueError("The requested four GPUs are not present on this host")


def query_signatures(audits):
    signatures = {}
    for audit in audits:
        if audit["drift"] != 0:
            raise ValueError("Retrieval drift in a worker")
        for call in audit["calls"]:
            if not call["matched_reference"]:
                raise ValueError("A worker observed retrieval drift")
            query, signature = call["query"], call["visible_sha256"]
            if query in signatures and signatures[query] != signature:
                raise ValueError("Cross-worker same-query retrieval drift: " + query)
            signatures[query] = signature
    return signatures


def freeze_plan(args):
    protocol = load_protocol(args.protocol)
    samples, selection = load_samples(args.samples_dir, args.smoke)
    ids = devices(args.gpus)
    if len(samples) < 4:
        raise ValueError("Four workers require at least four frozen questions")
    local = load_json(args.local)
    checkpoint = local_path(local["agent_checkpoint"])
    if not (checkpoint / "config.json").is_file():
        raise ValueError("Missing local checkpoint config.json")
    assets = {k: asset_identity(local_path(local[k]), k in ("agent_checkpoint", "retriever_model"))
              for k in ("agent_checkpoint", "retriever_model", "corpus_file", "index_file")}
    smoke_signatures = {}
    smoke_hash = None
    if not args.smoke:
        if args.smoke_run_dir is None:
            raise ValueError("Development requires --smoke-run-dir")
        _, sm = validated_cases(args.smoke_run_dir, allow_smoke=True)
        if not sm["smoke"] or sm.get("execution") != DP_VERSION:
            raise ValueError("Need a completed FOUR-CARD smoke")
        if sm["protocol_sha256"] != sha(protocol) or sm["selection_manifest_sha256"] != file_hash(args.samples_dir / "selection_manifest.json"):
            raise ValueError("Smoke belongs to different samples/protocol")
        if sm["assets"] != assets or sm["gpu_devices"] != ids or sm["backend"] != args.backend or sm["gpu_memory_utilization"] != args.gpu_memory_utilization or sm["retrieval_url"] != local["retrieval_url"]:
            raise ValueError("Smoke resources/runtime/GPU mapping changed")
        smoke_signatures = query_signatures([load_json(args.smoke_run_dir / "retrieval_audit.json")])
        smoke_hash = file_hash(args.smoke_run_dir / "run_manifest.json")
    plan = {"execution": DP_VERSION, "version": VERSION, "protocol": protocol,
            "protocol_sha256": sha(protocol), "prompts_sha256": sha(PROMPTS),
            "selection_manifest_sha256": file_hash(args.samples_dir / "selection_manifest.json"),
            "sample_file_sha256": file_hash(args.samples_dir / ("smoke.jsonl" if args.smoke else "samples.jsonl")),
            "samples_dir": str(args.samples_dir.resolve()), "local_file": str(args.local.resolve()),
            "local_file_sha256": file_hash(args.local), "run_dir": str(args.run_dir.resolve()),
            "repository_commit": git_output("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git_output("status", "--porcelain")),
            "submodules": git_output("submodule", "status", "--recursive"),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "smoke": args.smoke, "assets": assets, "gpu_devices": ids,
            "backend": args.backend, "tensor_parallel_size": 1,
            "gpu_memory_utilization": args.gpu_memory_utilization, "request_timeout": args.request_timeout,
            "retrieval_url": local["retrieval_url"], "historical_exclusions": selection["exclusions"],
            "declared_asset_versions": {k: local.get(k) for k in ("checkpoint_version", "corpus_version", "index_version", "actual_index_type")},
            "sample_ids": [s["sample_id"] for s in samples],
            "splits": {s["sample_id"]: s["split"] for s in samples},
            "smoke_query_signatures": smoke_signatures, "smoke_run_manifest_sha256": smoke_hash,
            "workers": [{"index": i, "gpu": ids[i],
                         "sample_ids": [s["sample_id"] for s in samples[i::4]],
                         "run_dir": str((args.run_dir / "workers" / ("worker_" + str(i))).resolve())}
                        for i in range(4)]}
    return plan


def worker_context(plan_path, index):
    plan = load_json(plan_path)
    if plan["execution"] != DP_VERSION or len(plan["workers"]) != 4 or not 0 <= index < 4:
        raise ValueError("Invalid four-worker plan")
    worker = plan["workers"][index]
    if worker["index"] != index or os.environ.get("CUDA_VISIBLE_DEVICES") != worker["gpu"]:
        raise ValueError("Worker GPU/index binding mismatch")
    if git_output("rev-parse", "HEAD") != plan["repository_commit"]:
        raise ValueError("Code changed after parallel plan freezing")
    if file_hash(plan["local_file"]) != plan["local_file_sha256"]:
        raise ValueError("Local resource configuration changed")
    samples, _ = load_samples(plan["samples_dir"], plan["smoke"])
    if file_hash(Path(plan["samples_dir"]) / "selection_manifest.json") != plan["selection_manifest_sha256"]:
        raise ValueError("Selection changed after freezing")
    selected = samples[index::4]
    if [s["sample_id"] for s in selected] != worker["sample_ids"]:
        raise ValueError("Shard assignment changed")
    local = load_json(plan["local_file"])
    assets = {k: asset_identity(local_path(local[k]), k in ("agent_checkpoint", "retriever_model"))
              for k in ("agent_checkpoint", "retriever_model", "corpus_file", "index_file")}
    if assets != plan["assets"] or sha(plan["protocol"]) != plan["protocol_sha256"] or sha(PROMPTS) != plan["prompts_sha256"]:
        raise ValueError("Frozen resources/protocol changed")
    return plan, worker, selected, local


def runtime(plan, local):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from .backends import HFGenerator, VLLMGenerator
    model = local_path(local["agent_checkpoint"])
    generator = (VLLMGenerator(model, plan["protocol"]["agent"], 1, plan["gpu_memory_utilization"])
                 if plan["backend"] == "vllm" else HFGenerator(model))
    retriever = AuditedRetriever(SearchR1Retriever(local["retrieval_url"], plan["request_timeout"]),
                                plan["protocol"]["retrieval"]["visible_top_k"])
    retriever.expected.update(plan["smoke_query_signatures"])
    return generator, retriever


def control_data(folder, expected_ids=None):
    """Audit immutable controls independently, before any action files exist."""
    folder = Path(folder)
    m = load_json(folder / "control_manifest.json")
    if m["status"] != "controls_ready" or not m["all_original_controls_exact"]:
        raise ValueError("Worker did not finish all controls")
    if sha(m["protocol"]) != m["protocol_sha256"] or sha(PROMPTS) != m["prompts_sha256"]:
        raise ValueError("Control protocol/prompts mismatch")
    for name, h in m["artifact_sha256"].items():
        if file_hash(folder / name) != h:
            raise ValueError("Immutable control artifact changed: " + name)
    groups = [read_rows(folder / name) for name in FILES[:3]]
    ids = m["sample_ids"]
    if expected_ids is not None and ids != expected_ids:
        raise ValueError("Control shard differs from plan")
    if not ids or len(set(ids)) != len(ids) or any([r["sample_id"] for r in rows] != ids for rows in groups):
        raise ValueError("Incomplete/duplicate controls")
    sources, states, controls = groups
    agent, retrieval = m["protocol"]["agent"], m["protocol"]["retrieval"]
    for source, state, control in zip(*groups):
        if state != dict(observable(source, agent, retrieval), split=source["split"]):
            raise ValueError("Control observable state differs from source")
        if not control["exact"] or not all(exact(source, control[k]) for k in ("keep_a", "keep_b")):
            raise ValueError("Control/history mismatch")
        for row in (source, control["keep_a"], control["keep_b"]):
            if row["status"] == "infrastructure_error":
                raise ValueError("Invalid infrastructure control")
            scores = answer_scores(row["final_answer"], source["gold_answer"]) if row["status"] == "answered" else {"em": 0, "f1": 0.0}
            if row["em"] != scores["em"] or row["f1"] != scores["f1"]:
                raise ValueError("Control raw score mismatch")
            if row["generated_tokens"] != sum(len(s["token_ids"]) for s in row["steps"]) or row["model_input_tokens"] != sum(s["input_tokens"] for s in row["steps"]):
                raise ValueError("Control raw costs mismatch")
            if row["search_calls"] != len(row["prefixes"]) or row["search_calls"] > agent["max_search_turns"]:
                raise ValueError("Control search count/cap mismatch")
    audit = load_json(folder / "control_retrieval_audit.json")
    query_signatures([audit])
    expected = sum(s["search_calls"] for s in sources)
    expected += sum(c[k]["search_calls"] - (1 if s["reachable"] else 0)
                    for c, s in zip(controls, states) for k in ("keep_a", "keep_b"))
    if len(audit["calls"]) != expected:
        raise ValueError("Control retrieval call count mismatch")
    return sources, states, controls, m


def worker_controls(plan_path, index):
    plan, worker, samples, local = worker_context(plan_path, index)
    folder = Path(worker["run_dir"])
    if folder.exists():
        raise ValueError("Refuse overwriting worker control directory")
    folder.mkdir(parents=True)
    m = {k: copy.deepcopy(plan[k]) for k in ("version", "protocol", "protocol_sha256", "prompts_sha256",
          "selection_manifest_sha256", "sample_file_sha256", "smoke", "repository_commit", "working_tree_dirty",
          "submodules", "timestamp_utc", "assets", "backend", "tensor_parallel_size", "gpu_memory_utilization",
          "retrieval_url", "declared_asset_versions", "historical_exclusions")}
    m.update(status="initializing", sample_ids=worker["sample_ids"], expected_questions=len(samples), recorded_questions=0,
             parallel_binding={"plan_sha256": file_hash(plan_path), "worker_index": index, "gpu": worker["gpu"]},
             cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], research_GO=False, RL_GO=False)
    generator = retriever = None
    start = time.monotonic()
    write_json(folder / "run_manifest.json", m)
    try:
        generator, retriever = runtime(plan, local)
        m["packages"] = {}
        for name in ("torch", "transformers", "vllm"):
            try: m["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: m["packages"][name] = None
        agent, retrieval = plan["protocol"]["agent"], plan["protocol"]["retrieval"]
        sources = []
        m["status"] = "baseline"
        with (folder / "sources.jsonl").open("x", encoding="utf-8") as output:
            for sample in samples:
                row = run_sample(sample, generator, retriever, agent, retrieval)
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); output.flush()
                if row["status"] == "infrastructure_error": raise RuntimeError(str(row.get("error")))
                sources.append(row)
                m["baseline_recorded"] = len(sources); write_json(folder / "run_manifest.json", m)
                print("worker {} baseline {}/{}".format(index, len(sources), len(samples)), flush=True)
        states = [dict(observable(s, agent, retrieval, generator), split=s["split"]) for s in sources]
        jsonl(folder / "states.jsonl", states)
        m["status"] = "controls"
        with (folder / "controls.jsonl").open("x", encoding="utf-8") as output:
            for i, source in enumerate(sources):
                if states[i]["reachable"]:
                    a = continue_arm(source, "keep", generator, retriever, agent, retrieval)
                    b = continue_arm(source, "keep", generator, retriever, agent, retrieval)
                else:
                    a = run_sample(samples[i], generator, retriever, agent, retrieval)
                    b = run_sample(samples[i], generator, retriever, agent, retrieval)
                ok = exact(source, a) and exact(source, b) and exact(a, b)
                output.write(json.dumps({"sample_id": source["sample_id"], "exact": ok, "keep_a": a, "keep_b": b}, ensure_ascii=False, allow_nan=False) + "\n"); output.flush()
                if not ok: raise ValueError("Worker original control/history mismatch: " + source["sample_id"])
                m["controls_recorded"] = i + 1; write_json(folder / "run_manifest.json", m)
                print("worker {} controls {}/{} exact".format(index, i + 1, len(samples)), flush=True)
        m.update(status="controls_ready", all_original_controls_exact=True,
                 control_phase_seconds=time.monotonic() - start)
        write_json(folder / "control_retrieval_audit.json", {"calls": retriever.calls, "drift": 0})
        m["artifact_sha256"] = {n: file_hash(folder / n) for n in FILES[:3] + ("control_retrieval_audit.json",)}
        new_file(folder / "control_manifest.json", m)
        control_data(folder, worker["sample_ids"])
    except BaseException as exc:
        m.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(folder / "run_manifest.json", m)


def publish_barrier(plan_path):
    plan = load_json(plan_path)
    receipts, audits = [], []
    all_ids = []
    for worker in plan["workers"]:
        folder = Path(worker["run_dir"])
        _, states, _, m = control_data(folder, worker["sample_ids"])
        binding = {"plan_sha256": file_hash(plan_path), "worker_index": worker["index"], "gpu": worker["gpu"]}
        if m["parallel_binding"] != binding or m["assets"] != plan["assets"] or m["protocol_sha256"] != plan["protocol_sha256"] or m["repository_commit"] != plan["repository_commit"]:
            raise ValueError("Control worker provenance differs from frozen plan")
        if any(s["split"] != plan["splits"][s["sample_id"]] for s in states):
            raise ValueError("Shard split differs from frozen sample manifest")
        all_ids.extend(m["sample_ids"])
        receipts.append({"worker_index": worker["index"], "run_dir": worker["run_dir"],
                         "control_manifest_sha256": file_hash(folder / "control_manifest.json")})
        audits.append(load_json(folder / "control_retrieval_audit.json"))
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(plan["sample_ids"]):
        raise ValueError("Global controls have missing/duplicate questions")
    barrier = {"execution": DP_VERSION, "plan_sha256": file_hash(plan_path), "all_workers_controls_exact": True,
               "sample_ids": plan["sample_ids"], "workers": receipts,
               "query_signatures": query_signatures(audits)}
    new_file(Path(plan["run_dir"]) / "global_control_barrier.json", barrier)
    return barrier


def verify_barrier(plan_path):
    plan = load_json(plan_path)
    path = Path(plan["run_dir"]) / "global_control_barrier.json"
    barrier = load_json(path)
    if barrier["execution"] != DP_VERSION or barrier["plan_sha256"] != file_hash(plan_path) or not barrier["all_workers_controls_exact"]:
        raise ValueError("No valid global control barrier")
    if barrier["sample_ids"] != plan["sample_ids"] or [r["worker_index"] for r in barrier["workers"]] != list(range(4)):
        raise ValueError("Barrier does not cover exactly four workers/all questions")
    audits = []
    for receipt, worker in zip(barrier["workers"], plan["workers"]):
        folder = Path(worker["run_dir"])
        if receipt["run_dir"] != worker["run_dir"] or file_hash(folder / "control_manifest.json") != receipt["control_manifest_sha256"]:
            raise ValueError("Barrier control receipt changed")
        m = load_json(folder / "control_manifest.json")
        if m["sample_ids"] != worker["sample_ids"] or m["status"] != "controls_ready" or not m["all_original_controls_exact"]:
            raise ValueError("Invalid immutable control coverage")
        for name, h in m["artifact_sha256"].items():
            if file_hash(folder / name) != h: raise ValueError("Control artifact changed after global barrier")
        audits.append(load_json(folder / "control_retrieval_audit.json"))
    if barrier["query_signatures"] != query_signatures(audits):
        raise ValueError("Global retrieval signatures changed")
    return barrier


def worker_actions(plan_path, index):
    plan, worker, _, local = worker_context(plan_path, index)
    barrier = verify_barrier(plan_path)  # Checked BEFORE any model load or treatment.
    folder = Path(worker["run_dir"])
    sources, states, controls, m = control_data(folder, worker["sample_ids"])
    if load_json(folder / "run_manifest.json")["status"] != "controls_ready" or (folder / "cases.jsonl").exists():
        raise ValueError("Worker actions already started; refuse overwriting")
    start = time.monotonic(); completed = []
    generator = retriever = None
    try:
        generator, retriever = runtime(plan, local)
        retriever.expected.update(barrier["query_signatures"])
        retriever.calls = load_json(folder / "control_retrieval_audit.json")["calls"]
        m.update(status="actions", global_barrier_sha256=file_hash(Path(plan["run_dir"]) / "global_control_barrier.json"))
        write_json(folder / "run_manifest.json", m)
        agent, retrieval = plan["protocol"]["agent"], plan["protocol"]["retrieval"]
        with (folder / "cases.jsonl").open("x", encoding="utf-8") as output:
            for i, (source, state) in enumerate(zip(sources, states)):
                arms = {"keep": controls[i]["keep_a"]} if state["reachable"] else {}
                for arm in (ARMS[1:] if state["reachable"] else ARMS):
                    arms[arm] = continue_arm(source, arm, generator, retriever, agent, retrieval)
                case = {"version": VERSION, "sample_id": source["sample_id"], "dataset": source["dataset"],
                        "split": source["split"], "state": state, "state_sha256": sha(state), "arms": arms}
                output.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n"); output.flush()
                completed.append(case); m["recorded_questions"] = len(completed)
                write_json(folder / "run_manifest.json", m)
                print("worker {} actions {}/{}".format(index, i + 1, len(sources)), flush=True)
        m.update(status="completed", all_planned_questions_completed=True,
                 action_phase_seconds=time.monotonic() - start)
        write_json(folder / "metrics.json", summarize(completed))
    except BaseException as exc:
        m.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if retriever is not None:
            write_json(folder / "retrieval_audit.json", {"calls": retriever.calls,
                      "drift": sum(not r["matched_reference"] for r in retriever.calls)})
        if m["status"] == "completed":
            m["artifact_sha256"] = {n: file_hash(folder / n) for n in FILES + ("retrieval_audit.json",)}
        write_json(folder / "run_manifest.json", m)
    validated_cases(folder, allow_smoke=True)


def merge_ordered(plan, name):
    streams = [(Path(w["run_dir"]) / name).open(encoding="utf-8") for w in plan["workers"]]
    try:
        with (Path(plan["run_dir"]) / name).open("x", encoding="utf-8") as out:
            for i, sample_id in enumerate(plan["sample_ids"]):
                line = streams[i % 4].readline()
                if not line or json.loads(line)["sample_id"] != sample_id:
                    raise ValueError("Shard actual coverage/order differs from frozen plan: " + name)
                out.write(line.rstrip("\r\n") + "\n")
            if any(stream.read().strip() for stream in streams):
                raise ValueError("Shard contains extra rows: " + name)
    finally:
        for stream in streams: stream.close()


def verify_merged_provenance(folder, manifest):
    folder = Path(folder)
    if file_hash(folder / "parallel_plan.json") != manifest["parallel_plan_sha256"]:
        raise ValueError("Parallel plan changed after merge")
    plan = load_json(folder / "parallel_plan.json")
    verify_barrier(folder / "parallel_plan.json")
    if manifest["sample_ids"] != plan["sample_ids"] or manifest["gpu_devices"] != plan["gpu_devices"]:
        raise ValueError("Merged coverage/GPU mapping mismatch")
    if len(manifest["worker_manifest_sha256"]) != 4:
        raise ValueError("Merged receipt is missing workers")
    for worker, h in zip(plan["workers"], manifest["worker_manifest_sha256"]):
        p = Path(worker["run_dir"]) / "run_manifest.json"
        if file_hash(p) != h or load_json(p)["status"] != "completed":
            raise ValueError("Merged worker receipt changed or incomplete")


def merge(plan_path):
    plan = load_json(plan_path); folder = Path(plan["run_dir"])
    verify_barrier(plan_path)
    manifests, audits = [], []
    for worker in plan["workers"]:
        _, m = validated_cases(worker["run_dir"], allow_smoke=True)
        if m["sample_ids"] != worker["sample_ids"] or m["protocol_sha256"] != plan["protocol_sha256"] or m["assets"] != plan["assets"]:
            raise ValueError("Worker differs from merge plan")
        binding = {"plan_sha256": file_hash(plan_path), "worker_index": worker["index"], "gpu": worker["gpu"]}
        if m["parallel_binding"] != binding or m["global_barrier_sha256"] != file_hash(folder / "global_control_barrier.json"):
            raise ValueError("Worker GPU/global barrier provenance mismatch")
        for key in ("backend", "tensor_parallel_size", "gpu_memory_utilization", "retrieval_url", "repository_commit", "submodules"):
            if m[key] != plan[key]: raise ValueError("Worker execution configuration differs from plan: " + key)
        if manifests and m["packages"] != manifests[0]["packages"]:
            raise ValueError("Worker package versions differ")
        manifests.append(m); audits.append(load_json(Path(worker["run_dir"]) / "retrieval_audit.json"))
    query_signatures(audits)  # Includes NEW queries shared across treatment workers.
    for name in FILES: merge_ordered(plan, name)
    write_json(folder / "retrieval_audit.json", {"calls": [c for a in audits for c in a["calls"]], "drift": 0})
    m = {k: copy.deepcopy(manifests[0][k]) for k in ("version", "protocol", "protocol_sha256", "prompts_sha256",
         "selection_manifest_sha256", "sample_file_sha256", "smoke", "repository_commit", "working_tree_dirty",
         "submodules", "timestamp_utc", "assets", "backend", "tensor_parallel_size", "gpu_memory_utilization",
         "retrieval_url", "declared_asset_versions", "historical_exclusions", "packages")}
    m.update(execution=DP_VERSION, gpu_devices=plan["gpu_devices"], sample_ids=plan["sample_ids"],
             expected_questions=len(plan["sample_ids"]), recorded_questions=len(plan["sample_ids"]),
             status="completed", all_original_controls_exact=True, all_planned_questions_completed=True,
             parallel_plan_sha256=file_hash(plan_path),
             global_barrier_sha256=file_hash(folder / "global_control_barrier.json"),
             worker_manifest_sha256=[file_hash(Path(w["run_dir"]) / "run_manifest.json") for w in plan["workers"]],
             execution_cost={"worker_phase_seconds": [s["control_phase_seconds"] + s["action_phase_seconds"] for s in manifests],
                             "summed_model_phase_hours": sum(s["control_phase_seconds"] + s["action_phase_seconds"] for s in manifests) / 3600},
             artifact_sha256={n: file_hash(folder / n) for n in FILES + ("retrieval_audit.json",)},
             research_GO=False, RL_GO=False)
    write_json(folder / "run_manifest.json", m)
    cases, _ = validated_cases(folder, allow_smoke=True)
    write_json(folder / "metrics.json", summarize(cases))
    new_file(folder / "audit_receipt.json", {"version": VERSION, "execution": DP_VERSION,
             "mechanical_checks_passed": True, "smoke": plan["smoke"], "questions": len(cases),
             "worker_count": 4, "global_original_controls_exact": True,
             "parallel_plan_sha256": file_hash(plan_path), "metrics": summarize(cases)})
    return m


def stop_worker(proc):
    # Terminate only the process group created by this coordinator, never other GPU jobs/services.
    try:
        if os.name == "posix": os.killpg(proc.pid, signal.SIGTERM)
        elif proc.poll() is None: proc.terminate()
    except ProcessLookupError: pass
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix": os.killpg(proc.pid, signal.SIGKILL)
            else: proc.kill()
        except ProcessLookupError: pass
        proc.wait()


def run_phase(plan_path, phase):
    plan = load_json(plan_path); procs, logs = [], []
    start = time.monotonic(); last = start
    try:
        for worker in plan["workers"]:
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = worker["gpu"]
            env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
            log = (Path(plan["run_dir"]) / "logs" / (phase + "_worker_" + str(worker["index"]) + ".log")).open("x")
            logs.append(log)
            procs.append(subprocess.Popen([sys.executable, "-m", "work2.parallel_selective", "--worker",
                          str(worker["index"]), "--phase", phase, "--plan", str(Path(plan_path).resolve())],
                         env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        # ALL four processes are dispatched before waiting for any one worker.
        while True:
            codes = [p.poll() for p in procs]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError("A {} worker failed (exit codes {}); inspect logs; no partial merge".format(phase, codes))
            if all(c is not None for c in codes): break
            if time.monotonic() - last >= 30:
                progress = []
                for worker in plan["workers"]:
                    try:
                        m = load_json(Path(worker["run_dir"]) / "run_manifest.json")
                        count = m.get("recorded_questions", 0) if phase == "actions" else m.get("controls_recorded", 0)
                        progress.append("GPU{}:{} {}/{}".format(worker["gpu"], m["status"], count, len(worker["sample_ids"])))
                    except (FileNotFoundError, json.JSONDecodeError):
                        progress.append("GPU{}:starting".format(worker["gpu"]))
                print("{} finished {}/4, elapsed {:.1f}s; {}".format(phase, sum(c is not None for c in codes), time.monotonic() - start, "; ".join(progress)), flush=True)
                last = time.monotonic()
            time.sleep(.5)
    except BaseException:
        for proc in procs: stop_worker(proc)
        raise
    finally:
        for log in logs: log.close()
    return time.monotonic() - start


def coordinate(args):
    plan = freeze_plan(args)
    check_gpus(plan["gpu_devices"])
    probe = SearchR1Retriever(plan["retrieval_url"], args.request_timeout).search("SI1 service check", plan["protocol"]["retrieval"]["candidate_top_k"])
    if len(probe) < plan["protocol"]["retrieval"]["visible_top_k"]: raise ValueError("Insufficient retrieval results")
    if args.check_only:
        print("Four GPU IDs, samples, protocol/assets and service checked; no model loaded."); return
    if args.run_dir.exists(): raise ValueError("Run directory exists; keep historical artifacts")
    args.run_dir.mkdir(parents=True); (args.run_dir / "logs").mkdir()
    plan_path = args.run_dir / "parallel_plan.json"; new_file(plan_path, plan)
    status = {"execution": DP_VERSION, "status": "controls", "planned_questions": len(plan["sample_ids"]),
              "gpu_devices": plan["gpu_devices"], "worker_count": 4}
    write_json(args.run_dir / "coordinator_status.json", status)
    started = time.monotonic()
    try:
        status["controls_wall_seconds"] = run_phase(plan_path, "controls")
        publish_barrier(plan_path)
        status["status"] = "actions"; write_json(args.run_dir / "coordinator_status.json", status)
        print("ALL four workers' original controls passed; starting treatments.", flush=True)
        status["actions_wall_seconds"] = run_phase(plan_path, "actions")
        status["status"] = "merging"; write_json(args.run_dir / "coordinator_status.json", status)
        merge(plan_path)
        status.update(status="completed", wall_seconds=time.monotonic() - started)
        print("Four-card SI1 completed, {} questions; CPU aggregate audit passed.".format(len(plan["sample_ids"])), flush=True)
    except BaseException as exc:
        status.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        manifest_path = args.run_dir / "run_manifest.json"
        if manifest_path.exists():
            m = load_json(manifest_path)
            m.update(status="failed", error=status["error"])
            write_json(manifest_path, m)
        raise
    finally: write_json(args.run_dir / "coordinator_status.json", status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=int); p.add_argument("--phase", choices=("controls", "actions")); p.add_argument("--plan", type=Path)
    p.add_argument("--local", type=Path, default=Path("configs/local.json"))
    p.add_argument("--protocol", type=Path, default=Path("configs/selective_v1.json"))
    p.add_argument("--samples-dir", type=Path); p.add_argument("--run-dir", type=Path)
    p.add_argument("--smoke-run-dir", type=Path); p.add_argument("--smoke", action="store_true")
    p.add_argument("--gpus", default="0,1,2,3"); p.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    p.add_argument("--gpu-memory-utilization", type=float, default=.75); p.add_argument("--request-timeout", type=float, default=120)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    try:
        if args.worker is not None:
            if args.plan is None or args.phase is None: raise ValueError("Worker needs plan and phase")
            (worker_controls if args.phase == "controls" else worker_actions)(args.plan, args.worker)
        else:
            if args.samples_dir is None or args.run_dir is None: raise ValueError("Coordinator needs samples-dir and run-dir")
            if not 0 < args.gpu_memory_utilization < 1 or args.request_timeout <= 0: raise ValueError("Invalid runtime settings")
            coordinate(args)
    except (ValueError, KeyError, FileNotFoundError) as exc: p.error(str(exc))
    return 0


if __name__ == "__main__": raise SystemExit(main())

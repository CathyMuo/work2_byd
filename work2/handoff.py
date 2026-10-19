"""HR1 handoff replay: where does a search agent's RL gain come from?

Policies I (Qwen2.5-7B-Instruct with the Search-R1 template) and R (Search-R1 7B
PPO). For each question and source policy P, one scored trajectory tau_P is
sampled; from token-identical prefixes s_t (after P's t-th observation) a
continuer Q in {I, R} samples k continuations under three arms:
  real     the prefix as is
  placebo  every observation 1..t replaced by the frozen null observation
  evonly   P's <think> text dropped; only <search>query</search> + observations kept
Three phases keep ONE model per GPU process:
  r1  load R: sample tau_R, continue R on tau_R           (R>R)
  i2  load I: sample tau_I, continue I on tau_I and tau_R (I>I, R>I)
  r3  load R: continue R on tau_I                         (I>R)
"""
import argparse
import datetime
import importlib.metadata
import json
import os
import time
from pathlib import Path

from .audit_confirm import DEFAULT_EXCLUDE, select_fresh, write_selection
from .common import file_hash, git_output, load_json, local_path, write_json
from .credit_testbed import (LiveSearcher, batched_continue, new_rollout, null_documents, obs_end, query_end,
                             seed_for, spawn)
from .datasets import read_rows
from .pcr import gpu_list
from .replay import observation
from .rollout import PROMPT_VERSION, question_prompt
from .run_baseline import asset_identity, validate_design
from .selective import sha

VERSION = "HR1_handoff_replay_pilot_v1"
PHASES = ("r1", "i2", "r3")
MODEL_OF = {"r1": "R", "i2": "I", "r3": "R"}
CHECKPOINT_KEY = {"R": "agent_checkpoint", "I": "instruct_checkpoint"}
ARMS = ("real", "placebo", "evonly")


def load_protocol(path):
    p = load_json(path)
    if p.get("protocol") != VERSION:
        raise ValueError("Unsupported HR1 protocol")
    validate_design(dict(p, agent=dict(p["agent"], temperature=0)))
    s = p["sampling"]
    if not 0 < s["temperature"] <= 2 or s["k"] < 2:
        raise ValueError("Invalid sampling settings")
    return p


def points(trajectory, handoff_points):
    T = trajectory["search_calls"]
    return sorted({t for t in handoff_points if t <= T} | {T})


def state_tokens(traj, t, arm, null_obs, encode):
    """Token prefix for handing tau over after observation t under an arm."""
    if arm == "real" or t == 0:
        return traj["token_ids"][:obs_end(traj, t)]
    end = obs_end(traj, t)
    segs = [s for s in traj["segments"] if s["end"] <= end]
    if arm == "placebo":
        out = []
        for s in segs:
            out += list(null_obs) if s["kind"] == "obs" else traj["token_ids"][s["start"]:s["end"]]
        return out
    if arm == "evonly":
        out = list(traj["token_ids"][segs[0]["start"]:segs[0]["end"]])  # initial prompt
        for turn in range(1, t + 1):
            obs = next(s for s in segs if s["kind"] == "obs" and s["turn"] == turn)
            out += encode("<search> {} </search>".format(traj["turns"][turn - 1]["query"]))
            out += traj["token_ids"][obs["start"]:obs["end"]]
        return out
    raise ValueError("Unknown arm " + arm)


def sample_trajectory(sample, generator, searcher, protocol, policy, prompt=None):
    """prompt: shared initial tokens (HR1 renders it once with R's template for BOTH policies)."""
    agent, retrieval, s = protocol["agent"], protocol["retrieval"], protocol["sampling"]
    if prompt is None:
        prompt = generator.encode(generator.initial_prompt(question_prompt(sample["question"], agent["max_search_turns"])))
    traj = new_rollout(sample, prompt, 0, seed_for(protocol["seed"], sample["sample_id"], "trajectory", policy), track=True,
                       segments=[{"kind": "prompt", "start": 0, "end": len(prompt)}])
    batched_continue([traj], generator, searcher, agent, retrieval, s["temperature"], s["top_p"])
    return {"policy": policy, "status": traj["status"], "final_answer": traj["final_answer"], "em": traj["em"], "f1": traj["f1"],
            "search_calls": traj["search_calls"], "token_ids": traj["tokens"], "segments": traj["segments"],
            "texts": traj["texts"], "turns": traj["turns"], "prompt_tokens": prompt}


def continue_on(sample, source, continuer, null_obs, generator, searcher, protocol):
    """All (arm, t) branches of continuer Q on source trajectory tau_P; returns values and status counts."""
    agent, retrieval, s = protocol["agent"], protocol["retrieval"], protocol["sampling"]
    combo = source["policy"] + ">" + continuer
    branches = []
    for t in points(source, protocol["handoff_points"]):
        for arm in (ARMS if t >= 1 else ("real",)):
            tokens = state_tokens(source, t, arm, null_obs, generator.encode)
            for i in range(s["k"]):
                branches.append((arm, t, new_rollout(sample, tokens, t, seed_for(protocol["seed"], sample["sample_id"], combo, arm, t, i))))
    batched_continue([b[2] for b in branches], generator, searcher, agent, retrieval, s["temperature"], s["top_p"])
    values, statuses = {}, {}
    for arm, t, r in branches:
        values.setdefault(arm, {}).setdefault(str(t), []).append(r["em"])
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return combo, {"values": values, "statuses": statuses, "rollouts": len(branches)}


def run_phase_question(phase, sample, previous, generator, searcher, protocol):
    started = time.monotonic()
    if phase == "r1":
        null = null_documents(searcher, sample["decoy_question"], sample, protocol["retrieval"])
        if null is None:
            raise ValueError("No eligible null documents for " + sample["sample_id"])
        traj = sample_trajectory(sample, generator, searcher, protocol, "R")
        null_obs = generator.encode(observation(null))
        combo, block = continue_on(sample, traj, "R", null_obs, generator, searcher, protocol)
        row = {"sample_id": sample["sample_id"], "null_documents": [{"document_id": d["document_id"], "contents": d["contents"]} for d in null],
               "null_observation_token_ids": null_obs, "trajectories": {"R": traj}, "combos": {combo: block}}
    elif phase == "i2":
        traj_r = previous["trajectories"]["R"]
        native = generator.encode(generator.initial_prompt(question_prompt(sample["question"], protocol["agent"]["max_search_turns"])))
        traj = sample_trajectory(sample, generator, searcher, protocol, "I", prompt=list(traj_r["prompt_tokens"]))
        null_obs = previous["null_observation_token_ids"]
        combos = dict([continue_on(sample, traj, "I", null_obs, generator, searcher, protocol),
                       continue_on(sample, traj_r, "I", null_obs, generator, searcher, protocol)])
        row = {"sample_id": sample["sample_id"], "trajectories": {"I": traj}, "combos": combos,
               "template_identical": traj["prompt_tokens"] == traj_r["prompt_tokens"],
               "instruct_native_template_identical": native == traj_r["prompt_tokens"]}
    else:
        traj_i = previous["trajectories"]["I"]
        combo, block = continue_on(sample, traj_i, "R", previous["null_observation_token_ids"], generator, searcher, protocol)
        row = {"sample_id": sample["sample_id"], "combos": {combo: block}}
    row["seconds"] = time.monotonic() - started
    return row


# ---------------- orchestration ----------------

def freeze(args):
    protocol, local = load_protocol(args.protocol), load_json(args.local)
    if args.output.exists():
        raise ValueError("Output exists; keep frozen selections")
    defaults = list(DEFAULT_EXCLUDE) + ["outputs/ac1_fresh/samples.jsonl"]
    paths = list(args.exclude) + [Path(p) for p in defaults if Path(p).is_file() and Path(p) not in args.exclude]
    chosen, receipts, ids, questions = select_fresh(local, paths, args.per_dataset, protocol["seed"])
    n = len(chosen)
    for i, q in enumerate(chosen):
        q["decoy_sample_id"] = chosen[(i + n // 2) % n]["sample_id"]
        q["decoy_question"] = chosen[(i + n // 2) % n]["question"]
    write_selection(args.output, chosen, {"version": VERSION, "protocol_sha256": sha(protocol), "seed": protocol["seed"],
                                          "per_dataset": args.per_dataset, "exclusions": receipts,
                                          "excluded_ids": len(ids), "excluded_questions": len(questions),
                                          "selection": "sha256(seed:sample_id) per dataset; excluded by id and question incl. AC1 fresh; decoy = i+n/2"})
    print("Frozen {} HR1 questions; excluded {} historical ids from {} files.".format(n, len(ids), len(receipts)))


def frozen(run_dir):
    run = load_json(run_dir / "run_plan.json")
    if file_hash(run["samples_path"]) != run["samples_sha256"] or sha(run["protocol"]) != run["protocol_sha256"]:
        raise ValueError("Samples or protocol changed after freezing")
    return run


def phase_file(run_dir, phase, index):
    return run_dir / "workers" / phase / "worker_{}.jsonl".format(index)


def worker(run_dir, index, phase):
    run = frozen(run_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != run["gpus"][index]:
        raise ValueError("Worker GPU binding mismatch")
    if git_output("rev-parse", "HEAD") != run["repository_commit"]:
        raise ValueError("Code changed after run freezing")
    if file_hash(run["local_file"]) != run["local_file_sha256"]:
        raise ValueError("Local resource configuration changed")
    local, protocol = load_json(run["local_file"]), run["protocol"]
    samples = read_rows(run["samples_path"])[index::len(run["gpus"])]
    previous = {}
    if phase != "r1":
        prior = "r1" if phase == "i2" else "i2"
        rows = read_rows(phase_file(run_dir, prior, index))
        if [r["sample_id"] for r in rows] != [s["sample_id"] for s in samples]:
            raise ValueError("Previous phase {} incomplete for worker {}".format(prior, index))
        previous = {r["sample_id"]: r for r in rows}
        if phase == "r3":
            r1 = {r["sample_id"]: r for r in read_rows(phase_file(run_dir, "r1", index))}
            for sid, r in previous.items():
                r["null_observation_token_ids"] = r1[sid]["null_observation_token_ids"]
    out = phase_file(run_dir, phase, index); out.parent.mkdir(parents=True, exist_ok=True)
    done = read_rows(out) if out.exists() else []
    if [r["sample_id"] for r in done] != [s["sample_id"] for s in samples[:len(done)]]:
        raise ValueError("Existing phase output does not match frozen order")
    mpath = out.with_suffix(".manifest.json")
    manifest = {"version": VERSION, "phase": phase, "worker_index": index, "gpu": run["gpus"][index], "status": "loading",
                "expected": len(samples), "recorded": len(done)}
    write_json(mpath, manifest)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from .backends import VLLMGenerator
    from .retrieval import SearchR1Retriever
    model = MODEL_OF[phase]
    generator = VLLMGenerator(local_path(local[CHECKPOINT_KEY[model]]), protocol["agent"], 1, run["gpu_memory_utilization"])
    searcher = LiveSearcher(SearchR1Retriever(local["retrieval_url"], run["request_timeout"]), protocol["retrieval"])
    started = time.monotonic()
    try:
        with out.open("a", encoding="utf-8") as stream:
            for sample in samples[len(done):]:
                row = run_phase_question(phase, sample, previous.get(sample["sample_id"]), generator, searcher, protocol)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
                manifest.update(status="running", recorded=manifest["recorded"] + 1)
                write_json(mpath, manifest)
                print("{} worker {} {}/{} {} {:.0f}s".format(phase, index, manifest["recorded"], len(samples), sample["sample_id"], row["seconds"]), flush=True)
        manifest.update(status="completed")
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        manifest["seconds_this_session"] = time.monotonic() - started
        manifest["retrieval"] = searcher.stats()
        manifest["packages"] = {}
        for name in ("torch", "transformers", "vllm"):
            try: manifest["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: manifest["packages"][name] = None
        write_json(mpath, manifest)


def phase_done(run_dir, run, phase):
    for index in range(len(run["gpus"])):
        mpath = phase_file(run_dir, phase, index).with_suffix(".manifest.json")
        if not mpath.exists() or load_json(mpath)["status"] != "completed":
            return False
    return True


def merge(run_dir):
    run = frozen(run_dir)
    samples = read_rows(run["samples_path"])
    per = {s["sample_id"]: {k: s[k] for k in ("sample_id", "dataset", "question", "gold_answer")} for s in samples}
    retrieval = []
    for phase in PHASES:
        if not phase_done(run_dir, run, phase):
            raise ValueError("Phase {} incomplete".format(phase))
        for index in range(len(run["gpus"])):
            retrieval.append(load_json(phase_file(run_dir, phase, index).with_suffix(".manifest.json"))["retrieval"])
            for row in read_rows(phase_file(run_dir, phase, index)):
                rec = per[row["sample_id"]]
                rec.setdefault("trajectories", {}).update(row.get("trajectories", {}))
                rec.setdefault("combos", {}).update(row["combos"])
                for key in ("null_documents", "null_observation_token_ids", "template_identical", "instruct_native_template_identical"):
                    if key in row:
                        rec[key] = row[key]
    rows = [per[s["sample_id"]] for s in samples]
    for r in rows:
        if set(r["combos"]) != {"R>R", "I>I", "R>I", "I>R"} or set(r["trajectories"]) != {"R", "I"}:
            raise ValueError("Incomplete combos for " + r["sample_id"])
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    m = dict(load_json(run_dir / "run_manifest.json"), status="completed", recorded=len(rows),
             records_sha256=file_hash(run_dir / "records.jsonl"),
             rollouts=sum(c["rollouts"] for r in rows for c in r["combos"].values()),
             template_identical=sum(bool(r.get("template_identical")) for r in rows),
             instruct_native_template_identical=sum(bool(r.get("instruct_native_template_identical")) for r in rows),
             retrieval_audit_changes=sum(s["audit_topk_set_changes"] for s in retrieval),
             retrieval_audits=sum(s["audits"] for s in retrieval))
    write_json(run_dir / "run_manifest.json", m)
    print("Merged {} questions; {} rollouts; template identical {}/{}.".format(len(rows), m["rollouts"], m["template_identical"], len(rows)))
    return m


def run(args):
    protocol, local = load_protocol(args.protocol), load_json(args.local)
    gpus = gpu_list(args.gpus)
    for model, key in CHECKPOINT_KEY.items():
        if key not in local or not (local_path(local[key]) / "config.json").is_file():
            raise ValueError("Missing local {} ({}) with config.json".format(key, model))
    samples = args.samples_dir / "samples.jsonl"
    if load_json(args.samples_dir / "selection_manifest.json")["samples_sha256"] != file_hash(samples):
        raise ValueError("Frozen samples changed")
    from .retrieval import SearchR1Retriever
    probe = SearchR1Retriever(local["retrieval_url"], args.request_timeout).search_batch(["HR1 check", "second"], protocol["retrieval"]["candidate_top_k"])
    if len(probe) != 2:
        raise ValueError("Retriever batch interface check failed")
    if args.smoke:
        rows = read_rows(samples)
        subset = [r for d in ("hotpotqa", "2wiki") for r in [x for x in rows if x["dataset"] == d][:4]]
        args.run_dir.mkdir(parents=True, exist_ok=True)
        samples = args.run_dir / "smoke_samples.jsonl"
        if not samples.exists():
            with samples.open("x", encoding="utf-8") as f:
                for r in subset:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(read_rows(samples))
    print("{} questions on GPUs {}; R={} I={}; k={}".format(n, gpus, local["agent_checkpoint"], local["instruct_checkpoint"], protocol["sampling"]["k"]))
    if args.check_only:
        print("Check passed; no model loaded."); return
    if (args.run_dir / "run_plan.json").exists():
        if load_json(args.run_dir / "run_plan.json")["repository_commit"] != git_output("rev-parse", "HEAD"):
            raise ValueError("Existing run dir has a different code commit; use a new --run-dir")
        print("Resuming existing run directory.")
    else:
        args.run_dir.mkdir(parents=True, exist_ok=True); (args.run_dir / "logs").mkdir(exist_ok=True)
        write_json(args.run_dir / "run_plan.json", {
            "version": VERSION, "protocol": protocol, "smoke": args.smoke, "protocol_sha256": sha(protocol),
            "samples_path": str(samples.resolve()), "samples_sha256": file_hash(samples), "gpus": gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization, "request_timeout": args.request_timeout,
            "local_file": str(args.local.resolve()), "local_file_sha256": file_hash(args.local),
            "repository_commit": git_output("rev-parse", "HEAD"), "working_tree_dirty": bool(git_output("status", "--porcelain")),
            "prompt_version": PROMPT_VERSION,
            "assets": {k: asset_identity(local_path(local[k]), k in ("agent_checkpoint", "instruct_checkpoint", "retriever_model"))
                       for k in ("agent_checkpoint", "instruct_checkpoint", "retriever_model", "corpus_file", "index_file")}})
        write_json(args.run_dir / "run_manifest.json", {"version": VERSION, "status": "running", "research_GO": False})
    run_plan = frozen(args.run_dir)
    for phase in PHASES:
        if phase_done(args.run_dir, run_plan, phase):
            print("phase {} already complete; skip".format(phase)); continue
        seconds = spawn(args.run_dir, gpus, "work2.handoff", ["--phase", phase], phase)
        m = load_json(args.run_dir / "run_manifest.json")
        m.setdefault("phase_wall_seconds", {})[phase] = m.get("phase_wall_seconds", {}).get(phase, 0) + seconds
        write_json(args.run_dir / "run_manifest.json", m)
    merge(args.run_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("freeze"); q.add_argument("--protocol", type=Path, default=Path("configs/handoff_v1.json"))
    q.add_argument("--local", type=Path, default=Path("configs/local.json")); q.add_argument("--exclude", type=Path, nargs="*", default=[])
    q.add_argument("--per-dataset", type=int, default=150); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("run"); q.add_argument("--protocol", type=Path, default=Path("configs/handoff_v1.json"))
    q.add_argument("--local", type=Path, required=True); q.add_argument("--samples-dir", type=Path, required=True)
    q.add_argument("--run-dir", type=Path, required=True); q.add_argument("--gpus", default="0,1,2,3")
    q.add_argument("--gpu-memory-utilization", type=float, default=.8); q.add_argument("--request-timeout", type=float, default=300)
    q.add_argument("--check-only", action="store_true"); q.add_argument("--smoke", action="store_true", help="first 4 questions per dataset; engineering only")
    q = sub.add_parser("worker"); q.add_argument("--run-dir", type=Path, required=True); q.add_argument("--index", type=int, required=True)
    q.add_argument("--phase", choices=PHASES, required=True)
    q = sub.add_parser("merge"); q.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == "freeze": freeze(a)
        elif a.command == "run": run(a)
        elif a.command == "worker": worker(a.run_dir, a.index, a.phase)
        else: merge(a.run_dir)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""U2: same-state placebo audits of two published recovery mechanisms in search agents.

ReSeek (arXiv 2510.00568; weights TencentBAC/ReSeek-qwen2.5-3b-em-grpo)
  After every observation the agent writes <judge>Yes|No</judge>; "No" is trained to
  trigger another search. From each state s_t (after observation t of the agent's
  greedy trajectory) k sampled continuations per arm:
    nat        natural continuation (the agent's own judge)
    force_no   the SAME pre-judge text, judge value forced to No
    force_yes  the SAME pre-judge text, judge value forced to Yes (recovery suppressed)
    perm       observation t shown in another document order (P1 placebo), natural judge
MR-Search (arXiv 2603.11327; no weights; test-time version = the paper's Search-R1-S)
  After a Search-R1 agent's answered greedy episode, a new user turn opens episode 2
  (fresh search budget); the final answer is the last valid answer:
    reflect    MR-Search's retry prompt, verbatim, in MR-Search's own turn format
    reask      the same text with the reflection sentence replaced by a plain re-ask
    neutral    token-matched semantically empty user turn (P2 placebo)
    fresh      an independent new episode without history (retry luck)
Inference only. Estimands, state sets and gates: research/U2_PROTOCOL.md.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path

from .common import file_hash, git_output, load_json, local_path, write_json
from .credit_testbed import LiveSearcher, batched_continue, new_rollout, obs_end, query_end, seed_for, spawn
from .datasets import read_rows
from .pcr import datasets_of, gpu_list, rank_key
from .placebo import neutral_instruction, permutations_for
from .replay import observation
from .rollout import OFFICIAL_PROMPT_VERSION, RESEEK_PROMPT_VERSION, SYSTEM_OF, question_prompt
from .run_baseline import asset_identity, validate_design
from .selective import sha

RESEEK = "U2_reseek_judge_audit_v1"
MRSEARCH = "U2_mrsearch_reflection_audit_v1"
ARMS = {RESEEK: ("nat", "force_no", "force_yes", "perm"), MRSEARCH: ("reflect", "reask", "neutral", "fresh")}
VARIANT = {RESEEK: "reseek", MRSEARCH: "official"}
PROMPT_VERSION_OF = {"reseek": RESEEK_PROMPT_VERSION, "official": OFFICIAL_PROMPT_VERSION}
JUDGE = re.compile(r"<judge>\s*(yes|no)\s*</judge>", re.IGNORECASE)
JUDGE_STOPS = ["</judge>", "</search>", "</answer>"]
# MR-Search meta-search/llm_agent/generation.py `retry_prompt`, verbatim (note the spaces before each newline).
MR_RETRY_PROMPT = ("Reflect on your current answer to the question and provide an improved answer by searching for additional "
                   "external information using search engines. \nYou must conduct reasoning inside <think> and </think> first "
                   "every time you get new information. After reasoning, if you find you lack some knowledge, you can call a "
                   "search engine by <search> query </search> and it will return the top searched results between <information> "
                   "and </information>. \nYou can search as many times as your want. If you find no further external knowledge "
                   "needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
                   "For example, <answer> Beijing </answer>.")
MR_REFLECT_SENTENCE = ("Reflect on your current answer to the question and provide an improved answer by searching for "
                       "additional external information using search engines.")
MR_REASK_PROMPT = MR_RETRY_PROMPT.replace(MR_REFLECT_SENTENCE, "Answer the question again.")


def load_protocol(path):
    p = load_json(path)
    if p.get("protocol") not in ARMS:
        raise ValueError("Unsupported U2 protocol")
    validate_design(dict(p, agent=dict(p["agent"], temperature=0)))
    if p["agent"].get("prompt_variant") != VARIANT[p["protocol"]]:
        raise ValueError("U2 {} requires prompt_variant {}".format(p["protocol"], VARIANT[p["protocol"]]))
    s = p["sampling"]
    if not 0 < s["temperature"] <= 2 or not 0 < s["top_p"] <= 1 or s["k"] < 2:
        raise ValueError("Invalid sampling settings")
    if tuple(p["arms"]) != ARMS[p["protocol"]]:
        raise ValueError("Arms are fixed by the U2 protocol: " + ", ".join(ARMS[p["protocol"]]))
    return p


def initial_tokens(generator, sample, agent):
    variant = agent["prompt_variant"]
    user = question_prompt(sample["question"], agent["max_search_turns"], variant)
    system = SYSTEM_OF.get(variant)
    text = generator.initial_prompt(user, system=system) if system is not None else generator.initial_prompt(user)
    return text, generator.encode(text)


def judge_of(text):
    m = JUDGE.search(text)
    return m.group(1).capitalize() if m else None


def greedy_trajectory(sample, prompt, generator, searcher, agent, retrieval, seed):
    traj = new_rollout(sample, prompt, 0, seed, track=True, segments=[{"kind": "prompt", "start": 0, "end": len(prompt)}])
    batched_continue([traj], generator, searcher, agent, retrieval, 0.0, 1.0)
    return traj


def summary(traj, keep_tokens=False):
    out = {"status": traj["status"], "final_answer": traj["final_answer"], "em": traj["em"], "f1": traj["f1"],
           "search_calls": traj["search_calls"], "texts": traj["texts"], "turns": traj["turns"]}
    if keep_tokens:
        out.update(token_ids=traj["tokens"], segments=traj["segments"])
    return out


def cut_at_judge(generator, ids, text, value):
    """Tokens of `text` up to and including '<judge>' (token-exact where possible) followed by '<value></judge>'."""
    pre = text[:text.index("<judge>") + len("<judge>")]
    j = len(ids)
    while j > 0 and not pre.startswith(generator.decode(ids[:j])):
        j -= 1
    head = generator.decode(ids[:j])
    return list(ids[:j]) + generator.encode(pre[len(head):] + value + "</judge>")


# ---------------- ReSeek ----------------

def reseek_question(sample, generator, searcher, protocol):
    agent, retrieval, s = protocol["agent"], protocol["retrieval"], protocol["sampling"]
    started = time.monotonic()
    text, prompt = initial_tokens(generator, sample, agent)
    traj = greedy_trajectory(sample, prompt, generator, searcher, agent, retrieval, seed_for(protocol["seed"], sample["sample_id"], "source"))
    T = traj["search_calls"]
    after = {seg["after_turn"]: txt for seg, txt in zip([x for x in traj["segments"] if x["kind"] == "gen"], traj["texts"])}
    states, branches = [], []
    # Stage A: one batched call samples, for every state, k heads that stop right after the agent's </judge>.
    prefixes = {t: traj["tokens"][:obs_end(traj, t)] for t in range(1, T + 1)}
    jobs = [(t, i) for t in range(1, T + 1) for i in range(s["k"])]
    outs = generator.sample_batch([prefixes[t] for t, _ in jobs], [agent["max_generation_tokens_per_turn"]] * len(jobs),
                                  [seed_for(protocol["seed"], sample["sample_id"], "stageA", t, i) for t, i in jobs],
                                  s["temperature"], s["top_p"], stop=JUDGE_STOPS) if jobs else []
    for t in range(1, T + 1):
        prefix = prefixes[t]
        heads = [o for (tt, _), o in zip(jobs, outs) if tt == t]
        state = {"t": t, "source_judge": judge_of(after.get(t, "")), "answer_hit": traj["turns"][t - 1]["answer_hit"],
                 "stage_a_judge": [], "arms": {a: {"em": [], "status": [], "extra_searches": [], "first_judge": []} for a in ARMS[RESEEK]}}
        for i, h in enumerate(heads):
            present = "<judge>" in h["text"] and h["text"].rstrip().endswith("</judge>")
            value = judge_of(h["text"]) if present else None
            state["stage_a_judge"].append(value if present else "absent")
            for arm in ("nat", "force_no", "force_yes"):
                seed = seed_for(protocol["seed"], sample["sample_id"], arm, t, i)
                if arm == "nat":
                    # Judge-terminated head: keep the sampled tokens as they are; otherwise resample the whole segment.
                    tokens, first = (prefix + h["token_ids"], value) if present else (prefix, None)
                else:
                    forced = "No" if arm == "force_no" else "Yes"
                    tokens = prefix + (cut_at_judge(generator, h["token_ids"], h["text"], forced) if present
                                       else generator.encode("<judge>{}</judge>".format(forced)))
                    first = forced
                branches.append((state, arm, first, new_rollout(sample, tokens, t, seed, track=True)))
        q = traj["turns"][t - 1]["query"]
        visible = searcher.search_many([q])[q][:retrieval["visible_top_k"]]
        order = permutations_for(sample["sample_id"], t, len(visible))[0]
        perm_prefix = traj["tokens"][:query_end(traj, t)] + generator.encode(observation([visible[i] for i in order]))
        for i in range(s["k"]):
            branches.append((state, "perm", None, new_rollout(sample, perm_prefix, t, seed_for(protocol["seed"], sample["sample_id"], "perm", t, i), track=True)))
        state["perm_order"] = order
        states.append(state)
    batched_continue([b[3] for b in branches], generator, searcher, agent, retrieval, s["temperature"], s["top_p"])
    for state, arm, first, r in branches:
        a = state["arms"][arm]
        a["em"].append(r["em"]); a["status"].append(r["status"])
        a["extra_searches"].append(r["search_calls"] - state["t"])
        a["first_judge"].append(first if first is not None else (judge_of(r["texts"][0]) if r["texts"] else None))
    return {"version": RESEEK, "sample_id": sample["sample_id"], "dataset": sample["dataset"], "question": sample["question"],
            "gold_answer": sample["gold_answer"], "initial_prompt": text, "source": summary(traj),
            "source_judges": [judge_of(after.get(t, "")) for t in range(1, T + 1)],
            "states": states, "branch_rollouts": len(branches), "seconds": time.monotonic() - started}


# ---------------- MR-Search ----------------

def mr_turns(generator):
    """Rendered episode-2 user turns; the neutral text matches the retry prompt's token count (+-2)."""
    target = len(generator.encode(MR_RETRY_PROMPT + "\n\n"))
    neutral = neutral_instruction(generator.encode, target).strip("\n")
    contents = {"reflect": MR_RETRY_PROMPT, "reask": MR_REASK_PROMPT, "neutral": neutral}
    return {arm: generator.user_turn(c) for arm, c in contents.items()}


def mrsearch_question(sample, generator, searcher, protocol, turns):
    agent, retrieval, s = protocol["agent"], protocol["retrieval"], protocol["sampling"]
    started = time.monotonic()
    text, prompt = initial_tokens(generator, sample, agent)
    ep1 = greedy_trajectory(sample, prompt, generator, searcher, agent, retrieval, seed_for(protocol["seed"], sample["sample_id"], "source"))
    branches = []
    for arm in ARMS[MRSEARCH]:
        for i in range(s["k"]):
            seed = seed_for(protocol["seed"], sample["sample_id"], arm, i)
            tokens = prompt if arm == "fresh" else ep1["tokens"] + generator.encode(turns[arm])
            # Episode 2 gets its own full search budget, as in MR-Search's per-episode max_turns.
            branches.append((arm, new_rollout(sample, tokens, 0, seed)))
    batched_continue([b[1] for b in branches], generator, searcher, agent, retrieval, s["temperature"], s["top_p"])
    arms = {a: {"final_em": [], "ep2_answered": [], "ep2_em": [], "ep2_searches": [], "status": [], "changed": []} for a in ARMS[MRSEARCH]}
    for arm, r in branches:
        answered = r["status"] == "answered"
        final_em = r["em"] if answered else (ep1["em"] if ep1["status"] == "answered" else 0)
        a = arms[arm]
        a["final_em"].append(final_em); a["ep2_answered"].append(answered); a["ep2_em"].append(r["em"])
        a["ep2_searches"].append(r["search_calls"]); a["status"].append(r["status"])
        a["changed"].append(answered and ep1["status"] == "answered" and r["final_answer"].strip().casefold() != ep1["final_answer"].strip().casefold())
    return {"version": MRSEARCH, "sample_id": sample["sample_id"], "dataset": sample["dataset"], "question": sample["question"],
            "gold_answer": sample["gold_answer"], "initial_prompt": text, "episode1": summary(ep1), "arms": arms,
            "branch_rollouts": len(branches), "seconds": time.monotonic() - started}


# ---------------- orchestration (same pattern as work2.credit_testbed) ----------------

def prepare(args):
    if args.output.exists():
        raise ValueError("Plan directory exists; keep frozen selections")
    samples = args.samples_dir / "samples.jsonl"
    if load_json(args.samples_dir / "selection_manifest.json")["samples_sha256"] != file_hash(samples):
        raise ValueError("Frozen U1 samples changed")
    rows = read_rows(samples)
    chosen, short = [], {}
    for dataset in datasets_of(rows):
        pool = sorted((r for r in rows if r["dataset"] == dataset), key=lambda r: rank_key(args.seed, r["sample_id"]))
        if len(pool) < args.per_dataset:
            short[dataset] = len(pool)
        chosen += pool[:args.per_dataset]
    questions = [{k: r[k] for k in ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts")} for r in chosen]
    smoke = [q["sample_id"] for d in datasets_of(questions) for q in [x for x in questions if x["dataset"] == d][:2]]
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", {
        "version": "U2_plan_v1", "seed": args.seed, "samples_path": str(samples.resolve()), "samples_sha256": file_hash(samples),
        "per_dataset": args.per_dataset, "short_datasets": short, "questions": questions, "smoke_question_ids": smoke,
        "selection": "U1 frozen questions; per dataset sha256(seed:sample_id) order, first N; outcome-blind; shared by both audits",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print("Frozen {} U2 questions (smoke {}); short datasets {}.".format(len(questions), len(smoke), short or "none"))


def frozen(run_dir):
    run = load_json(run_dir / "run_plan.json")
    if file_hash(run["plan_path"]) != run["plan_sha256"] or sha(run["protocol"]) != run["protocol_sha256"]:
        raise ValueError("Plan or protocol changed after freezing")
    return run, load_json(run["plan_path"])


def shard(run, plan, index):
    by_id = {q["sample_id"]: q for q in plan["questions"]}
    return [by_id[i] for i in run["question_ids"][index::len(run["gpus"])]]


def worker(run_dir, index):
    run, plan = frozen(run_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != run["gpus"][index]:
        raise ValueError("Worker GPU binding mismatch")
    if git_output("rev-parse", "HEAD") != run["repository_commit"]:
        raise ValueError("Code changed after run freezing")
    if file_hash(run["local_file"]) != run["local_file_sha256"]:
        raise ValueError("Local resource configuration changed")
    local, protocol = load_json(run["local_file"]), run["protocol"]
    questions = shard(run, plan, index)
    folder = run_dir / "workers" / ("worker_" + str(index)); folder.mkdir(parents=True, exist_ok=True)
    out = folder / "records.jsonl"
    done = read_rows(out) if out.exists() else []
    if [r["sample_id"] for r in done] != [q["sample_id"] for q in questions[:len(done)]]:
        raise ValueError("Existing worker output does not match frozen order")
    manifest = {"version": protocol["protocol"], "worker_index": index, "gpu": run["gpus"][index], "status": "loading",
                "expected": len(questions), "recorded": len(done), "resumed_from": len(done),
                "retrieval_sessions": (load_json(folder / "run_manifest.json").get("retrieval_sessions", [])
                                       if (folder / "run_manifest.json").exists() else [])}
    write_json(folder / "run_manifest.json", manifest)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from .backends import VLLMGenerator
    from .retrieval import SearchR1Retriever
    generator = VLLMGenerator(local_path(local["agent_checkpoint"]), protocol["agent"], 1, run["gpu_memory_utilization"])
    searcher = LiveSearcher(SearchR1Retriever(local["retrieval_url"], run["request_timeout"]), protocol["retrieval"])
    turns = mr_turns(generator) if protocol["protocol"] == MRSEARCH else None
    if turns is not None:
        manifest["user_turns"] = turns
    started = time.monotonic()
    try:
        with out.open("a", encoding="utf-8") as stream:
            for q in questions[len(done):]:
                if protocol["protocol"] == RESEEK:
                    row = reseek_question(q, generator, searcher, protocol)
                    note = "T={} judges={}".format(row["source"]["search_calls"], row["source_judges"])
                else:
                    row = mrsearch_question(q, generator, searcher, protocol, turns)
                    note = "ep1 em={} reflect={}".format(row["episode1"]["em"], row["arms"]["reflect"]["final_em"])
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
                manifest.update(status="running", recorded=manifest["recorded"] + 1)
                write_json(folder / "run_manifest.json", manifest)
                print("worker {} {}/{} {} {} {:.0f}s".format(index, manifest["recorded"], len(questions), q["sample_id"], note, row["seconds"]), flush=True)
        manifest.update(status="completed")
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        manifest["seconds_this_session"] = time.monotonic() - started
        manifest.setdefault("retrieval_sessions", []).append(searcher.stats())
        manifest["packages"] = {}
        for name in ("torch", "transformers", "vllm"):
            try: manifest["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: manifest["packages"][name] = None
        write_json(folder / "run_manifest.json", manifest)


def merge(run_dir):
    run, plan = frozen(run_dir)
    rows, sessions, turns = [], [], None
    for index in range(len(run["gpus"])):
        folder = run_dir / "workers" / ("worker_" + str(index))
        m = load_json(folder / "run_manifest.json")
        got = read_rows(folder / "records.jsonl")
        if m["status"] != "completed" or [r["sample_id"] for r in got] != [q["sample_id"] for q in shard(run, plan, index)]:
            raise ValueError("Worker {} incomplete or out of order".format(index))
        rows += got
        sessions += m.get("retrieval_sessions", [])
        if m.get("user_turns") is not None:
            if turns is not None and m["user_turns"] != turns:
                raise ValueError("Workers rendered different user turns")
            turns = m["user_turns"]
    order = {q: i for i, q in enumerate(run["question_ids"])}
    rows.sort(key=lambda r: order[r["sample_id"]])
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    audits = sum(s["audits"] for s in sessions)
    changes = sum(s["audit_topk_set_changes"] for s in sessions)
    m = load_json(run_dir / "run_manifest.json")
    m.update(status="completed", recorded=len(rows), records_sha256=file_hash(run_dir / "records.jsonl"),
             branch_rollouts=sum(r["branch_rollouts"] for r in rows), user_turns=turns,
             retrieval={"sessions": len(sessions), "audits": audits, "audit_topk_set_changes": changes,
                        "retried_batches": sum(s["retried_batches"] for s in sessions)})
    write_json(run_dir / "run_manifest.json", m)
    print("Merged {} questions; {} branch rollouts; retrieval audit Top-k set changes {}/{}.".format(len(rows), m["branch_rollouts"], changes, audits))
    return m


def run(args):
    protocol, plan = load_protocol(args.protocol), load_json(args.plan_dir / "plan.json")
    gpus, local = gpu_list(args.gpus), load_json(args.local)
    if not (local_path(local["agent_checkpoint"]) / "config.json").is_file():
        raise ValueError("Missing local checkpoint config.json")
    from .retrieval import SearchR1Retriever
    probe = SearchR1Retriever(local["retrieval_url"], args.request_timeout).search_batch(["U2 service check", "second query"], protocol["retrieval"]["candidate_top_k"])
    if len(probe) != 2 or len(probe[0]) < protocol["retrieval"]["visible_top_k"]:
        raise ValueError("Retriever batch interface check failed")
    ids = plan["smoke_question_ids"] if args.smoke else [q["sample_id"] for q in plan["questions"]]
    print("{}: {} questions on GPUs {}; agent {}; k={}.".format(protocol["protocol"], len(ids), gpus, local["agent_checkpoint"], protocol["sampling"]["k"]))
    if args.check_only:
        print("Check passed; no model loaded."); return
    if args.run_dir.exists():
        old = load_json(args.run_dir / "run_plan.json")
        if old["question_ids"] != ids or old["gpus"] != gpus or old["repository_commit"] != git_output("rev-parse", "HEAD") \
                or old["protocol_sha256"] != sha(protocol):
            raise ValueError("Existing run dir differs (questions/GPUs/commit/protocol); use a new --run-dir")
        print("Resuming existing run directory.")
    else:
        args.run_dir.mkdir(parents=True); (args.run_dir / "logs").mkdir()
        write_json(args.run_dir / "run_plan.json", {
            "version": protocol["protocol"], "protocol": protocol, "protocol_sha256": sha(protocol), "agent_label": args.label,
            "plan_path": str((args.plan_dir / "plan.json").resolve()), "plan_sha256": file_hash(args.plan_dir / "plan.json"),
            "question_ids": ids, "smoke": args.smoke, "gpus": gpus, "gpu_memory_utilization": args.gpu_memory_utilization,
            "request_timeout": args.request_timeout, "local_file": str(args.local.resolve()), "local_file_sha256": file_hash(args.local),
            "repository_commit": git_output("rev-parse", "HEAD"), "working_tree_dirty": bool(git_output("status", "--porcelain")),
            "prompt_version": PROMPT_VERSION_OF[protocol["agent"]["prompt_variant"]],
            "assets": {k: asset_identity(local_path(local[k]), k in ("agent_checkpoint", "retriever_model"))
                       for k in ("agent_checkpoint", "retriever_model", "corpus_file", "index_file") if k in local}})
        write_json(args.run_dir / "run_manifest.json", {"version": protocol["protocol"], "status": "running", "smoke": args.smoke,
                                                         "planned": len(ids), "research_GO": False})
    seconds = spawn(args.run_dir, gpus, "work2.u2", [], "u2")
    m = load_json(args.run_dir / "run_manifest.json")
    write_json(args.run_dir / "run_manifest.json", dict(m, wall_seconds=m.get("wall_seconds", 0) + seconds))
    merge(args.run_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--samples-dir", type=Path, required=True)
    q.add_argument("--per-dataset", type=int, default=300); q.add_argument("--seed", type=int, default=20261023)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("run"); q.add_argument("--protocol", type=Path, required=True); q.add_argument("--plan-dir", type=Path, required=True)
    q.add_argument("--run-dir", type=Path, required=True); q.add_argument("--local", type=Path, required=True)
    q.add_argument("--label", required=True, help="agent label recorded in run_plan.json")
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

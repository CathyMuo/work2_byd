"""AC1 confirmation of the interventional-audit claims on NEVER-USED questions.

freeze   -> outcome-blind fresh questions, excluding every historical cohort by id AND question text
sources  -> greedy E0 source rollouts for one checkpoint (N GPUs, strict retrieval drift checks, resume)
report   -> mechanical H1/H2/H3 verdicts from a PCR (arm group audit) and a CT1 (k=16) analysis
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

from .common import file_hash, git_output, load_json, local_path, write_json
from .credit_testbed import spawn
from .datasets import canonicalize, read_rows
from .pcr import gpu_list, rank_key
from .replay import AuditedRetriever
from .replay import PROMPT_VARIANT_OF
from .rollout import run_sample
from .run_baseline import asset_identity, validate_design
from .selective import exclusions, question_key, sha

VERSION = "AC1_interventional_audit_confirmation_v1"
FC1_VERSION = "FC1_harness_fidelity_v1"  # reuses the E0 source runner with a prompt variant
DEFAULT_EXCLUDE = ("outputs/samples/pilot.jsonl", "outputs/a1_clean_scale500/clean_rollouts.jsonl",
                   "outputs/si1_samples/samples.jsonl", "outputs/si1_dp4_samples/samples.jsonl")


def load_protocol(path):
    p = load_json(path)
    if p.get("protocol") not in (VERSION, FC1_VERSION):
        raise ValueError("Unsupported AC1/FC1 protocol")
    if p["agent"].get("prompt_variant", "bounded") not in set(PROMPT_VARIANT_OF.values()):
        raise ValueError("Unknown prompt variant")
    validate_design(p)
    return p


def select_fresh(local, paths, per_dataset, seed):
    """Outcome-blind fresh questions; excluded by id AND normalized question; cross-dataset dedup."""
    if not paths:
        raise ValueError("No historical cohorts to exclude were found; pass --exclude explicitly")
    ids, questions, receipts = exclusions(paths)
    chosen, seen = [], set()
    for dataset, key in (("hotpotqa", "hotpotqa_file"), ("2wiki", "two_wiki_file")):
        rows = [canonicalize(r, dataset) for r in read_rows(local_path(local[key]))]
        pool = sorted((r for r in rows if r["sample_id"] not in ids and question_key(r["question"]) not in questions),
                      key=lambda r: rank_key(seed, r["sample_id"]))
        picked = []
        for r in pool:
            q = question_key(r["question"])
            if q in seen:
                continue
            seen.add(q); picked.append(r)
            if len(picked) == per_dataset:
                break
        if len(picked) < per_dataset:
            raise ValueError("Only {} unused {} questions".format(len(picked), dataset))
        chosen += picked
    return chosen, receipts, ids, questions


def write_selection(output, chosen, manifest):
    output.mkdir(parents=True)
    with (output / "samples.jsonl").open("x", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(output / "selection_manifest.json", dict(manifest, sample_ids=[r["sample_id"] for r in chosen],
                                                       samples_sha256=file_hash(output / "samples.jsonl"),
                                                       created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


def freeze(args):
    protocol, local = load_protocol(args.protocol), load_json(args.local)
    if args.output.exists():
        raise ValueError("Output exists; keep frozen selections")
    paths = list(args.exclude) + [Path(p) for p in DEFAULT_EXCLUDE if Path(p).is_file() and Path(p) not in args.exclude]
    chosen, receipts, ids, questions = select_fresh(local, paths, args.per_dataset, protocol["seed"])
    args.output.mkdir(parents=True)
    with (args.output / "samples.jsonl").open("x", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(args.output / "selection_manifest.json", {
        "version": VERSION, "protocol_sha256": sha(protocol), "seed": protocol["seed"], "per_dataset": args.per_dataset,
        "exclusions": receipts, "excluded_ids": len(ids), "excluded_questions": len(questions),
        "sample_ids": [r["sample_id"] for r in chosen], "samples_sha256": file_hash(args.output / "samples.jsonl"),
        "selection": "sha256(seed:sample_id) order per dataset; excluded by id and normalized question; cross-dataset question dedup; outcome-blind",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print("Frozen {} fresh questions; excluded {} historical ids / {} questions from {} files.".format(
        len(chosen), len(ids), len(questions), len(receipts)))


def frozen(run_dir):
    run = load_json(run_dir / "run_plan.json")
    if file_hash(run["samples_path"]) != run["samples_sha256"] or sha(run["protocol"]) != run["protocol_sha256"]:
        raise ValueError("Samples or protocol changed after freezing")
    return run


def worker(run_dir, index):
    run = frozen(run_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != run["gpus"][index]:
        raise ValueError("Worker GPU binding mismatch")
    if git_output("rev-parse", "HEAD") != run["repository_commit"]:
        raise ValueError("Code changed after run freezing")
    if file_hash(run["local_file"]) != run["local_file_sha256"]:
        raise ValueError("Local resource configuration changed")
    local, protocol = load_json(run["local_file"]), run["protocol"]
    samples = read_rows(run["samples_path"])[index::len(run["gpus"])]
    folder = run_dir / "workers" / ("worker_" + str(index)); folder.mkdir(parents=True, exist_ok=True)
    out = folder / "sources.jsonl"
    done = read_rows(out) if out.exists() else []
    if [r["sample_id"] for r in done] != [s["sample_id"] for s in samples[:len(done)]]:
        raise ValueError("Existing worker output does not match frozen order")
    manifest = {"version": VERSION, "worker_index": index, "gpu": run["gpus"][index], "status": "loading",
                "expected": len(samples), "recorded": len(done)}
    write_json(folder / "run_manifest.json", manifest)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from .backends import VLLMGenerator
    from .retrieval import SearchR1Retriever
    agent, retrieval = protocol["agent"], protocol["retrieval"]
    generator = VLLMGenerator(local_path(local["agent_checkpoint"]), agent, 1, run["gpu_memory_utilization"])
    retriever = AuditedRetriever(SearchR1Retriever(local["retrieval_url"], run["request_timeout"]), retrieval["visible_top_k"])
    for row in done:
        for p in row["prefixes"]:
            retriever.seed(p["search_query"], p["visible_documents"])
    started = time.monotonic()
    try:
        with out.open("a", encoding="utf-8") as stream:
            for sample in samples[len(done):]:
                row = run_sample(sample, generator, retriever, agent, retrieval)
                if row["status"] == "infrastructure_error":
                    raise RuntimeError(str(row.get("error")))
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
                manifest.update(status="running", recorded=manifest["recorded"] + 1)
                write_json(folder / "run_manifest.json", manifest)
                if manifest["recorded"] % 25 == 0:
                    print("worker {} {}/{}".format(index, manifest["recorded"], len(samples)), flush=True)
        manifest.update(status="completed")
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        manifest["seconds_this_session"] = time.monotonic() - started
        manifest["retrieval_calls_this_session"] = len(retriever.calls)
        write_json(folder / "run_manifest.json", manifest)


def merge(run_dir):
    run = frozen(run_dir)
    order = [r["sample_id"] for r in read_rows(run["samples_path"])]
    rows = []
    for index in range(len(run["gpus"])):
        folder = run_dir / "workers" / ("worker_" + str(index))
        if load_json(folder / "run_manifest.json")["status"] != "completed":
            raise ValueError("Worker {} incomplete".format(index))
        rows += read_rows(folder / "sources.jsonl")
    pos = {s: i for i, s in enumerate(order)}
    if sorted(pos[r["sample_id"]] for r in rows) != list(range(len(order))):
        raise ValueError("Merged coverage missing or duplicated")
    rows.sort(key=lambda r: pos[r["sample_id"]])
    with (run_dir / "sources.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    em = sum(r["em"] for r in rows) / len(rows)
    m = dict(load_json(run_dir / "run_manifest.json"), status="completed", recorded=len(rows), clean_em=em,
             searched=sum(bool(r["prefixes"]) for r in rows), sources_sha256=file_hash(run_dir / "sources.jsonl"))
    write_json(run_dir / "run_manifest.json", m)
    print("Merged {} E0 sources; clean EM {:.4f}; {} with >=1 search.".format(len(rows), em, m["searched"]))


def sources(args):
    protocol, local = load_protocol(args.protocol), load_json(args.local)
    gpus = gpu_list(args.gpus)
    if not (local_path(local["agent_checkpoint"]) / "config.json").is_file():
        raise ValueError("Missing local checkpoint config.json: " + str(local["agent_checkpoint"]))
    samples = args.samples_dir / "samples.jsonl"
    manifest = load_json(args.samples_dir / "selection_manifest.json")
    if manifest["samples_sha256"] != file_hash(samples):
        raise ValueError("Frozen samples changed")
    print("{} fresh questions on GPUs {}; checkpoint {}".format(len(manifest["sample_ids"]), gpus, local["agent_checkpoint"]))
    if args.check_only:
        print("Check passed; no model loaded."); return
    if args.run_dir.exists():
        if load_json(args.run_dir / "run_plan.json")["repository_commit"] != git_output("rev-parse", "HEAD"):
            raise ValueError("Existing run dir has a different code commit; use a new --run-dir")
        print("Resuming existing run directory.")
    else:
        args.run_dir.mkdir(parents=True); (args.run_dir / "logs").mkdir()
        write_json(args.run_dir / "run_plan.json", {
            "version": VERSION, "protocol": protocol, "protocol_sha256": sha(protocol),
            "samples_path": str(samples.resolve()), "samples_sha256": file_hash(samples), "gpus": gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization, "request_timeout": args.request_timeout,
            "local_file": str(args.local.resolve()), "local_file_sha256": file_hash(args.local),
            "repository_commit": git_output("rev-parse", "HEAD"), "working_tree_dirty": bool(git_output("status", "--porcelain")),
            "assets": {k: asset_identity(local_path(local[k]), k in ("agent_checkpoint", "retriever_model"))
                       for k in ("agent_checkpoint", "retriever_model", "corpus_file", "index_file")},
            "declared_asset_versions": {k: local.get(k) for k in ("checkpoint_version", "corpus_version", "index_version", "actual_index_type")}})
        write_json(args.run_dir / "run_manifest.json", {"version": VERSION, "status": "running"})
    seconds = spawn(args.run_dir, gpus, "work2.audit_confirm", [], "sources")
    m = load_json(args.run_dir / "run_manifest.json")
    write_json(args.run_dir / "run_manifest.json", dict(m, wall_seconds=m.get("wall_seconds", 0) + seconds))
    merge(args.run_dir)


def verdicts(pcr, ct1):
    out = {}
    methods = pcr["recovery"]["semantic"].get("selected_fragile", {}).get("methods", {}) if pcr["recovery"]["semantic"].get("available") else {}
    h1 = {}
    for m in ("review", "reflect"):
        v = methods.get(m)
        h1[m] = None if v is None else bool((v["placebo_explained_fraction"] or 0) >= 1 / 3 or v["direction_reversed"])
    out["H1_recovery_explained_by_placebo"] = {"per_method": h1, "confirmed": any(x for x in h1.values() if x),
                                               "degraded_states": pcr["recovery"]["semantic"].get("selected_fragile", {}).get("degraded_states")}
    dele = pcr["deletion"]["semantic"]
    ci = dele["all_matched"]["support_minus_nonsupport"]["ci95"] if dele.get("available") else None
    out["H2_support_deletion_hurts_more"] = {"ci95": ci, "matched_states": dele.get("matched_states"),
                                             "confirmed": bool(ci and ci[1] < 0)}
    rel_e = ct1["reliability"]["E"]["spearman_brown"]
    inf = (ct1.get("exploratory") or {}).get("informative_signals") or {}
    per = {}
    for s in ("ig", "ealr", "fact_hit"):
        a = inf.get(s, {}).get("vs_E") if inf.get(s, {}).get("available") else None
        per[s] = None if a is None else {"spearman": a["spearman"], "ci95": a["ci95"],
                                         "below": bool(a["spearman"] is not None and a["spearman"] < .3 and a["ci95"] and a["ci95"][1] < .4)}
    out["H3_evidence_signals_do_not_track_E"] = {"E_reliability": rel_e, "E_reliability_ok": bool(rel_e is not None and rel_e >= .8),
                                                  "per_signal": per,
                                                  # fact_hit is undefined (constant) on datasets without supporting facts
                                                  # (FlashRAG NQ/Bamboogle/...): it is then skipped; ig and ealr are required.
                                                  "confirmed": bool(rel_e is not None and rel_e >= .8 and per.get("ig") and per.get("ealr")
                                                                    and all(v["below"] for v in per.values() if v and v["spearman"] is not None))}
    out["n_confirmed"] = sum(out[k]["confirmed"] for k in ("H1_recovery_explained_by_placebo", "H2_support_deletion_hurts_more", "H3_evidence_signals_do_not_track_E"))
    return out


def report(args):
    pcr, ct1 = load_json(args.pcr_report), load_json(args.ct1_report)
    v = verdicts(pcr, ct1)
    result = {"version": VERSION, "label": args.label, "analysis_commit": git_output("rev-parse", "HEAD"),
              "inputs": {"pcr_report_sha256": file_hash(args.pcr_report), "ct1_report_sha256": file_hash(args.ct1_report),
                         "pcr_states_sha256": pcr.get("states_sha256"), "ct1_records_sha256": ct1.get("records_sha256")},
              "verdicts": v, "descriptive": {"lucky_share": pcr["luck"]["semantic"]["lucky_share_of_accuracy"],
                                             "clean_em": pcr["luck"]["semantic"]["clean_em"],
                                             "A_reliability": ct1["reliability"]["A"]["spearman_brown"],
                                             "mean_A": ct1["truth"]["mean_A"], "mean_E": ct1["truth"]["mean_E"],
                                             "answer_hit_vs_E": ct1["signals"].get("answer_hit", {}).get("vs_E")},
              "note": "Mechanical verdicts per research/AUDIT_CONFIRM_PROTOCOL.md section 4; the human decision goes to DECISION_LOG.md."}
    if args.output.exists():
        raise ValueError("Output exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    lines = ["# AC1 确认结果：{}".format(args.label), ""]
    for k in ("H1_recovery_explained_by_placebo", "H2_support_deletion_hurts_more", "H3_evidence_signals_do_not_track_E"):
        lines.append("- **{}**：{}　{}".format(k, "确认" if v[k]["confirmed"] else "未确认", json.dumps({x: y for x, y in v[k].items() if x != "confirmed"}, ensure_ascii=False)))
    lines += ["", "确认条数：{}/3".format(v["n_confirmed"]), "", "描述量：" + json.dumps(result["descriptive"], ensure_ascii=False)]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("freeze"); q.add_argument("--protocol", type=Path, default=Path("configs/audit_confirm_v1.json"))
    q.add_argument("--local", type=Path, default=Path("configs/local.json")); q.add_argument("--exclude", type=Path, nargs="*", default=[])
    q.add_argument("--per-dataset", type=int, default=500); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("sources"); q.add_argument("--protocol", type=Path, default=Path("configs/audit_confirm_v1.json"))
    q.add_argument("--local", type=Path, default=Path("configs/local.json")); q.add_argument("--samples-dir", type=Path, required=True)
    q.add_argument("--run-dir", type=Path, required=True); q.add_argument("--gpus", default="0,1,2,3")
    q.add_argument("--gpu-memory-utilization", type=float, default=.75); q.add_argument("--request-timeout", type=float, default=120)
    q.add_argument("--check-only", action="store_true")
    q = sub.add_parser("worker"); q.add_argument("--run-dir", type=Path, required=True); q.add_argument("--index", type=int, required=True)
    q = sub.add_parser("merge"); q.add_argument("--run-dir", type=Path, required=True)
    q = sub.add_parser("report"); q.add_argument("--pcr-report", type=Path, required=True); q.add_argument("--ct1-report", type=Path, required=True)
    q.add_argument("--label", required=True); q.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == "worker": worker(a.run_dir, a.index)
        elif a.command == "merge": merge(a.run_dir)
        else: {"freeze": freeze, "sources": sources, "report": report}[a.command](a)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

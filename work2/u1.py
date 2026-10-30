"""U1: the H1/H3 audit across many search agents and datasets.

freeze   -> U1 question set: the AC1 fresh HotpotQA/2Wiki questions (same questions as AC1) plus
            outcome-blind samples of FlashRAG-format NQ / MuSiQue / Bamboogle (no history overlap)
check    -> every roster agent has a local config with a loadable checkpoint
summary  -> cross-agent table: per agent and per dataset clean EM, H1 explained fractions,
            H3 signal agreement on informative turns, mechanical verdicts and the pooled claim
The per-agent runs reuse existing engines (audit_confirm sources, pcr, credit_testbed,
credit_signals, *_analysis, audit_confirm report); see scripts/run_u1.sh.
"""
import argparse
import json
import math
from pathlib import Path

from .audit_confirm import DEFAULT_EXCLUDE, verdicts, write_selection
from .common import file_hash, git_output, load_json, local_path, write_json
from .credit_analysis import agreement, reliability, turn_rows
from .datasets import canonicalize_flashrag, read_rows
from .placebo import usable_states
from .pcr import rank_key
from .pcr_analysis import derive, recovery
from .selective import exclusions, question_key, sha

VERSION = "U1_multi_agent_audit_v1"
SIGNALS = ("ig", "ealr", "fact_hit", "answer_hit")


def load_protocol(path):
    p = load_json(path)
    if p.get("protocol") != VERSION:
        raise ValueError("Unsupported U1 protocol")
    return p


def freeze(args):
    protocol, local = load_protocol(args.protocol), load_json(args.local)
    if args.output.exists():
        raise ValueError("Output exists; keep frozen selections")
    ac1 = read_rows(args.ac1_fresh / "samples.jsonl")
    if load_json(args.ac1_fresh / "selection_manifest.json")["samples_sha256"] != file_hash(args.ac1_fresh / "samples.jsonl"):
        raise ValueError("AC1 fresh samples changed")
    chosen = [r for r in ac1 if r["dataset"] in ("hotpotqa", "2wiki")]
    seen = {question_key(r["question"]) for r in chosen}
    paths = list(args.exclude) + [Path(p) for p in DEFAULT_EXCLUDE + ("outputs/hr1_fresh/samples.jsonl",)
                                  if Path(p).is_file() and Path(p) not in args.exclude]
    _, history_questions, receipts = exclusions(paths) if paths else (set(), set(), [])
    missing, counts = {}, {"hotpotqa": sum(r["dataset"] == "hotpotqa" for r in chosen), "2wiki": sum(r["dataset"] == "2wiki" for r in chosen)}
    for dataset, spec in protocol["new_datasets"].items():
        key = spec["local_key"]
        if key not in local or not local_path(local[key]).is_file():
            missing[dataset] = "local key {} missing or file not found".format(key)
            continue
        rows = [canonicalize_flashrag(r, dataset) for r in read_rows(local_path(local[key]))]
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate ids in " + dataset)
        pool = sorted((r for r in rows if question_key(r["question"]) not in history_questions),
                      key=lambda r: rank_key(protocol["seed"], r["sample_id"]))
        limit = spec["per_dataset"] if spec["per_dataset"] > 0 else len(pool)
        picked = []
        for r in pool:
            q = question_key(r["question"])
            if q in seen:
                continue
            seen.add(q); picked.append(r)
            if len(picked) == limit:
                break
        chosen += picked
        counts[dataset] = len(picked)
    write_selection(args.output, chosen, {"version": VERSION, "protocol_sha256": sha(protocol), "seed": protocol["seed"],
                                          "counts": counts, "missing_datasets": missing, "exclusions": receipts,
                                          "ac1_fresh_sha256": file_hash(args.ac1_fresh / "samples.jsonl"),
                                          "selection": "AC1 fresh HotpotQA/2Wiki reused verbatim; new datasets by sha256(seed:sample_id), excluding historical question text, cross-dataset dedup; outcome-blind"})
    print("Frozen U1 questions: {}; missing: {}".format(counts, missing or "none"))


def check(args):
    roster = load_json(args.roster)["agents"]
    ok = True
    for a in roster:
        cfg = Path("configs/local.u1_{}.json".format(a["label"]))
        if not cfg.is_file():
            print("{:<16} local config {} not found".format(a["label"], cfg)); ok = False; continue
        path = local_path(load_json(cfg)["agent_checkpoint"])
        good = (path / "config.json").is_file()
        ok &= good
        print("{:<16} {:<8} {}".format(a["label"], "OK" if good else "MISSING", path))
    print("All roster checkpoints present." if ok else "Some roster checkpoints are missing (they will be skipped by run_u1.sh all).")


def rate(xs):
    return sum(xs) / len(xs) if xs else None


def h1_block(states, resamples):
    rec = recovery([derive(s) for s in states], "sem", resamples) if states else {"available": False}
    if not rec.get("available"):
        return {"available": False}
    sf = rec["selected_fragile"]
    out = {"degraded_states": sf["degraded_states"], "stable_states": sf["stable_states"],
           "placebo_repair": sf["placebo_repair_rates"], "methods": {}}
    for m in ("review", "reflect"):
        v = sf["methods"].get(m)
        if v:
            out["methods"][m] = {"repair": v["repair_rate_on_degraded"], "explained": v["placebo_explained_fraction"],
                                 "reversed": v["direction_reversed"],
                                 "confirmed": bool((v["placebo_explained_fraction"] or 0) >= 1 / 3 or v["direction_reversed"]),
                                 "repair_minus_reperturb": v.get("repair_minus_reperturb")}
    out["confirmed"] = any(v["confirmed"] for v in out["methods"].values())
    return out


def h3_block(records, signals, resamples):
    rows = [r for rec in records for r in turn_rows(rec, signals.get(rec["sample_id"]))]
    if len(rows) < 10:
        return {"available": False, "turns": len(rows)}
    inf = [r for r in rows if r["informative"]]
    out = {"turns": len(rows), "informative_turns": len(inf),
           "E_reliability": reliability(rows, "E", resamples)["spearman_brown"], "signals": {}}
    for s in SIGNALS:
        a = agreement(inf, s, "E", resamples) if len(inf) >= 10 else None
        out["signals"][s] = None if not a or a["spearman"] is None else {"spearman": a["spearman"], "ci95": a["ci95"]}
    ev = [out["signals"][s] for s in ("ig", "ealr", "fact_hit")]
    out["confirmed"] = bool(out["E_reliability"] is not None and out["E_reliability"] >= .8
                            and out["signals"]["ig"] and out["signals"]["ealr"]
                            and all(v["spearman"] < .3 and v["ci95"] and v["ci95"][1] < .4 for v in ev if v))
    return out


def agent_summary(root, label, resamples):
    d = Path(root) / label
    out = {"label": label}
    sources = read_rows(d / "sources" / "sources.jsonl") if (d / "sources" / "sources.jsonl").exists() else []
    datasets = sorted({r["dataset"] for r in sources})
    out["clean_em"] = {ds: rate([r["em"] for r in sources if r["dataset"] == ds]) for ds in datasets}
    out["clean_em"]["all"] = rate([r["em"] for r in sources])
    fail = ("malformed_action", "generation_limit", "context_limit")
    out["format_fail_rate"] = rate([r["status"] in fail for r in sources])
    pcr_states = usable_states(read_rows(d / "pcr" / "states.jsonl")) if (d / "pcr" / "states.jsonl").exists() else []
    ct_records = read_rows(d / "ct" / "records.jsonl") if (d / "ct" / "records.jsonl").exists() else []
    sig_path = d / "ct" / "signals_policy.jsonl"
    signals = {r["sample_id"]: r for r in read_rows(sig_path)} if sig_path.exists() else {}
    pcr_report, ct_report = d / "pcr" / "analysis" / "pcr_report.json", d / "ct" / "analysis" / "ct1_report.json"
    if pcr_report.exists() and ct_report.exists():
        out["pooled_verdicts"] = verdicts(load_json(pcr_report), load_json(ct_report))
    out["by_dataset"] = {ds: {"H1": h1_block([s for s in pcr_states if s["dataset"] == ds], resamples // 4),
                              "H3": h3_block([r for r in ct_records if r["dataset"] == ds], signals, resamples // 4)}
                         for ds in sorted({s["dataset"] for s in pcr_states} | {r["dataset"] for r in ct_records})}
    out["pooled"] = {"H1": h1_block(pcr_states, resamples), "H3": h3_block(ct_records, signals, resamples)}
    return out


def summary(args):
    roster = load_json(args.roster)["agents"]
    agents = [agent_summary(args.root, a["label"], args.resamples) for a in roster if (Path(args.root) / a["label"]).is_dir()]
    if not agents:
        raise ValueError("No agent run directories under " + str(args.root))
    format_ok = [a for a in agents if a["format_fail_rate"] is not None and a["format_fail_rate"] < .10]
    n = len(format_ok)
    need = math.ceil(.8 * n) if n else None
    h1 = sum(bool(a["pooled"]["H1"].get("confirmed")) for a in format_ok)
    h3 = sum(bool(a["pooled"]["H3"].get("confirmed")) for a in format_ok)
    report = {"version": VERSION, "analysis_commit": git_output("rev-parse", "HEAD"), "agents": agents,
              "included_agents": [a["label"] for a in format_ok],
              "excluded_for_format_failure": [a["label"] for a in agents if a not in format_ok],
              "claim": {"H1_agents_confirmed": h1, "H3_agents_confirmed": h3, "included": n, "needed_for_claim": need,
                        "H1_holds": bool(need and h1 >= need), "H3_holds": bool(need and h3 >= need)},
              "note": "Mechanical; criteria in research/U1_PROTOCOL.md. Human decision goes to DECISION_LOG.md."}
    if args.output.exists():
        raise ValueError("Output exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


def f3(x):
    return "NA" if x is None else "{:.3f}".format(x)


def markdown(r):
    L = ["# U1 跨智能体审计汇总（自动生成，供回传）", "",
         "纳入智能体：{}；因格式失败排除：{}".format(", ".join(r["included_agents"]) or "无", ", ".join(r["excluded_for_format_failure"]) or "无"),
         "**H1 成立的智能体 {}/{}，H3 成立的智能体 {}/{}（需要 ≥ {}）**".format(
             r["claim"]["H1_agents_confirmed"], r["claim"]["included"], r["claim"]["H3_agents_confirmed"], r["claim"]["included"], r["claim"]["needed_for_claim"]), "",
         "| agent | clean EM | 格式失败 | H1 D | review explained | reflect explained | H1 | E 信度 | ig | ealr | fact_hit | answer_hit | H3 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in r["agents"]:
        h1, h3 = a["pooled"]["H1"], a["pooled"]["H3"]
        m = h1.get("methods", {})
        s = h3.get("signals", {})
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            a["label"], f3(a["clean_em"].get("all")), f3(a["format_fail_rate"]), h1.get("degraded_states", "NA"),
            f3(m.get("review", {}).get("explained")), f3(m.get("reflect", {}).get("explained")), h1.get("confirmed"),
            f3(h3.get("E_reliability")), *[f3((s.get(k) or {}).get("spearman")) for k in SIGNALS], h3.get("confirmed")))
    L += ["", "## 分数据集（clean EM / H1 D 与 review explained / H3 ig·ealr）", ""]
    for a in r["agents"]:
        parts = []
        for ds, v in a["by_dataset"].items():
            h1, h3 = v["H1"], v["H3"]
            parts.append("{}: EM {} | D {} rev {} | ig {} ealr {}".format(
                ds, f3(a["clean_em"].get(ds)), h1.get("degraded_states", "NA"), f3(h1.get("methods", {}).get("review", {}).get("explained")),
                f3((h3.get("signals", {}).get("ig") or {}).get("spearman")), f3((h3.get("signals", {}).get("ealr") or {}).get("spearman"))))
        L.append("- **{}**：{}".format(a["label"], "；".join(parts)))
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("freeze"); q.add_argument("--protocol", type=Path, default=Path("configs/u1_protocol.json"))
    q.add_argument("--local", type=Path, default=Path("configs/local.json")); q.add_argument("--ac1-fresh", type=Path, default=Path("outputs/ac1_fresh"))
    q.add_argument("--exclude", type=Path, nargs="*", default=[]); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("check"); q.add_argument("--roster", type=Path, default=Path("configs/u1_agents.json"))
    q = sub.add_parser("summary"); q.add_argument("--roster", type=Path, default=Path("configs/u1_agents.json"))
    q.add_argument("--root", type=Path, default=Path("outputs/u1")); q.add_argument("--resamples", type=int, default=1000)
    q.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        {"freeze": freeze, "check": check, "summary": summary}[a.command](a)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

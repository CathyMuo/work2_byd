"""FC1 harness-fidelity summary over E0 source runs (one run per checkpoint x prompt variant).

Each --run LABEL=DIR points at an `audit_confirm sources` run directory (merged
sources.jsonl). Reports greedy EM per dataset with question-cluster bootstrap CIs,
behaviour rates, and paired differences on the shared questions for --pair A-B.
Decision rules live in research/FIDELITY_PROTOCOL.md; this script is mechanical.
"""
import argparse
import collections
import json
from pathlib import Path

from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows
from .pcr_analysis import cluster_bootstrap

FAIL = ("malformed_action", "generation_limit", "context_limit")


def load_run(folder):
    folder = Path(folder)
    m, plan = load_json(folder / "run_manifest.json"), load_json(folder / "run_plan.json")
    if m.get("status") != "completed" or file_hash(folder / "sources.jsonl") != m["sources_sha256"]:
        raise ValueError("Need a completed, unmodified sources run: " + str(folder))
    rows = read_rows(folder / "sources.jsonl")
    return rows, {"checkpoint": plan["assets"]["agent_checkpoint"]["path"],
                  "prompt_variant": plan["protocol"]["agent"].get("prompt_variant", "bounded"),
                  "prompt_versions": sorted({r["prompt_version"] for r in rows}),
                  "sources_sha256": m["sources_sha256"]}


def mean_ci(rows, key, resamples):
    stat = lambda s: sum(key(r) for r in s) / len(s) if s else None
    return {"n": len(rows), "estimate": stat(rows), "ci95": cluster_bootstrap(rows, stat, resamples)}


def describe(rows, resamples):
    out = {"all": mean_ci(rows, lambda r: r["em"], resamples)}
    for d in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == d]
        out[d] = mean_ci(sub, lambda r: r["em"], resamples // 2)
    status = collections.Counter(r["status"] for r in rows)
    n = len(rows)
    out["behaviour"] = {"mean_search_calls": sum(r["search_calls"] for r in rows) / n,
                        "zero_search_rate": sum(r["search_calls"] == 0 for r in rows) / n,
                        "format_or_length_fail_rate": sum(status[s] for s in FAIL) / n,
                        "budget_exhausted_rate": status["search_budget_exhausted"] / n,
                        "answered_rate": status["answered"] / n,
                        "mean_generated_tokens": sum(r["generated_tokens"] for r in rows) / n}
    return out


def paired(a_rows, b_rows, resamples):
    b = {r["sample_id"]: r for r in b_rows}
    rows = [{"sample_id": r["sample_id"], "dataset": r["dataset"], "d": r["em"] - b[r["sample_id"]]["em"]}
            for r in a_rows if r["sample_id"] in b]
    e = mean_ci(rows, lambda r: r["d"], resamples)
    lo = e["ci95"][0] if e["ci95"] else None
    e.update(ge_3pp_ci_lo_gt0=bool(e["estimate"] is not None and e["estimate"] >= .03 and lo is not None and lo > 0),
             ge_5pp_ci_lo_gt0=bool(e["estimate"] is not None and e["estimate"] >= .05 and lo is not None and lo > 0))
    return e


def summarize(runs, pairs, resamples):
    data = {}
    for item in runs:
        label, _, folder = item.partition("=")
        if not label or not folder or label in data:
            raise ValueError("Use unique --run LABEL=DIR")
        rows, meta = load_run(folder)
        data[label] = (rows, meta)
    ids = [{r["sample_id"] for r in rows} for rows, _ in data.values()]
    report = {"analysis_commit": git_output("rev-parse", "HEAD"), "runs": {}, "pairs": {},
              "shared_questions": len(set.intersection(*ids)) if ids else 0,
              "bootstrap": {"unit": "question", "strata": "dataset", "resamples": resamples, "seed": 42}}
    for label, (rows, meta) in data.items():
        report["runs"][label] = dict(meta, **describe(rows, resamples))
    for p in pairs:
        a, _, b = p.partition("-")
        if a not in data or b not in data:
            raise ValueError("Unknown pair labels: " + p)
        report["pairs"][p] = paired(data[a][0], data[b][0], resamples)
    return report


def fmt(e):
    if not e or e.get("estimate") is None:
        return "NA"
    ci = e.get("ci95")
    return "{:.3f} [{}]".format(e["estimate"], "{:.3f}, {:.3f}".format(*ci) if ci else "no CI")


def markdown(r):
    L = ["# FC1 评测保真度结果（自动生成，供回传）", "", "共享题数：{}".format(r["shared_questions"]), "",
         "| run | prompt | EM all | hotpotqa | 2wiki | 平均检索 | 零检索 | 格式/长度失败 | 预算耗尽 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, v in r["runs"].items():
        b = v["behaviour"]
        L.append("| {} | {} | {} | {} | {} | {:.2f} | {:.3f} | {:.3f} | {:.3f} |".format(
            label, v["prompt_variant"], fmt(v["all"]), fmt(v.get("hotpotqa")), fmt(v.get("2wiki")),
            b["mean_search_calls"], b["zero_search_rate"], b["format_or_length_fail_rate"], b["budget_exhausted_rate"]))
    L += ["", "| pair (A − B) | ΔEM [95% CI] | ≥3pp 且下界>0 | ≥5pp 且下界>0 |", "| --- | --- | --- | --- |"]
    for p, e in r["pairs"].items():
        L.append("| {} | {} | {} | {} |".format(p, fmt(e), e["ge_3pp_ci_lo_gt0"], e["ge_5pp_ci_lo_gt0"]))
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", default=[], help="LABEL=DIR of a merged sources run")
    p.add_argument("--pair", action="append", default=[], help="LABEL_A-LABEL_B paired EM difference")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.output.exists():
            raise ValueError("Output exists")
        report = summarize(a.run, a.pair, a.resamples)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, report)
    a.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

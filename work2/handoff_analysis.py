"""HR1 analysis: handoff curves, 2x2 Shapley decomposition at end, evidence-free gap, gates.

Mechanical gate values per research/HANDOFF_PROTOCOL.md section 4; the GO decision
is recorded by a human in research/DECISION_LOG.md.
"""
import argparse
import json
from pathlib import Path

from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows
from .pcr_analysis import cluster_bootstrap

COMBOS = ("I>I", "R>R", "R>I", "I>R")
FORMAT_FAIL = ("malformed_action", "generation_limit")


def avg(v):
    return sum(v) / len(v)


def question_row(rec):
    """Per-question means: real/placebo/evonly at each handoff t and at end (= T of the source)."""
    row = {"sample_id": rec["sample_id"], "dataset": rec["dataset"]}
    for combo in COMBOS:
        src = combo.split(">")[0]
        T = rec["trajectories"][src]["search_calls"]
        vals = rec["combos"][combo]["values"]
        for t, xs in vals["real"].items():
            row["{}|real|{}".format(combo, t)] = avg(xs)
        row["{}|real|end".format(combo)] = avg(vals["real"][str(T)])
        for arm in ("placebo", "evonly"):
            for t, xs in vals.get(arm, {}).items():
                row["{}|{}|{}".format(combo, arm, t)] = avg(xs)
            # With no observation to replace (T = 0) the arm equals the real prefix.
            row["{}|{}|end".format(combo, arm)] = avg(vals[arm][str(T)]) if T >= 1 else row["{}|real|end".format(combo)]
        st = rec["combos"][combo]["statuses"]
        row[combo + "|format_fail"] = sum(st.get(s, 0) for s in FORMAT_FAIL) / max(1, sum(st.values()))
    row["traj_em_R"], row["traj_em_I"] = rec["trajectories"]["R"]["em"], rec["trajectories"]["I"]["em"]
    row["T_R"], row["T_I"] = rec["trajectories"]["R"]["search_calls"], rec["trajectories"]["I"]["search_calls"]
    return row


def f(key):
    return lambda r: r[key]


def est(rows, stat, resamples):
    have = [r for r in rows if stat(r) is not None] if callable(stat) else rows
    s = lambda sample: avg([stat(r) for r in sample]) if sample else None
    return {"n": len(have), "estimate": s(have) if have else None, "ci95": cluster_bootstrap(have, s, resamples) if have else None}


def components(r):
    g = lambda c, arm="real": r["{}|{}|end".format(c, arm)]
    acq = .5 * ((g("R>I") - g("I>I")) + (g("R>R") - g("I>R")))
    util = .5 * ((g("I>R") - g("I>I")) + (g("R>R") - g("R>I")))
    return {"gain_end": g("R>R") - g("I>I"), "acquisition": acq, "utilization": util,
            "evidence_free": g("R>R", "placebo") - g("I>I", "placebo"),
            "evidence_free_evonly": g("R>R", "evonly") - g("I>I", "evonly")}


def ratio(rows, num, den, resamples):
    stat = lambda sample: (sum(components(r)[num] for r in sample) / sum(components(r)[den] for r in sample)
                           if sample and abs(sum(components(r)[den] for r in sample)) > 1e-9 else None)
    return {"estimate": stat(rows), "ci95": cluster_bootstrap(rows, stat, resamples)}


def half_width(e):
    return (e["ci95"][1] - e["ci95"][0]) / 2 if e.get("ci95") else None


def analyze(run_dir, resamples=2000):
    m = load_json(run_dir / "run_manifest.json")
    if m.get("status") != "completed" or file_hash(run_dir / "records.jsonl") != m["records_sha256"]:
        raise ValueError("Need a completed, unmodified HR1 run")
    rows = [question_row(r) for r in read_rows(run_dir / "records.jsonl")]
    curves = {}
    for combo in COMBOS:
        curves[combo] = {}
        for t in ("0", "1", "2", "3", "end"):
            for arm in ("real", "placebo", "evonly"):
                key = "{}|{}|{}".format(combo, arm, t)
                have = [r for r in rows if key in r]
                if have:
                    curves[combo]["{}@{}".format(arm, t)] = est(have, f(key), resamples)
    comp = {k: est(rows, (lambda r, k=k: components(r)[k]), resamples)
            for k in ("gain_end", "acquisition", "utilization", "evidence_free", "evidence_free_evonly")}
    gain0 = est(rows, lambda r: r["R>R|real|0"] - r["I>I|real|0"], resamples)
    shares = {k: ratio(rows, k, "gain_end", resamples) for k in ("acquisition", "utilization", "evidence_free", "evidence_free_evonly")}
    mart = {p: est(rows, (lambda r, p=p: r["{0}>{0}|real|0".format(p)] - r["{0}>{0}|real|end".format(p)]), resamples) for p in ("R", "I")}
    with_ev = [r for r in rows if r["T_R"] >= 1]
    placebo_drop = est(with_ev, lambda r: r["R>R|real|end"] - r["R>R|placebo|end"], resamples)
    contains0 = lambda e: e["ci95"] is not None and e["ci95"][0] <= 0 <= e["ci95"][1]
    a1 = all(contains0(mart[p]) for p in mart) and placebo_drop["estimate"] is not None and placebo_drop["estimate"] >= .05 \
        and placebo_drop["ci95"] is not None and placebo_drop["ci95"][0] > 0
    hw = {k: half_width(comp[k]) for k in ("acquisition", "utilization", "evidence_free")}
    a2 = all(v is not None and v <= .03 for v in hw.values())
    sh = lambda k: shares[k]["estimate"]
    lo = lambda k: shares[k]["ci95"][0] if shares[k]["ci95"] else None
    a3 = bool((sh("evidence_free") is not None and sh("evidence_free") >= .3)
              or any(sh(k) is not None and sh(k) >= .6 and lo(k) is not None and lo(k) >= .5 for k in ("acquisition", "utilization")))
    gain_zero = contains0(comp["gain_end"])
    gates = {"G_A1_protocol_valid": a1, "G_A1_detail": {"martingale_0_minus_end": mart, "placebo_drop_R_end": placebo_drop},
             "G_A2_precision_halfwidth_le_3pp": a2, "G_A2_halfwidths": hw,
             "G_A3_informative": a3, "G_A3_iii": "not applicable (single RL checkpoint)",
             "gain_end_ci_contains_0": gain_zero,
             "GO": bool(a1 and a2 and a3 and not gain_zero),
             "note": "Mechanical values only; human decision in research/DECISION_LOG.md (HANDOFF_PROTOCOL section 4)."}
    return {"analysis_commit": git_output("rev-parse", "HEAD"), "records_sha256": m["records_sha256"], "questions": len(rows),
            "template_identical": m.get("template_identical"), "rollouts": m.get("rollouts"),
            "trajectory_em": {"R": avg([r["traj_em_R"] for r in rows]), "I": avg([r["traj_em_I"] for r in rows])},
            "mean_searches": {"R": avg([r["T_R"] for r in rows]), "I": avg([r["T_I"] for r in rows])},
            "format_fail_rate": {c: avg([r[c + "|format_fail"] for r in rows]) for c in COMBOS},
            "gain_0": gain0, "components": comp, "shares_of_gain_end": shares, "curves": curves, "gates": gates,
            "bootstrap": {"unit": "question", "strata": "dataset", "resamples": resamples, "seed": 42}}


def fmt(e):
    if not isinstance(e, dict) or e.get("estimate") is None:
        return "NA"
    ci = e.get("ci95")
    return "{:.4f} [{}]".format(e["estimate"], "{:.4f}, {:.4f}".format(*ci) if ci else "no CI")


def markdown(r):
    L = ["# HR1 交接重放 pilot 结果摘要（自动生成，供回传）", "",
         "questions={} rollouts={} template_identical={}".format(r["questions"], r["rollouts"], r["template_identical"]),
         "trajectory EM: R={:.3f} I={:.3f}; mean searches R={:.2f} I={:.2f}".format(r["trajectory_em"]["R"], r["trajectory_em"]["I"],
                                                                                 r["mean_searches"]["R"], r["mean_searches"]["I"]),
         "format fail rate: " + json.dumps({k: round(v, 4) for k, v in r["format_fail_rate"].items()}), "",
         "## 分解（end 处，EM）", "", "| 量 | 值 [95% CI] | 占 gain_end |", "| --- | --- | --- |",
         "| gain_0 (R>R − I>I @0) | {} | |".format(fmt(r["gain_0"]))]
    for k in ("gain_end", "acquisition", "utilization", "evidence_free", "evidence_free_evonly"):
        L.append("| {} | {} | {} |".format(k, fmt(r["components"][k]), fmt(r["shares_of_gain_end"].get(k)) if k in r["shares_of_gain_end"] else ""))
    L += ["", "## 交接曲线（real 臂）", "", "| combo | @0 | @1 | @2 | @3 | @end |", "| --- | --- | --- | --- | --- | --- |"]
    for combo, c in r["curves"].items():
        L.append("| {} | {} |".format(combo, " | ".join(fmt(c.get("real@" + t)) for t in ("0", "1", "2", "3", "end"))))
    L += ["", "## 门槛（机械值）", "", "```json", json.dumps(r["gates"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    out = a.output_dir or a.run_dir / "analysis"
    try:
        if out.exists():
            raise ValueError("Output exists; use a new --output-dir")
        report = analyze(a.run_dir, a.resamples)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    out.mkdir(parents=True)
    write_json(out / "hr1_report.json", report)
    (out / "hr1_report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

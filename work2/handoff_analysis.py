"""HR1/HR2 analysis: handoff curves, 2x2 Shapley decomposition at end, evidence-free gap, gates.

Mechanical gate values per research/HANDOFF_PROTOCOL.md section 4 (HR1) and
research/HR2_PROTOCOL.md section 4 (HR2); the GO decision is recorded by a human
in research/DECISION_LOG.md. `compare` pairs two HR2 runs that share questions
and the I policy but differ in R (PPO vs GRPO) for criterion G-A3(iii).
"""
import argparse
import json
from pathlib import Path

from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows
from .pcr_analysis import cluster_bootstrap

COMBOS = ("I>I", "R>R", "R>I", "I>R")
FORMAT_FAIL = ("malformed_action", "generation_limit")
NON_REAL = ("placebo", "evonly", "evonly_placebo")
HR2 = "HR2_handoff_replay_v03_v1"


def avg(v):
    return sum(v) / len(v)


def question_row(rec, arms=("placebo", "evonly")):
    """Per-question means: every arm at each handoff t and at end (= T of the source)."""
    row = {"sample_id": rec["sample_id"], "dataset": rec["dataset"]}
    for combo in COMBOS:
        src = combo.split(">")[0]
        T = rec["trajectories"][src]["search_calls"]
        vals = rec["combos"][combo]["values"]
        for t, xs in vals["real"].items():
            row["{}|real|{}".format(combo, t)] = avg(xs)
        row["{}|real|end".format(combo)] = avg(vals["real"][str(T)])
        for arm in arms:
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
    out = {"gain_end": g("R>R") - g("I>I"), "acquisition": acq, "utilization": util,
           "evidence_free": g("R>R", "placebo") - g("I>I", "placebo"),
           "evidence_free_evonly": g("R>R", "evonly") - g("I>I", "evonly")}
    if "R>R|evonly_placebo|end" in r:
        # HR2 primary evidence-free gap: queries kept, thinking dropped, every observation nulled.
        out["evidence_free_evonly_placebo"] = g("R>R", "evonly_placebo") - g("I>I", "evonly_placebo")
    return out


def share_stat(num, den="gain_end", key=lambda r: r):
    def stat(sample):
        d = sum(components(key(r))[den] for r in sample) if sample else 0
        return sum(components(key(r))[num] for r in sample) / d if sample and abs(d) > 1e-9 else None
    return stat


def ratio(rows, num, den, resamples):
    stat = share_stat(num, den)
    return {"estimate": stat(rows), "ci95": cluster_bootstrap(rows, stat, resamples)}


def half_width(e):
    return (e["ci95"][1] - e["ci95"][0]) / 2 if e.get("ci95") else None


def contains0(e):
    return e["ci95"] is not None and e["ci95"][0] <= 0 <= e["ci95"][1]


def completed_rows(run_dir):
    m = load_json(run_dir / "run_manifest.json")
    if m.get("status") != "completed" or file_hash(run_dir / "records.jsonl") != m["records_sha256"]:
        raise ValueError("Need a completed, unmodified HR run: " + str(run_dir))
    arms = tuple(load_json(run_dir / "run_plan.json")["protocol"].get("arm_set", ("real", "placebo", "evonly")))[1:]
    return m, [question_row(r, arms) for r in read_rows(run_dir / "records.jsonl")]


def analyze(run_dir, resamples=2000):
    m, rows = completed_rows(run_dir)
    version = load_json(run_dir / "run_plan.json")["protocol"]["protocol"]
    hr2 = version == HR2
    arms = ("real",) + NON_REAL
    curves = {}
    for combo in COMBOS:
        curves[combo] = {}
        for t in ("0", "1", "2", "3", "end"):
            for arm in arms:
                key = "{}|{}|{}".format(combo, arm, t)
                have = [r for r in rows if key in r]
                if have:
                    curves[combo]["{}@{}".format(arm, t)] = est(have, f(key), resamples)
    names = list(components(rows[0])) if rows else []
    comp = {k: est(rows, (lambda r, k=k: components(r)[k]), resamples) for k in names}
    gain0 = est(rows, lambda r: r["R>R|real|0"] - r["I>I|real|0"], resamples)
    shares = {k: ratio(rows, k, "gain_end", resamples) for k in names if k != "gain_end"}
    mart = {p: est(rows, (lambda r, p=p: r["{0}>{0}|real|0".format(p)] - r["{0}>{0}|real|end".format(p)]), resamples) for p in ("R", "I")}
    with_ev = [r for r in rows if r["T_R"] >= 1]
    placebo_drop = est(with_ev, lambda r: r["R>R|real|end"] - r["R>R|placebo|end"], resamples)
    ef = "evidence_free_evonly_placebo" if hr2 else "evidence_free"
    if hr2:
        # HR1 showed the placebo arm keeps evidence paraphrased in <think>; HR2 tests evidence use with thinking removed.
        drop = est(with_ev, lambda r: r["R>R|evonly|end"] - r["R>R|evonly_placebo|end"], resamples)
    else:
        drop = placebo_drop
    a1 = all(contains0(mart[p]) for p in mart) and drop["estimate"] is not None and drop["estimate"] >= .05 \
        and drop["ci95"] is not None and drop["ci95"][0] > 0
    hw = {k: half_width(comp[k]) for k in ("acquisition", "utilization", ef)}
    a2 = all(v is not None and v <= .03 for v in hw.values())
    sh = lambda k: shares[k]["estimate"]
    lo = lambda k: shares[k]["ci95"][0] if shares[k]["ci95"] else None
    a3 = bool((sh(ef) is not None and sh(ef) >= .3)
              or any(sh(k) is not None and sh(k) >= .6 and lo(k) is not None and lo(k) >= .5 for k in ("acquisition", "utilization")))
    gain_zero = contains0(comp["gain_end"])
    gates = {"protocol": version, "evidence_free_primary": ef,
             "G_A1_protocol_valid": a1,
             "G_A1_detail": {"martingale_0_minus_end": mart,
                             ("evonly_minus_evonly_placebo_R_end" if hr2 else "placebo_drop_R_end"): drop,
                             "placebo_drop_R_end_hr1_definition": placebo_drop},
             "G_A2_precision_halfwidth_le_3pp": a2, "G_A2_halfwidths": hw,
             "G_A3_informative_i_or_ii": a3,
             "G_A3_iii": ("see `compare` (PPO vs GRPO)" if hr2 else "not applicable (single RL checkpoint)"),
             "gain_end_ci_contains_0": gain_zero,
             "GO": bool(a1 and a2 and a3 and not gain_zero),
             "note": "Mechanical values only; human decision in research/DECISION_LOG.md."}
    return {"analysis_commit": git_output("rev-parse", "HEAD"), "protocol": version, "records_sha256": m["records_sha256"],
            "questions": len(rows), "template_identical": m.get("template_identical"), "rollouts": m.get("rollouts"),
            "trajectory_em": {"R": avg([r["traj_em_R"] for r in rows]), "I": avg([r["traj_em_I"] for r in rows])},
            "mean_searches": {"R": avg([r["T_R"] for r in rows]), "I": avg([r["T_I"] for r in rows])},
            "format_fail_rate": {c: avg([r[c + "|format_fail"] for r in rows]) for c in COMBOS},
            "gain_0": gain0, "components": comp, "shares_of_gain_end": shares, "curves": curves, "gates": gates,
            "bootstrap": {"unit": "question", "strata": "dataset", "resamples": resamples, "seed": 42}}


def compare(run_a, run_b, resamples=2000):
    """G-A3(iii): do two RL checkpoints gain by different routes? Paired over the same questions."""
    plans = [load_json(d / "run_plan.json") for d in (run_a, run_b)]
    if plans[0]["samples_sha256"] != plans[1]["samples_sha256"] or plans[0]["protocol_sha256"] != plans[1]["protocol_sha256"]:
        raise ValueError("Runs must share frozen questions and protocol")
    if plans[0]["protocol"]["protocol"] != HR2:
        raise ValueError("compare is defined for HR2 runs")
    ids = [p["assets"]["agent_checkpoint"] for p in plans]
    if ids[0] == ids[1]:
        raise ValueError("Both runs use the same R checkpoint")
    if plans[0]["assets"]["instruct_checkpoint"] != plans[1]["assets"]["instruct_checkpoint"]:
        raise ValueError("Runs must share the I checkpoint")
    rows = [completed_rows(d)[1] for d in (run_a, run_b)]
    b = {r["sample_id"]: r for r in rows[1]}
    if sorted(b) != sorted(r["sample_id"] for r in rows[0]):
        raise ValueError("Runs cover different questions")
    joint = [{"sample_id": r["sample_id"], "dataset": r["dataset"], "A": r, "B": b[r["sample_id"]]} for r in rows[0]]
    out = {"run_a": str(run_a), "run_b": str(run_b), "questions": len(joint), "share_difference_a_minus_b": {},
           "component_difference_a_minus_b": {}}
    for k in components(joint[0]["A"]):
        out["component_difference_a_minus_b"][k] = est(joint, (lambda r, k=k: components(r["A"])[k] - components(r["B"])[k]), resamples)
    for k in ("acquisition", "utilization", "evidence_free_evonly_placebo"):
        sa, sb = share_stat(k, key=lambda r: r["A"]), share_stat(k, key=lambda r: r["B"])
        stat = lambda s, sa=sa, sb=sb: (None if not s or sa(s) is None or sb(s) is None else sa(s) - sb(s))
        out["share_difference_a_minus_b"][k] = {"estimate": stat(joint), "ci95": cluster_bootstrap(joint, stat, resamples),
                                                "a": sa(joint), "b": sb(joint)}
    d = out["share_difference_a_minus_b"]
    out["G_A3_iii"] = any(d[k]["estimate"] is not None and abs(d[k]["estimate"]) >= .2 and d[k]["ci95"] is not None
                          and not d[k]["ci95"][0] <= 0 <= d[k]["ci95"][1] for k in ("acquisition", "utilization"))
    out["note"] = "G-A3(iii): |share_A - share_B| >= 20 pp with paired 95% CI excluding 0 for acquisition or utilization."
    return out


def fmt(e):
    if not isinstance(e, dict) or e.get("estimate") is None:
        return "NA"
    ci = e.get("ci95")
    return "{:.4f} [{}]".format(e["estimate"], "{:.4f}, {:.4f}".format(*ci) if ci else "no CI")


def markdown(r):
    name = "HR2" if r.get("protocol") == HR2 else "HR1"
    L = ["# {} 交接重放结果摘要（自动生成，供回传）".format(name), "",
         "questions={} rollouts={} template_identical={}".format(r["questions"], r["rollouts"], r["template_identical"]),
         "trajectory EM: R={:.3f} I={:.3f}; mean searches R={:.2f} I={:.2f}".format(r["trajectory_em"]["R"], r["trajectory_em"]["I"],
                                                                                 r["mean_searches"]["R"], r["mean_searches"]["I"]),
         "format fail rate: " + json.dumps({k: round(v, 4) for k, v in r["format_fail_rate"].items()}), "",
         "## 分解（end 处，EM）", "", "| 量 | 值 [95% CI] | 占 gain_end |", "| --- | --- | --- |",
         "| gain_0 (R>R − I>I @0) | {} | |".format(fmt(r["gain_0"]))]
    for k in r["components"]:
        L.append("| {} | {} | {} |".format(k, fmt(r["components"][k]), fmt(r["shares_of_gain_end"].get(k)) if k in r["shares_of_gain_end"] else ""))
    arms = {key.split("@")[0] for c in r["curves"].values() for key in c}
    for arm in [a for a in ("real",) + NON_REAL if a in arms]:
        L += ["", "## 交接曲线（{} 臂）".format(arm), "", "| combo | @0 | @1 | @2 | @3 | @end |", "| --- | --- | --- | --- | --- | --- |"]
        for combo, c in r["curves"].items():
            L.append("| {} | {} |".format(combo, " | ".join(fmt(c.get("{}@{}".format(arm, t))) for t in ("0", "1", "2", "3", "end"))))
    L += ["", "## 门槛（机械值）", "", "```json", json.dumps(r["gates"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(L)


def compare_markdown(c):
    L = ["# HR2 检查点比较（G-A3(iii)，自动生成）", "", "A = {}".format(c["run_a"]), "B = {}".format(c["run_b"]),
         "questions = {}".format(c["questions"]), "", "| 份额 | A | B | A − B [95% CI] |", "| --- | --- | --- | --- |"]
    for k, e in c["share_difference_a_minus_b"].items():
        g = lambda v: "NA" if v is None else "{:.3f}".format(v)
        L.append("| {} | {} | {} | {} |".format(k, g(e["a"]), g(e["b"]), fmt(e)))
    L += ["", "| 分量差 A − B | 值 [95% CI] |", "| --- | --- |"]
    L += ["| {} | {} |".format(k, fmt(e)) for k, e in c["component_difference_a_minus_b"].items()]
    L += ["", "G_A3_iii = {}".format(c["G_A3_iii"]), c["note"], ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--compare-with", type=Path, help="second HR2 run (other R checkpoint); writes the G-A3(iii) comparison")
    a = p.parse_args()
    try:
        if a.compare_with:
            out = a.output_dir or a.run_dir / "compare"
            if out.exists():
                raise ValueError("Output exists; use a new --output-dir")
            report = compare(a.run_dir, a.compare_with, a.resamples)
            out.mkdir(parents=True)
            write_json(out / "hr2_compare.json", report)
            (out / "hr2_compare.md").write_text(compare_markdown(report), encoding="utf-8")
            print(compare_markdown(report))
            return 0
        out = a.output_dir or a.run_dir / "analysis"
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

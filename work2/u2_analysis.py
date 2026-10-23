"""U2 analysis: repair / harm of a published recovery mechanism against same-state placebos.

Mechanical values per research/U2_PROTOCOL.md section 4; verdicts are recorded by a
human in research/DECISION_LOG.md. Bootstrap: whole questions, stratified by dataset.
"""
import argparse
import collections
import json
from pathlib import Path

from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows
from .pcr_analysis import cluster_bootstrap
from .u2 import ARMS, MRSEARCH, RESEEK

FORMAT_FAIL = ("malformed_action", "generation_limit")
PAPER_MEAN_EM = {RESEEK: 0.312}


def avg(v):
    v = list(v)
    return sum(v) / len(v) if v else None


def est(rows, stat, resamples):
    s = lambda sample: avg(stat(r) for r in sample) if sample else None
    return {"n_units": len(rows), "n_questions": len({r["sample_id"] for r in rows}), "estimate": s(rows),
            "ci95": cluster_bootstrap(rows, s, resamples) if rows else None}


def ratio(rows, num, den, resamples):
    def stat(sample):
        d = sum(den(r) for r in sample) if sample else 0
        return sum(num(r) for r in sample) / d if sample and d > 1e-9 else None
    return {"estimate": stat(rows), "ci95": cluster_bootstrap(rows, stat, resamples) if rows else None}


def col(key):
    return lambda r: r[key]


def diff(a, b):
    return lambda r: r[a] - r[b]


def by_dataset(rows, key):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["dataset"]].append(r[key])
    return {d: {"n": len(v), "mean": avg(v)} for d, v in groups.items()}


def h1_verdict(explained, net):
    """AC1 rule: placebo explains >= 1/3 of the method's repair, or the placebo-corrected net is <= 0."""
    if explained["estimate"] is None or net["estimate"] is None:
        return None
    return bool(explained["estimate"] >= 1 / 3 or net["estimate"] <= 0)


def reseek(records, resamples):
    rows, turns = [], []
    for rec in records:
        for st in rec["states"]:
            row = {"sample_id": rec["sample_id"], "dataset": rec["dataset"], "t": st["t"], "J": st["source_judge"],
                   "src_em": rec["source"]["em"], "answer_hit": st["answer_hit"]}
            for arm in ARMS[RESEEK]:
                a = st["arms"][arm]
                row["V_" + arm] = avg(a["em"])
                row["extra_" + arm] = avg(a["extra_searches"])
            row["stageA_present"] = avg(j != "absent" for j in st["stage_a_judge"])
            row["stageA_no"] = avg(j == "No" for j in st["stage_a_judge"])
            row["perm_no"] = avg(j == "No" for j in st["arms"]["perm"]["first_judge"])
            rows.append(row)
            turns.append(row)
    sources = [rec["source"] for rec in records]
    observed = [j for rec in records for j in rec["source_judges"]]
    fidelity = {"questions": len(records), "source_em": avg(s["em"] for s in sources),
                "source_em_by_dataset": by_dataset([{"dataset": r["dataset"], "em": r["source"]["em"]} for r in records], "em"),
                "paper_mean_em": PAPER_MEAN_EM[RESEEK],
                "source_format_fail": avg(s["status"] in FORMAT_FAIL for s in sources),
                "source_mean_searches": avg(s["search_calls"] for s in sources),
                "source_judge_rate": avg(j is not None for j in observed) if observed else None,
                "source_judge_no_rate": avg(j == "No" for j in observed) if observed else None,
                "stage_a_judge_rate": avg(r["stageA_present"] for r in rows) if rows else None,
                "judge_no_given_answer_in_obs": avg(r["J"] == "No" for r in turns if r["answer_hit"]),
                "judge_no_given_no_answer_in_obs": avg(r["J"] == "No" for r in turns if not r["answer_hit"])}
    fidelity["passes"] = bool(fidelity["source_judge_rate"] is not None and fidelity["source_judge_rate"] >= .8
                              and fidelity["source_format_fail"] <= .1)
    fidelity["em_gap_flag"] = abs(fidelity["source_em"] - PAPER_MEAN_EM[RESEEK]) > .05
    sets = {"S_fail": [r for r in rows if r["src_em"] == 0 and r["J"] == "Yes"],
            "S_ok": [r for r in rows if r["src_em"] == 1 and r["J"] == "Yes"],
            "S_natno": [r for r in rows if r["J"] == "No"],
            "S_natno_wrong": [r for r in rows if r["J"] == "No" and r["src_em"] == 0],
            "all": rows}
    out = {"fidelity": fidelity, "sets": {}}
    for name, rs in sets.items():
        block = {"n_states": len(rs), "n_questions": len({r["sample_id"] for r in rs}),
                 "value": {arm: est(rs, col("V_" + arm), resamples) for arm in ARMS[RESEEK]},
                 "extra_searches": {arm: avg(r["extra_" + arm] for r in rs) for arm in ARMS[RESEEK]}}
        if rs:
            block["force_no_minus_nat"] = est(rs, diff("V_force_no", "V_nat"), resamples)
            block["force_no_minus_perm"] = est(rs, diff("V_force_no", "V_perm"), resamples)
            block["nat_minus_force_yes"] = est(rs, diff("V_nat", "V_force_yes"), resamples)
            block["explained_by_nat"] = ratio(rs, col("V_nat"), col("V_force_no"), resamples)
            block["explained_by_perm"] = ratio(rs, col("V_perm"), col("V_force_no"), resamples)
        out["sets"][name] = block
    f = out["sets"]["S_fail"]
    out["gates"] = {
        "F_fidelity": fidelity["passes"],
        "H1_reseek_resample": h1_verdict(f["explained_by_nat"], f["force_no_minus_nat"]) if sets["S_fail"] else None,
        "H1_reseek_perm": h1_verdict(f["explained_by_perm"], f["force_no_minus_perm"]) if sets["S_fail"] else None,
        "natural_trigger_effect_nat_minus_force_yes": out["sets"]["S_natno"].get("nat_minus_force_yes"),
        "harm_of_forced_no_on_S_ok": out["sets"]["S_ok"].get("force_no_minus_nat"),
        "note": "H1 verdicts count only if F_fidelity passes. Mechanical values; human decision in research/DECISION_LOG.md."}
    return out


def mrsearch(records, resamples):
    rows = []
    for rec in records:
        ep1 = rec["episode1"]
        row = {"sample_id": rec["sample_id"], "dataset": rec["dataset"], "ep1_answered": ep1["status"] == "answered",
               "ep1_em": ep1["em"] if ep1["status"] == "answered" else 0}
        for arm in ARMS[MRSEARCH]:
            a = rec["arms"][arm]
            row["V_" + arm] = avg(a["final_em"])
            row["net_" + arm] = row["V_" + arm] - row["ep1_em"]
            row["ep2_answered_" + arm] = avg(a["ep2_answered"])
            row["changed_" + arm] = avg(a["changed"])
            row["ep2_searches_" + arm] = avg(a["ep2_searches"])
        rows.append(row)
    fidelity = {"questions": len(rows), "ep1_em": avg(r["ep1_em"] for r in rows),
                "ep1_em_by_dataset": by_dataset(rows, "ep1_em"),
                "ep1_format_fail": avg(rec["episode1"]["status"] in FORMAT_FAIL for rec in records),
                "ep1_mean_searches": avg(rec["episode1"]["search_calls"] for rec in records),
                "ep2_answered_rate": {arm: avg(r["ep2_answered_" + arm] for r in rows) for arm in ARMS[MRSEARCH]}}
    fidelity["passes"] = bool(fidelity["ep1_format_fail"] <= .1 and fidelity["ep2_answered_rate"]["reflect"] >= .8)
    sets = {"S_wrong": [r for r in rows if r["ep1_answered"] and r["ep1_em"] == 0],
            "S_right": [r for r in rows if r["ep1_answered"] and r["ep1_em"] == 1],
            "S_unanswered": [r for r in rows if not r["ep1_answered"]], "all": rows}
    out = {"fidelity": fidelity, "sets": {}}
    for name, rs in sets.items():
        block = {"n_questions": len(rs), "value": {arm: est(rs, col("V_" + arm), resamples) for arm in ARMS[MRSEARCH]},
                 "net_vs_episode1": {arm: est(rs, col("net_" + arm), resamples) for arm in ARMS[MRSEARCH]},
                 "changed_answer_rate": {arm: avg(r["changed_" + arm] for r in rs) for arm in ARMS[MRSEARCH]},
                 "ep2_searches": {arm: avg(r["ep2_searches_" + arm] for r in rs) for arm in ARMS[MRSEARCH]}}
        if rs:
            for other in ("neutral", "reask", "fresh"):
                block["reflect_minus_" + other] = est(rs, diff("V_reflect", "V_" + other), resamples)
                block["explained_by_" + other] = ratio(rs, col("V_" + other), col("V_reflect"), resamples)
        out["sets"][name] = block
    w = out["sets"]["S_wrong"]
    out["gates"] = {
        "F_fidelity": fidelity["passes"],
        "H1_mr_neutral": h1_verdict(w["explained_by_neutral"], w["reflect_minus_neutral"]) if sets["S_wrong"] else None,
        "H1_mr_fresh": h1_verdict(w["explained_by_fresh"], w["reflect_minus_fresh"]) if sets["S_wrong"] else None,
        "H1_mr_reask": h1_verdict(w["explained_by_reask"], w["reflect_minus_reask"]) if sets["S_wrong"] else None,
        "harm_on_S_right": {arm: out["sets"]["S_right"]["net_vs_episode1"][arm] for arm in ARMS[MRSEARCH]},
        "net_all_questions": {arm: out["sets"]["all"]["net_vs_episode1"][arm] for arm in ARMS[MRSEARCH]},
        "note": "H1 verdicts count only if F_fidelity passes; primary placebo = neutral. Mechanical values; human decision in research/DECISION_LOG.md."}
    return out


def analyze(run_dir, resamples=2000):
    m = load_json(run_dir / "run_manifest.json")
    if m.get("status") != "completed" or file_hash(run_dir / "records.jsonl") != m["records_sha256"]:
        raise ValueError("Need a completed, unmodified U2 run")
    plan = load_json(run_dir / "run_plan.json")
    version = plan["protocol"]["protocol"]
    records = read_rows(run_dir / "records.jsonl")
    body = reseek(records, resamples) if version == RESEEK else mrsearch(records, resamples)
    return dict(body, protocol=version, agent_label=plan.get("agent_label"), smoke=plan.get("smoke"),
                analysis_commit=git_output("rev-parse", "HEAD"), records_sha256=m["records_sha256"],
                branch_rollouts=m.get("branch_rollouts"), retrieval=m.get("retrieval"),
                bootstrap={"unit": "question", "strata": "dataset", "resamples": resamples, "seed": 42})


def fmt(e):
    if not isinstance(e, dict) or e.get("estimate") is None:
        return "NA"
    ci = e.get("ci95")
    return "{:.3f} [{}]".format(e["estimate"], "{:.3f}, {:.3f}".format(*ci) if ci else "no CI")


def markdown(r):
    arms = ARMS[r["protocol"]]
    L = ["# U2 审计结果摘要（{}，智能体 {}；自动生成，供回传）".format(r["protocol"], r["agent_label"]), "",
         "branch_rollouts={} smoke={}".format(r["branch_rollouts"], r["smoke"]), "", "## 保真度", "", "```json",
         json.dumps(r["fidelity"], ensure_ascii=False, indent=2), "```", "", "## 各状态集合（各臂的平均 EM）", "",
         "| 集合 | n | " + " | ".join(arms) + " |", "| --- | --- |" + " --- |" * len(arms)]
    for name, b in r["sets"].items():
        n = b.get("n_states", b.get("n_questions"))
        L.append("| {} | {} | {} |".format(name, n, " | ".join(fmt(b["value"][a]) for a in arms)))
    L += ["", "## 差值与 explained", "", "| 集合 | 量 | 值 [95% CI] |", "| --- | --- | --- |"]
    for name, b in r["sets"].items():
        for k, v in b.items():
            if k.startswith(("force_no_minus", "nat_minus", "reflect_minus", "explained_by")):
                L.append("| {} | {} | {} |".format(name, k, fmt(v)))
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
    write_json(out / "u2_report.json", report)
    (out / "u2_report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CT1 analysis: interventional truth (A_t, E_t), reliability, signal agreement, gates.

Gates follow research/CREDIT_PROTOCOL.md section 6; values are mechanical, the GO
decision is recorded by a human in research/DECISION_LOG.md.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows
from .pcr_analysis import cluster_bootstrap

DEADZONE = 0.05
USEFUL_EVIDENCE = 0.25
SIGNALS = ("ig", "ig_deadzone", "trace", "ealr", "fact_hit", "answer_hit", "random")
GATE_SIGNALS = ("ig", "ealr", "fact_hit")


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else None


def spearman(x, y):
    return pearson(ranks(x), ranks(y)) if len(x) >= 3 else None


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    r = ranks(pos + neg)
    return (sum(r[:len(pos)]) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def turn_rows(record, policy=None, reference=None):
    """One row per search turn t = 1..T with truth, halves and signals."""
    t = record["trajectory"]
    V, N = record["values"]["V"], record["values"]["N"]
    if t["search_calls"] == 0 or not record["null_eligible"]:
        return []
    cb = mean(record["values"]["C"])
    half = lambda xs, h: mean(xs[h::2])
    rows, seen_facts, seen_answer = [], set(), False
    for turn in range(1, t["search_calls"] + 1):
        info = t["turns"][turn - 1]
        new_facts = set(info["fact_ids"]) - seen_facts
        seen_facts |= set(info["fact_ids"])
        answer_new = info["answer_hit"] and not seen_answer
        seen_answer = seen_answer or info["answer_hit"]
        row = {"sample_id": record["sample_id"], "dataset": record["dataset"], "turn": turn,
               "A": mean(V[turn]) - mean(V[turn - 1]), "E": mean(V[turn]) - mean(N[turn - 1]),
               "A_h": [half(V[turn], h) - half(V[turn - 1], h) for h in (0, 1)],
               "E_h": [half(V[turn], h) - half(N[turn - 1], h) for h in (0, 1)],
               "cb": cb, "memory_known": cb >= .5, "traj_success": t["em"] == 1,
               "fact_hit": float(len(new_facts)), "answer_hit": float(answer_new),
               "random": int(hashlib.sha256("{}|{}".format(record["sample_id"], turn).encode()).hexdigest()[:8], 16) / 16 ** 8}
        if policy is not None:
            g = policy["gold_mean_logprob"]
            row["ig"] = g[turn] - g[turn - 1]
            row["ig_deadzone"] = row["ig"] if abs(row["ig"]) >= DEADZONE else 0.0
            row["ealr"] = policy["ealr"].get(str(turn))
        if reference is not None:
            r = reference["gold_mean_logprob"]
            row["trace"] = r[turn] - r[turn - 1]
        rows.append(row)
    return rows


def reliability(rows, key, resamples):
    def stat(sample):
        r = pearson([x[key + "_h"][0] for x in sample], [x[key + "_h"][1] for x in sample])
        return 2 * r / (1 + r) if r is not None and r > -1 else None
    return {"spearman_brown": stat(rows), "ci95": cluster_bootstrap(rows, stat, resamples)}


def agreement(rows, signal, truth, resamples):
    have = [r for r in rows if r.get(signal) is not None]
    stat = lambda s: spearman([r[signal] for r in s], [r[truth] for r in s])
    return {"n_turns": len(have), "spearman": stat(have) if have else None,
            "ci95": cluster_bootstrap(have, stat, resamples) if len(have) >= 10 else None}


def strat_difference(rows, signal, resamples):
    """rho(signal,E | memory-known) - rho(signal,E | unknown), question-clustered."""
    have = [r for r in rows if r.get(signal) is not None]
    def stat(s):
        a = [r for r in s if r["memory_known"]]; b = [r for r in s if not r["memory_known"]]
        ra = spearman([r[signal] for r in a], [r["E"] for r in a]); rb = spearman([r[signal] for r in b], [r["E"] for r in b])
        return ra - rb if ra is not None and rb is not None else None
    return {"estimate": stat(have) if have else None, "ci95": cluster_bootstrap(have, stat, resamples) if have else None}


def signal_block(rows, resamples):
    out = {}
    for s in SIGNALS:
        have = [r for r in rows if r.get(s) is not None]
        if not have:
            out[s] = {"available": False}
            continue
        out[s] = {"available": True, "vs_A": agreement(rows, s, "A", resamples), "vs_E": agreement(rows, s, "E", resamples),
                  "auroc_useful_evidence": auroc([r[s] for r in have], [r["E"] >= USEFUL_EVIDENCE for r in have])}
    return out


def analyze(run_dir, resamples=2000):
    m = load_json(run_dir / "run_manifest.json")
    if m.get("status") != "completed" or file_hash(run_dir / "records.jsonl") != m["records_sha256"]:
        raise ValueError("Need a completed, unmodified CT1 run")
    records = read_rows(run_dir / "records.jsonl")
    sig = {}
    for role in ("policy", "reference"):
        path = run_dir / "signals_{}.jsonl".format(role)
        if path.exists():
            if m.get("signals", {}).get(role, {}).get("sha256") != file_hash(path):
                raise ValueError("Signal file changed after merge: " + role)
            sig[role] = {r["sample_id"]: r for r in read_rows(path)}
    rows = [row for rec in records for row in turn_rows(rec, sig.get("policy", {}).get(rec["sample_id"]),
                                                         sig.get("reference", {}).get(rec["sample_id"]))]
    if not rows:
        raise ValueError("No eligible search turns")
    rel = {"A": reliability(rows, "A", resamples), "E": reliability(rows, "E", resamples)}
    strata = {}
    for name, keep in (("memory_known", lambda r: r["memory_known"]), ("memory_unknown", lambda r: not r["memory_known"]),
                       ("turn_1", lambda r: r["turn"] == 1), ("turn_ge2", lambda r: r["turn"] >= 2),
                       ("traj_success", lambda r: r["traj_success"]), ("traj_failure", lambda r: not r["traj_success"])):
        sub = [r for r in rows if keep(r)]
        strata[name] = {"turns": len(sub), "mean_A": mean([r["A"] for r in sub]), "mean_E": mean([r["E"] for r in sub]),
                        "signals": signal_block(sub, resamples // 4) if len(sub) >= 10 else None}
    signals = signal_block(rows, resamples)
    diffs = {s: strat_difference(rows, s, resamples) for s in ("ig", "ealr") if signals[s].get("available")}
    pair = None
    both = [r for r in rows if r.get("ig") is not None and r.get("ealr") is not None]
    if len(both) >= 10:
        st = lambda s: spearman([r["ig"] for r in s], [r["ealr"] for r in s])
        pair = {"spearman": st(both), "ci95": cluster_bootstrap(both, st, resamples)}
    ga = all(rel[k]["spearman_brown"] is not None and rel[k]["spearman_brown"] >= .6 for k in ("A", "E"))
    low = [s for s in GATE_SIGNALS if signals[s].get("available") and signals[s]["vs_E"]["spearman"] is not None
           and signals[s]["vs_E"]["spearman"] < .3 and signals[s]["vs_E"]["ci95"] and signals[s]["vs_E"]["ci95"][1] < .5]
    strat = [s for s, d in diffs.items() if d["ci95"] and (d["ci95"][0] > 0 or d["ci95"][1] < 0)]
    disagree = bool(pair and pair["spearman"] is not None and pair["spearman"] < .3)
    mc_hours = m.get("mc_wall_seconds", 0) / 3600
    planned = len(records)
    gates = {"G_a_truth_reliability_ge_0.6": ga,
             "G_b_reportable_difference": bool(low or strat or disagree),
             "G_b_detail": {"low_agreement_signals": low, "memory_stratum_difference": strat, "ig_ealr_disagree": disagree},
             "G_c_cost_le_8h_for_600": (mc_hours * 600 / planned) <= 8 if planned else None,
             "mc_hours": mc_hours, "missing_signals": [s for s in GATE_SIGNALS if not signals[s].get("available")],
             "note": "Mechanical values only; record the human decision in research/DECISION_LOG.md (CREDIT_PROTOCOL section 6)."}
    return {"analysis_commit": git_output("rev-parse", "HEAD"), "records_sha256": m["records_sha256"],
            "questions": len(records), "eligible_questions": len({r["sample_id"] for r in rows}), "turns": len(rows),
            "trajectory_em": mean([r["trajectory"]["em"] for r in records]),
            "trajectory_reproduced": sum(r["trajectory"]["reproduced"] for r in records),
            "null_eligible": sum(r["null_eligible"] for r in records),
            "mean_closed_book_value": mean([mean(r["values"]["C"]) for r in records if r["null_eligible"]]),
            "memory_known_questions": len({r["sample_id"] for r in rows if r["memory_known"]}),
            "truth": {"mean_A": mean([r["A"] for r in rows]), "mean_E": mean([r["E"] for r in rows]),
                      "share_E_ge_useful": mean([float(r["E"] >= USEFUL_EVIDENCE) for r in rows]),
                      "spearman_A_E": spearman([r["A"] for r in rows], [r["E"] for r in rows])},
            "reliability": rel, "signals": signals, "memory_stratum_difference": diffs, "ig_vs_ealr": pair,
            "strata": strata, "gates": gates,
            "bootstrap": {"unit": "question", "strata": "dataset", "resamples": resamples, "seed": 42}}, rows


def fmt(e):
    if not isinstance(e, dict) or e.get("spearman") is None:
        return "NA"
    ci = e.get("ci95")
    return "{:.3f} [{}] (n={})".format(e["spearman"], "{:.3f}, {:.3f}".format(*ci) if ci else "no CI", e["n_turns"])


def markdown(r):
    L = ["# CT1 pilot 结果摘要（自动生成，供回传）", "",
         "questions={} eligible={} turns={} trajectory EM={:.3f} reproduced={} null_eligible={} memory_known_q={}".format(
             r["questions"], r["eligible_questions"], r["turns"], r["trajectory_em"], r["trajectory_reproduced"],
             r["null_eligible"], r["memory_known_questions"]),
         "mean closed-book V={}；truth mean A={:.3f} mean E={:.3f} share(E>=0.25)={:.3f} rho(A,E)={}".format(
             r["mean_closed_book_value"], r["truth"]["mean_A"], r["truth"]["mean_E"], r["truth"]["share_E_ge_useful"], r["truth"]["spearman_A_E"]), "",
         "## 真值信度（Spearman-Brown）", ""]
    for k in ("A", "E"):
        rel = r["reliability"][k]
        L.append("- {}: {} CI {}".format(k, rel["spearman_brown"], rel["ci95"]))
    L += ["", "## 信号一致性（全体轮次）", "", "| signal | rho vs A | rho vs E | AUROC(E>=0.25) |", "| --- | --- | --- | --- |"]
    for s, v in r["signals"].items():
        if v.get("available"):
            L.append("| {} | {} | {} | {} |".format(s, fmt(v["vs_A"]), fmt(v["vs_E"]),
                                                 "NA" if v["auroc_useful_evidence"] is None else "{:.3f}".format(v["auroc_useful_evidence"])))
        else:
            L.append("| {} | missing | missing | missing |".format(s))
    L += ["", "## 分层 rho(signal, E)", "", "| stratum | turns | mean E | ig | ealr | fact_hit |", "| --- | --- | --- | --- | --- | --- |"]
    for name, st in r["strata"].items():
        g = lambda s: fmt(st["signals"][s]["vs_E"]) if st["signals"] and st["signals"][s].get("available") else "NA"
        L.append("| {} | {} | {} | {} | {} | {} |".format(name, st["turns"], None if st["mean_E"] is None else round(st["mean_E"], 3), g("ig"), g("ealr"), g("fact_hit")))
    L += ["", "记忆层差异（known − unknown）：{}".format(json.dumps(r["memory_stratum_difference"], ensure_ascii=False)),
          "ig vs ealr：{}".format(json.dumps(r["ig_vs_ealr"], ensure_ascii=False)), "",
          "## 预登记门槛（机械值）", "", "```json", json.dumps(r["gates"], ensure_ascii=False, indent=2), "```", ""]
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
        report, rows = analyze(a.run_dir, a.resamples)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    out.mkdir(parents=True)
    write_json(out / "ct1_report.json", report)
    write_json(out / "ct1_turns.json", rows)
    (out / "ct1_report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

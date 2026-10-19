"""CPU analysis of PCR v1 states: path luck, placebo-corrected recovery and deletion.

Reads only the merged states.jsonl of a completed PCR run. Pre-registered gates
G1/G2/G3 (research/PCR_PROTOCOL.md) are computed mechanically; the GO decision
itself is recorded by a human after reading the report, never automatically.
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows
from .metrics import answer_scores, normalize
from .placebo import ARM_GROUPS, PERMUTATION_ARMS, PLACEBO_ARMS, VERSION

FORMAT_F1 = 0.8


def contained(a, b):
    a, b = normalize(a), normalize(b)
    return bool(a) and bool(b) and ((" " + a + " ") in (" " + b + " ") or (" " + b + " ") in (" " + a + " "))


def same_answer(a, b):
    """Two branch answers that differ only by surface form (alias, article, extra words)."""
    if normalize(a) == normalize(b):
        return True
    return contained(a, b) or answer_scores(a, b)["f1"] >= FORMAT_F1


def flip_kind(clean_answer, clean_em, other_answer, other_em):
    if clean_em == other_em:
        return "none"
    return "format" if same_answer(clean_answer, other_answer) else "semantic"


def answer_type(gold):
    g = normalize(gold)
    if g in ("yes", "no"):
        return "yes_no"
    if re.fullmatch(r"[\d\s,.]+", g):
        return "number"
    if re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", g) or re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", g):
        return "date"
    return "entity_or_other"


def cluster_bootstrap(rows, stat, resamples=2000, seed=42):
    """Whole-question resampling stratified by dataset; returns [lo, hi] or None."""
    strata = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        strata[r["dataset"]][r["sample_id"]].append(r)
    groups = [list(q.values()) for q in strata.values()]
    if resamples == 0 or not groups or any(len(g) < 2 for g in groups):
        return None
    rng, draws = random.Random(seed), []
    for _ in range(resamples):
        sample = [r for g in groups for cluster in rng.choices(g, k=len(g)) for r in cluster]
        value = stat(sample)
        if value is not None:
            draws.append(value)
    if len(draws) < resamples * 0.9:
        return None
    draws.sort()
    return [draws[int((len(draws) - 1) * .025)], draws[int((len(draws) - 1) * .975)]]


def estimate(rows, stat, resamples, seed=42):
    return {"n_states": len(rows), "n_questions": len({(r["dataset"], r["sample_id"]) for r in rows}),
            "estimate": stat(rows) if rows else None,
            "ci95": cluster_bootstrap(rows, stat, resamples, seed) if rows else None}


def ratio(num, den):
    def stat(rows):
        d = sum(den(r) for r in rows)
        return sum(num(r) for r in rows) / d if d else None
    return stat


def mean(f):
    return lambda rows: sum(f(r) for r in rows) / len(rows) if rows else None


def derive(state):
    """Per-state outcome vector with raw-EM and format-corrected ('semantic') views."""
    arms = state["arms"]
    clean = arms["clean_a"]
    y0 = clean["em"]
    raw, sem, kinds = {}, {}, {}
    for name, row in arms.items():
        if row is None:
            continue
        kind = flip_kind(clean["final_answer"], y0, row["final_answer"], row["em"])
        kinds[name] = kind
        raw[name] = row["em"]
        sem[name] = y0 if kind == "format" else row["em"]
    placebo = [a for a in PLACEBO_ARMS if a in raw]
    p_raw = sum(raw[a] for a in placebo) / len(placebo)
    p_sem = sum(sem[a] for a in placebo) / len(placebo)
    facts = set(state["known_fact_ids"])
    return {"sample_id": state["sample_id"], "dataset": state["dataset"], "turn": state["target_turn"],
            "y0": y0, "raw": raw, "sem": sem, "kinds": kinds, "placebo": placebo,
            "p_raw": p_raw, "p_sem": p_sem, "answer_type": answer_type(state["gold_answer"]),
            "source_search_calls": state["source_search_calls"],
            "evidence_retrieved_proxy": bool(facts) and facts <= set(state["candidate_fact_ids"]),
            "target_has_support": bool(state["target_visible_fact_ids"]),
            "deletion": state["deletion_eligibility"]}


def luck_block(rows, view, resamples):
    p = "p_" + view
    lucky = lambda r: int(r["y0"] == 1 and r[p] < .5)
    unlucky = lambda r: int(r["y0"] == 0 and r[p] >= .5)
    y = lambda r: r[view]
    return {"clean_em": estimate(rows, mean(lambda r: r["y0"]), resamples),
            "mEM": estimate(rows, mean(lambda r: r[p]), resamples),
            "rEM_all_placebos": estimate(rows, mean(lambda r: int(all(y(r)[a] == 1 for a in r["placebo"]))), resamples),
            "lucky_successes": sum(lucky(r) for r in rows),
            "unlucky_failures": sum(unlucky(r) for r in rows),
            "clean_correct": sum(r["y0"] for r in rows),
            "lucky_share_of_accuracy": estimate(rows, ratio(lucky, lambda r: r["y0"]), resamples),
            "luck_rate": estimate(rows, mean(lambda r: lucky(r) + unlucky(r)), resamples),
            "any_placebo_flip_rate": estimate(rows, mean(lambda r: int(any(y(r)[a] != r["y0"] for a in r["placebo"]))), resamples)}


def flip_table(rows):
    out = {}
    for arm in PLACEBO_ARMS:
        have = [r for r in rows if arm in r["raw"]]
        correct = [r for r in have if r["y0"] == 1]
        wrong = [r for r in have if r["y0"] == 0]
        kinds = collections.Counter(r["kinds"][arm] for r in have)
        out[arm] = {"states": len(have), "degraded_raw": sum(r["raw"][arm] == 0 for r in correct),
                    "improved_raw": sum(r["raw"][arm] == 1 for r in wrong),
                    "clean_correct": len(correct), "clean_wrong": len(wrong),
                    "format_flips": kinds["format"], "semantic_flips": kinds["semantic"],
                    "degraded_semantic": sum(r["sem"][arm] == 0 for r in correct),
                    "improved_semantic": sum(r["sem"][arm] == 1 for r in wrong)}
    return out


def strata(rows, resamples):
    keys = {"turn": lambda r: str(r["turn"]), "dataset": lambda r: r["dataset"],
            "answer_type": lambda r: r["answer_type"], "source_search_calls": lambda r: str(r["source_search_calls"]),
            "evidence_retrieved_proxy": lambda r: str(r["evidence_retrieved_proxy"]),
            "target_has_support": lambda r: str(r["target_has_support"])}
    out = {}
    for name, key in keys.items():
        out[name] = {}
        for value in sorted({key(r) for r in rows}):
            sub = [r for r in rows if key(r) == value]
            out[name][value] = {"states": len(sub), **{k: v for k, v in luck_block(sub, "sem", resamples // 4).items()
                                                      if k in ("clean_em", "mEM", "lucky_share_of_accuracy", "luck_rate")}}
    return out


def recovery(rows, view, resamples):
    have = [r for r in rows if all(a in r[view] for a in ARM_GROUPS["recovery"])]
    if not have:
        return {"available": False}
    y = lambda r, a: r[view][a]
    natural = {}
    for arm in [a for a in ("review", "neutral", "budget2", "reflect", "neutral_reflect") if all(a in r[view] for r in have)]:
        natural[arm] = {"rescue": sum(r["y0"] == 0 and y(r, arm) == 1 for r in have),
                        "harm": sum(r["y0"] == 1 and y(r, arm) == 0 for r in have),
                        "delta_em_vs_clean": estimate(have, mean(lambda r, a=arm: y(r, a) - r["y0"]), resamples),
                        "nce_vs_placebo_mean": estimate(have, mean(lambda r, a=arm: y(r, a) - r["p_" + view]), resamples)}
    natural["review_minus_neutral"] = estimate(have, mean(lambda r: y(r, "review") - y(r, "neutral")), resamples)
    natural["placebo_expected_rescue"] = sum(r["p_" + view] for r in have if r["y0"] == 0)
    natural["placebo_expected_harm"] = sum(1 - r["p_" + view] for r in have if r["y0"] == 1)
    degraded = [r for r in have if r["y0"] == 1 and y(r, "perm_1") == 0]
    stable = [r for r in have if r["y0"] == 1 and y(r, "perm_1") == 1]
    rate = lambda group, f: sum(f(r) for r in group) / len(group) if group else None
    placebo_repair = {"P1_reperturb_perm2": rate(degraded, lambda r: y(r, "perm_2")),
                      "P2_neutral_on_perm1": rate(degraded, lambda r: y(r, "perm1_neutral"))}
    placebo_harm = {"P1_reperturb_perm2": rate(stable, lambda r: 1 - y(r, "perm_2")),
                    "P2_neutral_on_perm1": rate(stable, lambda r: 1 - y(r, "perm1_neutral"))}
    methods = {}
    # Each instruction method has its own token-matched neutral placebo (P2); budget2 uses the review-matched one.
    neutral_for = {"review": "perm1_neutral", "budget2": "perm1_neutral", "reflect": "perm1_neutral_reflect"}
    for method in ("review", "budget2", "reflect"):
        arm = "perm1_" + method
        if not all(arm in r[view] and neutral_for[method] in r[view] for r in have):
            continue
        repair, harm = rate(degraded, lambda r: y(r, arm)), rate(stable, lambda r: 1 - y(r, arm))
        p2 = rate(degraded, lambda r: y(r, neutral_for[method]))
        best = max(v for v in (placebo_repair["P1_reperturb_perm2"], p2) if v is not None) if degraded else None
        ph = rate(stable, lambda r: 1 - y(r, neutral_for[method]))
        raw_net = (repair * len(degraded) - harm * len(stable)) if degraded and stable else None
        adj_net = ((repair - best) * len(degraded) - (harm - ph) * len(stable)) if degraded and stable else None
        methods[method] = {"repair_rate_on_degraded": repair, "harm_rate_on_stable": harm,
                           "P2_neutral_repair": p2, "P2_neutral_harm": ph,
                           "placebo_explained_fraction": (best / repair) if repair else None,
                           "uncorrected_net_count": raw_net, "placebo_corrected_net_count": adj_net,
                           "direction_reversed": (raw_net is not None and adj_net is not None and raw_net * adj_net < 0),
                           "repair_minus_reperturb": estimate(degraded, mean(lambda r, a=arm: y(r, a) - y(r, "perm_2")), resamples) if degraded else None}
    return {"available": True, "states": len(have), "natural_states": natural,
            "selected_fragile": {"selection": "clean correct and perm_1 wrong (B-style); stable = clean correct and perm_1 correct",
                                 "degraded_states": len(degraded), "stable_states": len(stable),
                                 "placebo_repair_rates": placebo_repair, "placebo_harm_rates": placebo_harm,
                                 "methods": methods}}


def deletion(rows, view, resamples):
    both = [r for r in rows if r["deletion"]["drop_support"]["eligible"] and r["deletion"]["drop_nonsupport"]["eligible"]
            and "drop_support" in r[view] and "drop_nonsupport" in r[view]]
    if not both:
        return {"available": False}
    y = lambda r, a: r[view][a]
    def block(sub):
        return {"nce_drop_support": estimate(sub, mean(lambda r: y(r, "drop_support") - r["p_" + view]), resamples),
                "nce_drop_nonsupport": estimate(sub, mean(lambda r: y(r, "drop_nonsupport") - r["p_" + view]), resamples),
                "support_minus_nonsupport": estimate(sub, mean(lambda r: y(r, "drop_support") - y(r, "drop_nonsupport")), resamples),
                "permutation_delta": estimate(sub, mean(lambda r: y(r, "perm_1") - r["y0"]), resamples),
                "drop_support_delta": estimate(sub, mean(lambda r: y(r, "drop_support") - r["y0"]), resamples)}
    unique = [r for r in both if r["deletion"]["drop_support"]["lost_fact_not_seen_earlier"]]
    return {"available": True, "matched_states": len(both),
            "eligibility": dict(collections.Counter(
                (r["deletion"]["drop_support"].get("reason") or "eligible") for r in rows)),
            "all_matched": block(both), "support_unique_this_turn": block(unique) if unique else None,
            "clean_correct_matched": block([r for r in both if r["y0"] == 1]) if any(r["y0"] for r in both) else None}


def gates(luck, rec, dele):
    share = luck["lucky_share_of_accuracy"]
    g1 = share["estimate"] is not None and share["ci95"] is not None and share["estimate"] >= .10 and share["ci95"][0] > .05
    g2 = None
    if rec.get("available"):
        ms = rec["selected_fragile"]["methods"].values()
        g2 = any((m["placebo_explained_fraction"] or 0) >= 1 / 3 or m["direction_reversed"] for m in ms)
    g3 = None
    if dele.get("available"):
        diff = dele["all_matched"]["support_minus_nonsupport"]["ci95"]
        g3 = diff is not None and diff[1] < 0
    return {"G1_semantic_lucky_share_ge_10pct_ci_lo_gt_5pct": g1,
            "G2_recovery_placebo_explains_ge_one_third_or_reverses": g2,
            "G3_support_deletion_distinguishable_from_nonsupport": g3,
            "note": "Mechanical gate values only. Record the human GO/NO-GO in research/DECISION_LOG.md per research/PCR_PROTOCOL.md section 5."}


def si1_crosscheck(states, si1_dir):
    cases = {c["sample_id"]: c for c in read_rows(Path(si1_dir) / "cases.jsonl")}
    checked = same = 0
    for s in states:
        c = cases.get(s["sample_id"])
        if s["target_turn"] != 1 or c is None or "review" not in s["arms"] or not c["state"]["reachable"]:
            continue
        checked += 1
        same += s["arms"]["review"]["final_answer"] == c["arms"]["review"]["final_answer"] and s["arms"]["review"]["em"] == c["arms"]["review"]["em"]
    return {"turn1_review_states_compared": checked, "identical_answer_and_em": same}


def cases(states, derived, limit=60):
    out = []
    for s, d in zip(states, derived):
        for arm in PERMUTATION_ARMS:
            if d["kinds"].get(arm) == "semantic":
                out.append({"sample_id": s["sample_id"], "turn": s["target_turn"], "dataset": s["dataset"],
                            "question": s["question"], "gold": s["gold_answer"], "clean_answer": s["arms"]["clean_a"]["final_answer"],
                            "clean_em": d["y0"], "arm": arm, "order": s["arms"][arm]["metadata"]["order"],
                            "arm_answer": s["arms"][arm]["final_answer"], "arm_em": s["arms"][arm]["em"],
                            "clean_queries_after": s["arms"]["clean_a"]["post_target_queries"],
                            "arm_queries_after": s["arms"][arm]["post_target_queries"], "p_sem": d["p_sem"]})
                break
        if len(out) >= limit:
            break
    return out


def analyze(run_dir, resamples=2000, si1_dir=None):
    manifest = load_json(run_dir / "run_manifest.json")
    if manifest.get("status") != "completed" or file_hash(run_dir / "states.jsonl") != manifest["states_sha256"]:
        raise ValueError("Need a completed, unmodified merged PCR run")
    states = read_rows(run_dir / "states.jsonl")
    if any(s["version"] != VERSION or not s["controls_exact"] for s in states):
        raise ValueError("Unexpected PCR version or failed control receipt")
    for s in states:
        for arm, row in s["arms"].items():
            if row is not None and row["status"] == "answered" and row["em"] != answer_scores(row["final_answer"], s["gold_answer"])["em"]:
                raise ValueError("Raw score mismatch: {} {}".format(s["sample_id"], arm))
    derived = [derive(s) for s in states]
    report = {"version": VERSION, "analysis_commit": git_output("rev-parse", "HEAD"),
              "states_sha256": manifest["states_sha256"], "states": len(states),
              "questions": len({s["sample_id"] for s in states}),
              "format_flip_rule": "EM flip whose two answers are equal after normalization, contain each other, or have token F1 >= {}".format(FORMAT_F1),
              "luck": {"semantic": luck_block(derived, "sem", resamples), "raw_em": luck_block(derived, "raw", resamples)},
              "by_turn": {str(t): luck_block([d for d in derived if d["turn"] == t], "sem", resamples)
                          for t in sorted({d["turn"] for d in derived})},
              "flip_table": flip_table(derived), "strata": strata(derived, resamples),
              "recovery": {"semantic": recovery(derived, "sem", resamples), "raw_em": recovery(derived, "raw", resamples)},
              "deletion": {"semantic": deletion(derived, "sem", resamples), "raw_em": deletion(derived, "raw", resamples)},
              "bootstrap": {"unit": "question (all turns kept together)", "strata": "dataset", "resamples": resamples, "seed": 42}}
    report["gates"] = gates(report["luck"]["semantic"], report["recovery"]["semantic"], report["deletion"]["semantic"])
    if si1_dir:
        report["si1_review_crosscheck"] = si1_crosscheck(states, si1_dir)
    return report, cases(states, derived)


def fmt(e):
    if not isinstance(e, dict) or e.get("estimate") is None:
        return "NA"
    ci = e.get("ci95")
    return "{:.4f} [{}]".format(e["estimate"], "{:.4f}, {:.4f}".format(*ci) if ci else "no CI")


def markdown(r):
    L = ["# PCR v1 结果摘要（自动生成，供回传）", "",
         "states={} questions={} states_sha256={}".format(r["states"], r["questions"], r["states_sha256"][:12]), ""]
    for view in ("semantic", "raw_em"):
        b = r["luck"][view]
        L += ["## 路径运气（{}）".format(view), "",
              "| 指标 | 值 [95% CI] |", "| --- | --- |",
              "| clean EM | {} |".format(fmt(b["clean_em"])), "| mEM | {} |".format(fmt(b["mEM"])),
              "| rEM（全部安慰剂都对） | {} |".format(fmt(b["rEM_all_placebos"])),
              "| lucky share of accuracy | {} |".format(fmt(b["lucky_share_of_accuracy"])),
              "| luck rate | {} |".format(fmt(b["luck_rate"])),
              "| lucky / unlucky / clean correct | {} / {} / {} |".format(b["lucky_successes"], b["unlucky_failures"], b["clean_correct"]), ""]
    L += ["## 逐臂翻转", "", "| arm | degraded raw/sem | improved raw/sem | format flips | semantic flips |", "| --- | --- | --- | --- | --- |"]
    for arm, t in r["flip_table"].items():
        L.append("| {} | {}/{} of {} | {}/{} of {} | {} | {} |".format(arm, t["degraded_raw"], t["degraded_semantic"], t["clean_correct"],
                                                                     t["improved_raw"], t["improved_semantic"], t["clean_wrong"], t["format_flips"], t["semantic_flips"]))
    rec = r["recovery"]["semantic"]
    if rec.get("available"):
        sf = rec["selected_fragile"]
        L += ["", "## Exp-2 恢复（semantic）", "", "degraded={} stable={}；安慰剂修复率 {}；安慰剂误伤率 {}".format(
            sf["degraded_states"], sf["stable_states"], sf["placebo_repair_rates"], sf["placebo_harm_rates"]), ""]
        for m, v in sf["methods"].items():
            L.append("- {}: {}".format(m, json.dumps(v, ensure_ascii=False)))
        L.append("- natural review−neutral: {}".format(fmt(rec["natural_states"]["review_minus_neutral"])))
    dele = r["deletion"]["semantic"]
    if dele.get("available"):
        a = dele["all_matched"]
        L += ["", "## Exp-3 删证据（semantic，matched={}）".format(dele["matched_states"]), "",
              "- NCE drop_support: {}".format(fmt(a["nce_drop_support"])),
              "- NCE drop_nonsupport: {}".format(fmt(a["nce_drop_nonsupport"])),
              "- support − nonsupport: {}".format(fmt(a["support_minus_nonsupport"]))]
    L += ["", "## 预登记门槛（机械值，GO 由人工记录）", "", "```json", json.dumps(r["gates"], ensure_ascii=False, indent=2), "```", ""]
    if "si1_review_crosscheck" in r:
        L += ["SI1 review 交叉复现：{}".format(r["si1_review_crosscheck"]), ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--si1-run-dir", type=Path, help="optional SI1 dev dir: turn-1 review arm determinism cross-check")
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    out = a.output_dir or a.run_dir / "analysis"
    try:
        if out.exists():
            raise ValueError("Output exists; use a new --output-dir")
        report, flips = analyze(a.run_dir, a.resamples, a.si1_run_dir)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    out.mkdir(parents=True)
    write_json(out / "pcr_report.json", report)
    (out / "pcr_report.md").write_text(markdown(report), encoding="utf-8")
    with (out / "flip_cases.jsonl").open("x", encoding="utf-8") as f:
        for row in flips:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # pack_feedback deliberately refuses .jsonl (raw trajectories); this small case list ships as JSON.
    write_json(out / "flip_cases.json", flips)
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

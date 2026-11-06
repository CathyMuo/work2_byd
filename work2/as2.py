"""AS2 analysis (research/AS2_PROTOCOL.md): filler variants, graded positive controls and sampled decoding.

Runs live in outputs/u1/<label>/as2 (experiments 1 and 2, all fourteen agents) and outputs/u1/<label>/as2s (experiment 3, four
Search-R1 agents). States are the AS1 states, so AS1's alert and hint arms are joined by (sample_id, turn) for the graded
ladder. Readings F1-F3, G1-G2 and S1-S2 are computed mechanically; the decision is written by hand into DECISION_LOG.md.
Question-cluster bootstrap stratified by dataset, all agents' states of a question kept together. CPU only.
"""
import argparse
from pathlib import Path

from .as1u1n import AGENTS
from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows
from .paper_h1 import fmt
from .paper_robust import boot
from .pcr_analysis import flip_kind
from .placebo import usable_states

SAMPLED_AGENTS = ("v03_3b_ppo", "v03_7b_grpo", "v03_14b_ppo", "v03_32b_grpo")
VARIANTS = {"prose": "perm1_fill_prose", "protocol": "perm1_fill_protocol", "shuffled": "perm1_fill_shuffled",
            "punct": "perm1_fill_punct", "syntax": "perm1_fill_syntax"}
ORDERS = ("perm_1", "perm_2", "perm_3", "perm_4", "perm_5")   # amendment 1: every non-identity order of a top-3 observation
I1_PP = 0.02
GRADED = {"pointer": ("perm1_pointer", "perm1_neutral_pointer"), "evidence": ("perm1_evidence", "perm1_neutral_evidence")}
LADDER = {"review": ("perm1_review", "perm1_neutral"), "alert": ("perm1_alert", "perm1_neutral_alert"),
          "hint": ("perm1_hint", "perm1_neutral_hint")}           # joined from AS1 on the same states
SAMPLED = {"review": ("s_perm1_review", "s_perm1_neutral"), "reflect": ("s_perm1_reflect", "s_perm1_neutral_reflect")}
LUCK = ("s_perm1", "s_perm2")
F1_PP, F2_PP = 0.03, 0.02
G1_PP, G2_PP = 0.08, 0.20
S1_PP, EXPLAINED = 0.05, 2 / 3
REPRO_MIN = 0.99
REPRO_ARMS = {"as2": ("clean_a", "perm_1", "perm1_review", "perm1_neutral"), "as2s": ("clean_a", "perm_1", "perm_2")}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def derive(state):
    """Raw and semantic outcomes per arm; a sampled arm's outcome is the mean over its samples. Ineligible arms are absent."""
    arms = state["arms"]
    clean = arms["clean_a"]
    y0 = clean["em"]

    def one(row):
        em = row["em"] or 0
        return em, (y0 if flip_kind(clean["final_answer"], y0, row["final_answer"], em) == "format" else em)

    raw, sem, meta = {}, {}, {}
    for name, row in arms.items():
        if row is None:
            continue
        if "samples" in row:
            pairs = [one(s) for s in row["samples"]]
            raw[name], sem[name] = mean(p[0] for p in pairs), mean(p[1] for p in pairs)
        else:
            raw[name], sem[name] = one(row)
            meta[name] = row.get("metadata", {})
    return {"sample_id": state["sample_id"], "dataset": state["dataset"], "turn": state["target_turn"], "y0": y0,
            "raw": raw, "sem": sem, "meta": meta}


def run_rows(run_dir):
    m = load_json(run_dir / "run_manifest.json")
    if m.get("status") != "completed" or file_hash(run_dir / "states.jsonl") != m["states_sha256"]:
        raise ValueError("Need a completed, unmodified run: " + str(run_dir))
    return [derive(s) for s in usable_states(read_rows(run_dir / "states.jsonl"), m)]


def split(rows, view):
    y = lambda r: r[view]["perm_1"]
    return ([r for r in rows if r["y0"] == 1 and y(r) == 0], [r for r in rows if r["y0"] == 1 and y(r) == 1],
            [r for r in rows if r["y0"] == 0])


def rate(group, f):
    return mean(f(r) for r in group) if group else None


def contrast(group, a, b, view, resamples):
    """Mean of y(a) - y(b) over the states where both arms ran."""
    g = [r for r in group if a in r[view] and b in r[view]]
    return boot(g, lambda r: r[view][a] - r[view][b], resamples)


def filler_block(rows, resamples, view="sem"):
    deg, st, wr = split(rows, view)
    out = {"states": len(rows), "degraded": len(deg), "stable": len(st), "clean_wrong": len(wr),
           "review_repair": rate(deg, lambda r: r[view]["perm1_review"]), "p2_repair": rate(deg, lambda r: r[view]["perm1_neutral"]),
           "review_minus_p2": contrast(deg, "perm1_review", "perm1_neutral", view, resamples),
           "harm_review_vs_p2": contrast(st, "perm1_neutral", "perm1_review", view, resamples), "variants": {}}
    for name, arm in VARIANTS.items():
        clean = lambda g, a=arm: [r for r in g if a in r[view] and not r["meta"].get(a, {}).get("leaks_answer")]
        d, s, w = clean(deg), clean(st), clean(wr)
        out["variants"][name] = {
            "leaks": sum(bool(r["meta"].get(arm, {}).get("leaks_answer")) for r in rows), "repair": rate(d, lambda r, a=arm: r[view][a]),
            "review_minus_v": contrast(d, "perm1_review", arm, view, resamples),
            "p2_minus_v": contrast(d, "perm1_neutral", arm, view, resamples),        # = (review - V) - (review - P2)
            "harm_review_vs_v": contrast(s, arm, "perm1_review", view, resamples),  # harm(review) - harm(V)
            "harm_v_vs_p2": contrast(s, "perm1_neutral", arm, view, resamples),     # harm(V) - harm(P2)
            "rescue_review_vs_v": contrast(w, "perm1_review", arm, view, resamples)}
    return out


def interface_block(rows, resamples, view="sem"):
    """Amendment 1: the review prompt and its filler inside the observation block vs after it (the H1 position)."""
    deg, st, wr = split(rows, view)
    have = lambda g: [r for r in g if "perm1_review_inobs" in r[view] and "perm1_neutral_inobs" in r[view]]
    d, s = have(deg), have(st)
    inside = lambda r: r[view]["perm1_review_inobs"] - r[view]["perm1_neutral_inobs"]
    after = lambda r: r[view]["perm1_review"] - r[view]["perm1_neutral"]
    return {"degraded": len(d), "stable": len(s),
            "repair_content_inside": boot(d, inside, resamples), "repair_content_after": boot(d, after, resamples),
            "harm_content_inside": boot(s, lambda r: -inside(r), resamples), "harm_content_after": boot(s, lambda r: -after(r), resamples),
            "harm_content_inside_minus_after": boot(s, lambda r: after(r) - inside(r), resamples),
            "review_inside_minus_after_harm": contrast(s, "perm1_review", "perm1_review_inobs", view, resamples),
            "filler_inside_minus_after_harm": contrast(s, "perm1_neutral", "perm1_neutral_inobs", view, resamples)}


def stable_weights(rows, plan_path):
    """Amendment 2 (bug fix): the AS1 selection keeps every degraded state but only a hash sample of the states stable in both views,
    so each such sampled state stands for available / selected of them. |S| in the luck model is the full stable set."""
    sel = load_json(plan_path)["selected_from"]
    w = sel["available"]["stable"] / sel["selected"]["stable"] if sel["selected"]["stable"] else 1.0
    for r in rows:
        r["w"] = w if (r["raw"].get("perm_1") == 1 and r["sem"].get("perm_1") == 1 and r["y0"] == 1) else 1.0
    return {"available_stable": sel["available"]["stable"], "selected_stable": sel["selected"]["stable"], "weight": w}


def boot_gap(deg, st, a, resamples, view="sem", seed=42):
    """Observed repair of perm_k on D minus its luck-model prediction sum_S w (1 - y) / |D|, with both terms resampled together
    (question clusters within dataset), so the interval includes the prediction's own sampling error."""
    import random
    if not deg:
        return None
    cl = {}
    for r in deg:
        c = cl.setdefault((r["dataset"], r["sample_id"]), [0.0, 0.0, 0])
        c[0] += r[view][a]; c[2] += 1
    for r in st:
        c = cl.setdefault((r["dataset"], r["sample_id"]), [0.0, 0.0, 0])
        c[1] += r.get("w", 1.0) * (1 - r[view][a])
    groups = [[v for k, v in cl.items() if k[0] == ds] for ds in sorted({k[0] for k in cl})]
    stat = lambda cs: (sum(c[0] for c in cs) - sum(c[1] for c in cs)) / max(1, sum(c[2] for c in cs))
    rng, draws = random.Random(seed), []
    for _ in range(resamples):
        draws.append(stat([c for g in groups for c in rng.choices(g, k=len(g))]))
    draws.sort()
    return {"estimate": stat(list(cl.values())), "ci95": [draws[int((resamples - 1) * .025)], draws[int((resamples - 1) * .975)]]}


def luck_block(rows, resamples, view="sem"):
    """Amendment 1: with all five non-identity orders, each re-draw perm_k (k = 2..5) is exchangeable with perm_1, so its repair on
    D should equal harm_S(perm_k) * |S| / |D| and the four repair rates should agree. Amendment 2: |S| is the full stable set
    (sampled stable states carry weight r["w"]), and the gap observed - predicted is reported with a joint interval."""
    rows = [r for r in rows if all(a in r[view] for a in ORDERS)]
    deg, st, _ = split(rows, view)
    s_full = sum(r.get("w", 1.0) for r in st)
    out = {"degraded": len(deg), "stable": len(st), "stable_represented": s_full, "orders": {}}
    for a in ORDERS[1:]:
        harm_sum = sum(r.get("w", 1.0) * (1 - r[view][a]) for r in st)
        out["orders"][a] = {"repair": boot(deg, lambda r, a=a: r[view][a], resamples), "harm": harm_sum / s_full if s_full else None,
                            "harm_events_in_sample": sum(1 - r[view][a] for r in st),
                            "predicted_repair": (harm_sum / len(deg)) if deg else None,
                            "gap": boot_gap(deg, st, a, resamples, view)}
    reps = [v["repair"]["estimate"] for v in out["orders"].values() if v["repair"]]
    out["repair_spread"] = (max(reps) - min(reps)) if reps else None
    out["mean_redraw_repair"] = boot(deg, lambda r: mean(r[view][a] for a in ORDERS[1:]), resamples)
    # share of degraded states that every one of the four re-draws leaves wrong (order-insensitive failures after selection)
    out["never_repaired"] = rate(deg, lambda r: float(all(r[view][a] == 0 for a in ORDERS[1:])))
    return out


def graded_block(rows, resamples, view="sem"):
    deg, st, wr = split(rows, view)
    out = {"degraded": len(deg), "contents": {}}
    for name, (arm, filler) in GRADED.items():
        d = [r for r in deg if arm in r[view]]
        entry = {"eligible_degraded": len(d), "eligibility": len(d) / len(deg) if deg else None,
                 "repair": rate(d, lambda r, a=arm: r[view][a]), "filler_repair": rate(d, lambda r, f=filler: r[view][f]),
                 "repair_minus_filler": contrast(d, arm, filler, view, resamples),
                 "harm_minus_filler": contrast(st, filler, arm, view, resamples),
                 "rescue_minus_filler": contrast(wr, arm, filler, view, resamples), "ladder": {}}
        # The same eligible degraded states under the AS1 contents (joined), so the ladder compares like with like.
        joined = [r for r in d if r.get("as1")]
        entry["ladder_states"] = len(joined)
        for other, (a, f) in LADDER.items():
            entry["ladder"][other] = boot([r for r in joined if a in r["as1"][view] and f in r["as1"][view]],
                                          lambda r, a=a, f=f: r["as1"][view][a] - r["as1"][view][f], resamples)
        entry["ladder"][name] = contrast(joined, arm, filler, view, resamples)
        out["contents"][name] = entry
    return out


def sampled_block(rows, resamples, view="sem"):
    deg, st, wr = split(rows, view)
    out = {"states": len(rows), "degraded": len(deg), "stable": len(st), "clean_wrong": len(wr),
           "luck": {a: rate(deg, lambda r, a=a: r[view][a]) for a in LUCK}, "methods": {}}
    for name, (arm, filler) in SAMPLED.items():
        rep, fil = rate(deg, lambda r: r[view][arm]), rate(deg, lambda r: r[view][filler])
        best = max(v for v in (fil, *out["luck"].values()) if v is not None) if deg else None
        out["methods"][name] = {"repair": rep, "filler_repair": fil, "explained": (best / rep) if rep else None,
                                "repair_minus_filler": contrast(deg, arm, filler, view, resamples),
                                "repair_minus_resample": contrast(deg, arm, "s_perm1", view, resamples),
                                "harm_minus_filler": contrast(st, filler, arm, view, resamples),
                                "rescue_minus_filler": contrast(wr, arm, filler, view, resamples)}
    return out


def reproduced(h1_dir, run_dir, arms):
    """Share of shared states whose greedy continuations match the H1 run byte for byte, per arm."""
    old = {(r["sample_id"], r["target_turn"]): r for r in read_rows(h1_dir / "states.jsonl")}
    new = read_rows(run_dir / "states.jsonl")
    out = {}
    for arm in arms:
        pairs = [(old[k]["arms"][arm], r["arms"][arm]) for r in new for k in [(r["sample_id"], r["target_turn"])] if k in old]
        out[arm] = sum(a["continuation_sha256"] == b["continuation_sha256"] for a, b in pairs) / len(pairs) if pairs else None
    return out


def above(e, pp):
    return bool(e and e["estimate"] >= pp and e["ci95"][0] > 0)


def readings(r):
    out = {}
    f = r["filler"]["pooled"]["sem"] if r["filler"]["pooled"] else None
    if f:
        out["F1_variants"] = [k for k, v in f["variants"].items() if above(v["p2_minus_v"], F1_PP)]
        out["F1"] = bool(out["F1_variants"])
        out["F2"] = above(f["variants"]["protocol"]["harm_v_vs_p2"], F2_PP)
        out["F3"] = not out["F1"]
    g = r["graded"]["pooled"]["sem"] if r["graded"]["pooled"] else None
    if g:
        e = g["contents"]["pointer"]["repair_minus_filler"]
        out["G1"] = bool(e and e["estimate"] > G1_PP and e["ci95"][0] > 0)
        out["G2"] = above(g["contents"]["evidence"]["repair_minus_filler"], G2_PP)
        out["G_reading"] = ("information ladder" if out["G1"] and out["G2"] else "only answer-bearing content clears the bound"
                            if out["G2"] else "report as is; positive control shows oracle-level content only")
    i = r["interface"]["pooled"]["sem"] if r["interface"]["pooled"] else None
    if i:
        e = i["harm_content_inside_minus_after"]
        out["I1"] = bool(e and e["estimate"] <= -I1_PP and e["ci95"][1] < 0)   # less excess harm inside: interface explains part
    lk = r["luck"]["pooled"]["sem"] if r["luck"]["pooled"] else None
    if lk:
        cis = [v["repair"]["ci95"] for v in lk["orders"].values() if v["repair"]]
        out["L1"] = bool(cis) and max(c[0] for c in cis) <= min(c[1] for c in cis)   # the four re-draws' intervals overlap
        out["L2"] = all(v["predicted_repair"] is not None and v["repair"]["ci95"][0] <= v["predicted_repair"] <= v["repair"]["ci95"][1]
                        for v in lk["orders"].values() if v["repair"])
        out["L2_gap_covers_zero"] = all(v["gap"] and v["gap"]["ci95"][0] <= 0 <= v["gap"]["ci95"][1] for v in lk["orders"].values())
    lf = r["luck_full"]["pooled"]["sem"] if r.get("luck_full") and r["luck_full"]["pooled"] else None
    if lf:   # amendment 2: the same test on the full stable sets
        out["L2_full"] = all(v["predicted_repair"] is not None and v["repair"]["ci95"][0] <= v["predicted_repair"] <= v["repair"]["ci95"][1]
                             for v in lf["orders"].values() if v["repair"])
        out["L2_full_gap_covers_zero"] = all(v["gap"] and v["gap"]["ci95"][0] <= 0 <= v["gap"]["ci95"][1] for v in lf["orders"].values())
        out["L_full_reading"] = ("luck model confirmed on every re-draw" if out["L2_full_gap_covers_zero"]
                                 else "luck model rejected for some re-draw; narrow the mechanism claim")
    s = r["sampled"]["pooled"]["sem"] if r["sampled"]["pooled"] else None
    if s:
        out["S1"] = any(above(m["repair_minus_filler"], S1_PP) for m in s["methods"].values())
        out["S2"] = all(m["explained"] is not None and m["explained"] >= EXPLAINED for m in s["methods"].values())
        out["S_reading"] = "unchanged under sampling" if out["S2"] and not out["S1"] else "report as is"
    reps = [v for part in ("as2", "as2s", "as2l") for agent in r["reproduction"].get(part, {}).values() for v in agent.values() if v is not None]
    out["repro_min"] = min(reps) if reps else None
    out["repro_ok"] = out["repro_min"] is not None and out["repro_min"] >= REPRO_MIN
    return out


def analyze(root, labels, sampled_labels, resamples):
    report = {"analysis_commit": git_output("rev-parse", "HEAD"), "labels": list(labels), "sampled_labels": list(sampled_labels),
              "bootstrap": {"unit": "question (all agents' states of a question together)", "strata": "dataset", "resamples": resamples},
              "reproduction": {"as2": {}, "as2s": {}, "as2l": {}}, "stable_weights": {}, "stable_weights_full": {}}
    pooled = {"as2": [], "as2s": [], "as2l": []}
    per = {"filler": {}, "graded": {}, "interface": {}, "sampled": {}, "luck": {}, "luck_full": {}}
    for label in labels:
        d = root / label / "as2"
        if not (d / "states.jsonl").is_file():
            per["filler"][label] = per["graded"][label] = per["interface"][label] = None
            continue
        rows = run_rows(d)
        as1 = root / label / "as1"
        if (as1 / "states.jsonl").is_file():
            joined = {(x["sample_id"], x["turn"]): x for x in run_rows(as1)}
            for x in rows:
                x["as1"] = joined.get((x["sample_id"], x["turn"]))
        pooled["as2"] += rows
        per["filler"][label] = {v: filler_block(rows, resamples, v) for v in ("sem", "raw")}
        per["graded"][label] = {v: graded_block(rows, resamples, v) for v in ("sem", "raw")}
        per["interface"][label] = {v: interface_block(rows, resamples, v) for v in ("sem", "raw")}
        report["reproduction"]["as2"][label] = reproduced(root / label / "pcr", d, REPRO_ARMS["as2"])
    for label in sampled_labels:
        d = root / label / "as2s"
        if not (d / "states.jsonl").is_file():
            per["sampled"][label] = per["luck"][label] = None
            continue
        rows = run_rows(d)
        report["stable_weights"][label] = stable_weights(rows, root / label / "as2s_plan" / "plan.json")
        pooled["as2s"] += rows
        per["sampled"][label] = {v: sampled_block(rows, resamples, v) for v in ("sem", "raw")}
        per["luck"][label] = {v: luck_block(rows, resamples, v) for v in ("sem", "raw")}
        report["reproduction"]["as2s"][label] = reproduced(root / label / "pcr", d, REPRO_ARMS["as2s"])
        # amendment 2: perm_1..perm_5 on every degraded and every stable state (greedy)
        d = root / label / "as2l"
        if (d / "states.jsonl").is_file():
            rows = run_rows(d)
            report["stable_weights_full"][label] = stable_weights(rows, root / label / "as2l_plan" / "plan.json")
            pooled["as2l"] += rows
            per["luck_full"][label] = {v: luck_block(rows, resamples, v) for v in ("sem", "raw")}
            report["reproduction"]["as2l"][label] = reproduced(root / label / "pcr", d, REPRO_ARMS["as2s"])
        else:
            per["luck_full"][label] = None
    if not pooled["as2"] and not pooled["as2s"]:
        raise ValueError("No completed AS2 run under " + str(root))
    for part, key, block in (("filler", "as2", filler_block), ("graded", "as2", graded_block), ("interface", "as2", interface_block),
                             ("sampled", "as2s", sampled_block), ("luck", "as2s", luck_block), ("luck_full", "as2l", luck_block)):
        report[part] = {"agents": per[part], "pooled": {v: block(pooled[key], resamples, v) for v in ("sem", "raw")} if pooled[key] else None}
    report["readings"] = readings(report)
    return report


def pc(v):
    return "NA" if v is None else "{:.1f}".format(100 * v)


def markdown(r):
    L = ["# AS2 结果（自动生成，供回传）", "", "单位：百分点；[] 为按题聚类 bootstrap 95% CI。语义口径为主。", "",
         "## 预登记判定（机械值）", "", "```", str(r["readings"]), "```", "", "## 复现（与 H1 逐字相同的比例）", "", "```",
         str(r["reproduction"]), "```", ""]
    if r["filler"]["pooled"]:
        L += ["## 实验 1：填充变体（review 等长；退化状态上 review − V，Δ_V = P2 − V；稳定状态上 harm(V) − harm(P2)）", "",
              "| 智能体 | D | review − P2 | " + " | ".join("{0} 修复 / review − {0} / Δ / 伤害差".format(k) for k in VARIANTS) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in VARIANTS) + " |"]
        for name, v in list(r["filler"]["agents"].items()) + [("**合并**", r["filler"]["pooled"])]:
            if not v:
                L.append("| {} | 缺失 |".format(name)); continue
            b = v["sem"]
            cells = [name, str(b["degraded"]), fmt(b["review_minus_p2"])]
            for k in VARIANTS:
                x = b["variants"][k]
                cells.append("{} / {} / {} / {}{}".format(pc(x["repair"]), fmt(x["review_minus_v"]), fmt(x["p2_minus_v"]), fmt(x["harm_v_vs_p2"]),
                                                          " （泄漏 {}）".format(x["leaks"]) if x["leaks"] else ""))
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
    if r["graded"]["pooled"]:
        L += ["## 实验 2：分级阳性对照（合格的退化状态上，内容 − 其等长文字）", "",
              "| 智能体 | D | pointer 合格 | pointer 修复差 | pointer 伤害差 | evidence 合格 | evidence 修复差 | evidence 伤害差 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, v in list(r["graded"]["agents"].items()) + [("**合并**", r["graded"]["pooled"])]:
            if not v:
                L.append("| {} | 缺失 |".format(name)); continue
            b = v["sem"]; p, e = b["contents"]["pointer"], b["contents"]["evidence"]
            L.append("| {} | {} | {} ({}) | {} | {} | {} ({}) | {} | {} |".format(
                name, b["degraded"], p["eligible_degraded"], pc(p["eligibility"]), fmt(p["repair_minus_filler"]), fmt(p["harm_minus_filler"]),
                e["eligible_degraded"], pc(e["eligibility"]), fmt(e["repair_minus_filler"]), fmt(e["harm_minus_filler"])))
        g = r["graded"]["pooled"]["sem"]["contents"]
        L += ["", "**剂量阶梯（合并；同一批合格状态上，与 AS1 拼接）**", ""]
        for name in GRADED:
            lad = g[name]["ladder"]
            L.append("- {} 合格状态（{} 个）：".format(name, g[name]["ladder_states"]) +
                     "，".join("{} {}".format(k, fmt(lad.get(k))) for k in ("review", "alert", "pointer", "evidence", "hint") if k in lad))
        L.append("")
    if r["sampled"]["pooled"]:
        L += ["## 实验 3：采样解码（T = 1.0，k = 4；退化状态上的修复率为样本均值）", "",
              "| 智能体 | D | 稳定 | clean 错 | 再采样 s_perm1 | P1 s_perm2 | review / 文字 | review 修复差 | review 解释比例 | review 伤害差 | reflect / 文字 | reflect 修复差 | reflect 解释比例 | reflect 伤害差 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, v in list(r["sampled"]["agents"].items()) + [("**合并**", r["sampled"]["pooled"])]:
            if not v:
                L.append("| {} | 缺失 |".format(name)); continue
            b = v["sem"]
            cells = [name, str(b["degraded"]), str(b["stable"]), str(b["clean_wrong"]), pc(b["luck"]["s_perm1"]), pc(b["luck"]["s_perm2"])]
            for m in SAMPLED:
                x = b["methods"][m]
                cells += ["{} / {}".format(pc(x["repair"]), pc(x["filler_repair"])), fmt(x["repair_minus_filler"]),
                          "NA" if x["explained"] is None else "{:.2f}".format(x["explained"]), fmt(x["harm_minus_filler"])]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
    if r["interface"]["pooled"]:
        L += ["## 修订 1：插入位置（观测块内 vs 块后）", "",
              "| 智能体 | D / 稳定 | 修复内容效应 块内 | 块后 | 伤害内容效应 块内 | 块后 | 伤害差（块内 − 块后） |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, v in list(r["interface"]["agents"].items()) + [("**合并**", r["interface"]["pooled"])]:
            if not v:
                L.append("| {} | 缺失 |".format(name)); continue
            b = v["sem"]
            L.append("| {} | {} / {} | {} | {} | {} | {} | {} |".format(
                name, b["degraded"], b["stable"], fmt(b["repair_content_inside"]), fmt(b["repair_content_after"]),
                fmt(b["harm_content_inside"]), fmt(b["harm_content_after"]), fmt(b["harm_content_inside_minus_after"])))
        L.append("")
    for part, title in (("luck", "修订 1：全部 5 种非原始顺序，稳定状态为抽样（按抽样比例加权，修订 2 的代码更正）"),
                        ("luck_full", "修订 2：全部 5 种非原始顺序，全部稳定状态")):
        if not (r.get(part) and r[part]["pooled"]):
            continue
        L += ["## " + title, "", "每格：观测修复 / 运气模型预测 / 差（观测 − 预测，联合区间） / 样本中“弄坏”事件数", "",
              "| 智能体 | D | 代表的稳定状态数 | " + " | ".join(ORDERS[1:]) + " | 四次都没修好 |",
              "| --- | --- | --- | " + " | ".join("---" for _ in ORDERS[1:]) + " | --- |"]
        for name, v in list(r[part]["agents"].items()) + [("**合并**", r[part]["pooled"])]:
            if not v:
                L.append("| {} | 缺失 |".format(name)); continue
            b = v["sem"]
            L.append("| {} | {} | {:.0f} | {} | {} |".format(name, b["degraded"], b["stable_represented"], " | ".join(
                "{} / {} / {} / {:.0f}".format(fmt(o["repair"]), pc(o["predicted_repair"]), fmt(o["gap"]), o["harm_events_in_sample"])
                for o in b["orders"].values()), pc(b["never_repaired"])))
        L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u1-root", type=Path, default=Path("outputs/u1"))
    p.add_argument("--labels", nargs="+", default=list(AGENTS))
    p.add_argument("--sampled-labels", nargs="*", default=list(SAMPLED_AGENTS))
    p.add_argument("--output", type=Path, required=True); p.add_argument("--resamples", type=int, default=2000)
    a = p.parse_args()
    try:
        if a.output.exists():
            raise ValueError("Output exists; use a new --output")
        report = analyze(a.u1_root, a.labels, a.sampled_labels, a.resamples)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    a.output.mkdir(parents=True)
    write_json(a.output / "as2.json", report)
    (a.output / "as2.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

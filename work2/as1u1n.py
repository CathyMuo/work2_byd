"""AS1 / U1N analysis (research/AS1_U1N_PROTOCOL.md): assay sensitivity of the replay audit and recovery on natural failures.

AS1 runs live in outputs/u1/<label>/as1 (H1 states, perm_1 observation, review / alert / hint each with its own matched
filler); U1N runs in outputs/u1/<label>/u1n (unperturbed observation, review / reflect with fillers, perm_1 as the luck
baseline). Readings A1 and N1-N3 are computed mechanically; the decision is written by hand into DECISION_LOG.md.
Question-cluster bootstrap stratified by dataset, all agents' states of a question kept together. CPU only.
"""
import argparse
from pathlib import Path

from .common import git_output, load_json, write_json
from .paper_h1 import agent_rows, fmt
from .paper_robust import boot

AGENTS = ("v03_3b_ppo", "v03_3b_grpo", "v01_7b_ppo", "v03_7b_ppo", "v03_7b_grpo", "v03_7b_it_grpo", "v03_14b_ppo", "v03_14b_it_grpo",
          "v03_32b_grpo", "zs_llama32_3b", "zs_qwen25_7b", "stepsearch_7b", "research_7b", "research_32b")
CONTENTS = {"review": ("perm1_review", "perm1_neutral"), "alert": ("perm1_alert", "perm1_neutral_alert"),
            "hint": ("perm1_hint", "perm1_neutral_hint")}
NATURAL = {"review": ("review", "neutral"), "reflect": ("reflect", "neutral_reflect")}
SENSITIVE_PP, SENSITIVE_AGENTS = 0.20, 12          # A1
RESCUE_PP, EXPLAINED = 0.03, 2 / 3                  # N1, N3
REPRO_MIN = 0.99                                    # A3
REPRO_ARMS = ("clean_a", "perm_1", "perm1_review", "perm1_neutral")


def rate(group, f):
    return sum(f(r) for r in group) / len(group) if group else None


def assay_block(rows, resamples, view="sem"):
    y = lambda r, a: r[view][a]
    degraded = [r for r in rows if r["y0"] == 1 and y(r, "perm_1") == 0]
    stable = [r for r in rows if r["y0"] == 1 and y(r, "perm_1") == 1]
    wrong = [r for r in rows if r["y0"] == 0]
    out = {"states": len(rows), "degraded": len(degraded), "stable": len(stable), "clean_wrong": len(wrong), "contents": {}}
    for name, (arm, filler) in CONTENTS.items():
        out["contents"][name] = {
            "repair": rate(degraded, lambda r, a=arm: y(r, a)), "filler_repair": rate(degraded, lambda r, f=filler: y(r, f)),
            "repair_minus_filler": boot(degraded, lambda r, a=arm, f=filler: y(r, a) - y(r, f), resamples),
            "harm_minus_filler": boot(stable, lambda r, a=arm, f=filler: y(r, f) - y(r, a), resamples),
            "rescue_minus_filler": boot(wrong, lambda r, a=arm, f=filler: y(r, a) - y(r, f), resamples)}
    return out


def sensitive(block):
    e = block["contents"]["hint"]["repair_minus_filler"]
    return bool(e and e["estimate"] >= SENSITIVE_PP and e["ci95"][0] > 0)


def reproduced(h1_dir, rows_dir):
    """Share of shared states whose continuations match the H1 run byte for byte, per arm (A3)."""
    from .datasets import read_rows
    old = {(r["sample_id"], r["target_turn"]): r for r in read_rows(h1_dir / "states.jsonl")}
    new = read_rows(rows_dir / "states.jsonl")
    out = {}
    for arm in REPRO_ARMS:
        pairs = [(old[k]["arms"][arm], r["arms"][arm]) for r in new for k in [(r["sample_id"], r["target_turn"])] if k in old]
        out[arm] = sum(a["continuation_sha256"] == b["continuation_sha256"] for a, b in pairs) / len(pairs) if pairs else None
    return out


def natural_block(rows, resamples, view="sem"):
    y = lambda r, a: r[view][a]
    wrong = [r for r in rows if r["y0"] == 0]
    right = [r for r in rows if r["y0"] == 1]
    out = {"states": len(rows), "clean_wrong": len(wrong), "clean_right": len(right),
           "perm1_rescue": rate(wrong, lambda r: y(r, "perm_1")), "methods": {}}
    for name, (arm, filler) in NATURAL.items():
        rescue, filler_rescue = rate(wrong, lambda r, a=arm: y(r, a)), rate(wrong, lambda r, f=filler: y(r, f))
        best = max(v for v in (filler_rescue, out["perm1_rescue"]) if v is not None) if wrong else None
        out["methods"][name] = {
            "rescue": rescue, "filler_rescue": filler_rescue,
            "explained": (best / rescue) if rescue else None,
            "rescue_minus_filler": boot(wrong, lambda r, a=arm, f=filler: y(r, a) - y(r, f), resamples),
            "rescue_minus_perm1": boot(wrong, lambda r, a=arm: y(r, a) - y(r, "perm_1"), resamples),
            "harm_minus_filler": boot(right, lambda r, a=arm, f=filler: y(r, f) - y(r, a), resamples),
            "all_minus_filler": boot(rows, lambda r, a=arm, f=filler: y(r, a) - y(r, f), resamples)}
    return out


def readings(r):
    a, n = r["assay"], r["natural"]
    agents = [k for k, v in a["agents"].items() if v]
    out = {"A1_pooled": sensitive(a["pooled"]["sem"]), "A1_agents": sum(sensitive(a["agents"][k]["sem"]) for k in agents),
           "A1_insensitive": [k for k in agents if not sensitive(a["agents"][k]["sem"])], "A1_agents_measured": len(agents)}
    out["A1"] = bool(out["A1_pooled"] and out["A1_agents"] >= SENSITIVE_AGENTS)
    out["A3_min"] = min((v for k in agents for v in a["reproduction"][k].values() if v is not None), default=None)
    out["A3"] = out["A3_min"] is not None and out["A3_min"] >= REPRO_MIN
    for name, m in n["pooled"]["sem"]["methods"].items():
        e, net = m["rescue_minus_filler"], m["all_minus_filler"]
        n1 = bool(e and e["ci95"][0] > 0 and e["estimate"] >= RESCUE_PP)
        n2 = bool(net and net["ci95"][0] > 0)
        n3 = bool(m["explained"] is not None and m["explained"] >= EXPLAINED)
        out[name] = {"N1": n1, "N2": n2, "N3": n3,
                     "reading": "same as H1" if n3 and not (n1 or n2) else "revise the paper" if (n1 or n2) else "partly consistent"}
    return out


def analyze(root, labels, resamples):
    report = {"analysis_commit": git_output("rev-parse", "HEAD"), "labels": list(labels),
              "bootstrap": {"unit": "question (all agents' states of a question together)", "strata": "dataset", "resamples": resamples}}
    for part, folder, block in (("assay", "as1", assay_block), ("natural", "u1n", natural_block)):
        per, pooled = {}, []
        for label in labels:
            d = root / label / folder
            if not (d / "states.jsonl").is_file():
                per[label] = None
                continue
            rows = agent_rows(d)
            pooled += rows
            per[label] = {v: block(rows, resamples, v) for v in ("sem", "raw")}
        report[part] = {"agents": per, "pooled": {v: block(pooled, resamples, v) for v in ("sem", "raw")} if pooled else None}
    if report["assay"]["pooled"] is None or report["natural"]["pooled"] is None:
        raise ValueError("Need at least one completed AS1 run and one completed U1N run under " + str(root))
    report["assay"]["reproduction"] = {k: reproduced(root / k / "pcr", root / k / "as1") for k, v in report["assay"]["agents"].items() if v}
    report["readings"] = readings(report)
    return report


def pc(v):
    return "NA" if v is None else "{:.1f}".format(100 * v)


def markdown(r):
    L = ["# AS1 / U1N 结果（自动生成，供回传）", "", "单位：百分点；[] 为按题聚类 bootstrap 95% CI。语义口径为主。", "",
         "## 预登记判定（机械值）", "", "```", str(r["readings"]), "```", "",
         "## AS1：退化状态上的修复（内容 − 其等长无意义文字），稳定状态上的额外伤害，clean 错状态上的额外救回", "",
         "| 智能体 | D | 稳定 | clean 错 | review 修复差 | alert 修复差 | hint 修复差 | hint 修复率 | review 伤害差 | alert 伤害差 | hint 伤害差 | alert 救回差 | hint 救回差 | 复现最低 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = list(r["assay"]["agents"].items()) + [("**合并**", r["assay"]["pooled"])]
    for name, v in rows:
        if not v:
            L.append("| {} | 缺失 |".format(name)); continue
        b, c = v["sem"], v["sem"]["contents"]
        rep = r["assay"]["reproduction"].get(name)
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, b["degraded"], b["stable"], b["clean_wrong"], fmt(c["review"]["repair_minus_filler"]), fmt(c["alert"]["repair_minus_filler"]),
            fmt(c["hint"]["repair_minus_filler"]), pc(c["hint"]["repair"]), fmt(c["review"]["harm_minus_filler"]),
            fmt(c["alert"]["harm_minus_filler"]), fmt(c["hint"]["harm_minus_filler"]), fmt(c["alert"]["rescue_minus_filler"]),
            fmt(c["hint"]["rescue_minus_filler"]), pc(min(rep.values())) if rep else ""))
    L += ["", "## U1N：自然失败（不换顺序）", "",
          "| 智能体 | clean 错 | clean 对 | perm_1 救回 | review 救回 / 文字 | review 救回差 | review 解释比例 | review 伤害差 | review 全状态 | reflect 救回 / 文字 | reflect 救回差 | reflect 解释比例 | reflect 伤害差 | reflect 全状态 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in list(r["natural"]["agents"].items()) + [("**合并**", r["natural"]["pooled"])]:
        if not v:
            L.append("| {} | 缺失 |".format(name)); continue
        b = v["sem"]
        cells = [name, str(b["clean_wrong"]), str(b["clean_right"]), pc(b["perm1_rescue"])]
        for m in ("review", "reflect"):
            x = b["methods"][m]
            cells += ["{} / {}".format(pc(x["rescue"]), pc(x["filler_rescue"])), fmt(x["rescue_minus_filler"]),
                      "NA" if x["explained"] is None else "{:.2f}".format(x["explained"]), fmt(x["harm_minus_filler"]), fmt(x["all_minus_filler"])]
        L.append("| " + " | ".join(cells) + " |")
    raw = r["natural"]["pooled"]["raw"]["methods"]
    L += ["", "原始 EM 口径（合并）：review 救回差 {}，全状态 {}；reflect 救回差 {}，全状态 {}。".format(
        fmt(raw["review"]["rescue_minus_filler"]), fmt(raw["review"]["all_minus_filler"]),
        fmt(raw["reflect"]["rescue_minus_filler"]), fmt(raw["reflect"]["all_minus_filler"])), ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u1-root", type=Path, default=Path("outputs/u1"))
    p.add_argument("--labels", nargs="+", default=list(AGENTS))
    p.add_argument("--output", type=Path, required=True); p.add_argument("--resamples", type=int, default=2000)
    a = p.parse_args()
    try:
        if a.output.exists():
            raise ValueError("Output exists; use a new --output")
        report = analyze(a.u1_root, a.labels, a.resamples)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    a.output.mkdir(parents=True)
    write_json(a.output / "as1_u1n.json", report)
    (a.output / "as1_u1n.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Paper table for H1: per-agent and pooled repair of recovery methods vs EACH same-state placebo.

The U1 verdict uses the lenient AC1 rule (best placebo / method >= 1/3). The paper
needs effect sizes: on degraded states D (clean correct, perm_1 wrong) the repair
rate of each method (perm1_review / perm1_reflect / perm1_budget2) minus the repair
rate of P1 (perm_2, re-perturbation) and of its token-matched P2 (neutral text),
with question-cluster bootstrap CIs. Pooled estimates keep a question's states from
ALL agents in one cluster (U1 agents share questions), which is the conservative
choice. Also reports two-sided flip noise (perm_1 rescue on clean-wrong states vs
perm_1 harm on clean-correct states) and the harm of each method on stable states.
CPU only; reads outputs/u1/<label>/pcr/states.jsonl of completed runs.
"""
import argparse
import json
from pathlib import Path

from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows
from .pcr_analysis import derive, estimate, mean

METHODS = {"review": "perm1_neutral", "reflect": "perm1_neutral_reflect", "budget2": "perm1_neutral"}


def agent_rows(run_dir):
    m = load_json(run_dir / "run_manifest.json")
    if m.get("status") != "completed" or file_hash(run_dir / "states.jsonl") != m["states_sha256"]:
        raise ValueError("Need a completed, unmodified PCR run: " + str(run_dir))
    return [derive(s) for s in read_rows(run_dir / "states.jsonl")]


def block(rows, view, resamples):
    y = lambda r, a: r[view][a]
    need = ("perm_1", "perm_2") + tuple("perm1_" + m for m in METHODS) + tuple(set(METHODS.values()))
    rows = [r for r in rows if all(a in r[view] for a in need)]
    degraded = [r for r in rows if r["y0"] == 1 and y(r, "perm_1") == 0]
    stable = [r for r in rows if r["y0"] == 1 and y(r, "perm_1") == 1]
    wrong = [r for r in rows if r["y0"] == 0]
    rate = lambda group, f: sum(f(r) for r in group) / len(group) if group else None
    out = {"states": len(rows), "degraded": len(degraded), "stable": len(stable), "clean_wrong": len(wrong),
           "noise": {"perm1_harm_on_clean_correct": rate([r for r in rows if r["y0"] == 1], lambda r: 1 - y(r, "perm_1")),
                     "perm1_rescue_on_clean_wrong": rate(wrong, lambda r: y(r, "perm_1"))},
           "P1_repair": estimate(degraded, mean(lambda r: y(r, "perm_2")), resamples) if degraded else None,
           "methods": {}}
    for m, p2 in METHODS.items():
        arm = "perm1_" + m
        if not degraded:
            continue
        out["methods"][m] = {
            "repair": estimate(degraded, mean(lambda r, a=arm: y(r, a)), resamples),
            "P2_repair": rate(degraded, lambda r, p=p2: y(r, p)),
            "minus_P1": estimate(degraded, mean(lambda r, a=arm: y(r, a) - y(r, "perm_2")), resamples),
            "minus_P2": estimate(degraded, mean(lambda r, a=arm, p=p2: y(r, a) - y(r, p)), resamples),
            "harm_on_stable": rate(stable, lambda r, a=arm: 1 - y(r, a)),
            "P1_harm_on_stable": rate(stable, lambda r: 1 - y(r, "perm_2")),
            # Matched harm control (2026-10-27): the same content-free text on the same stable state.
            "P2_harm_on_stable": rate(stable, lambda r, p=p2: 1 - y(r, p)),
            "harm_minus_P2": estimate(stable, mean(lambda r, a=arm, p=p2: y(r, p) - y(r, a)), resamples) if stable else None}
    return out


def analyze(root, labels, resamples):
    per, pooled = {}, []
    for label in labels:
        d = root / label / "pcr"
        if not (d / "states.jsonl").is_file():
            per[label] = None
            continue
        rows = agent_rows(d)
        pooled += rows
        per[label] = {v: block(rows, v, resamples) for v in ("sem", "raw")}
    if not pooled:
        raise ValueError("No completed U1 PCR runs under " + str(root))
    by_dataset = {}
    for ds in sorted({r["dataset"] for r in pooled}):
        by_dataset[ds] = block([r for r in pooled if r["dataset"] == ds], "sem", resamples)
    return {"analysis_commit": git_output("rev-parse", "HEAD"), "agents": per,
            "pooled": {v: block(pooled, v, resamples) for v in ("sem", "raw")}, "pooled_by_dataset": by_dataset,
            "bootstrap": {"unit": "question (all agents' states of a question together)", "strata": "dataset",
                          "resamples": resamples, "seed": 42},
            "view": "sem = format-flip corrected (primary, as in the U1 verdicts); raw = plain EM"}


def fmt(e, pct=True):
    if not isinstance(e, dict) or e.get("estimate") is None:
        return "NA"
    k = 100 if pct else 1
    ci = e.get("ci95")
    return "{:.1f} [{}]".format(e["estimate"] * k, "{:.1f}, {:.1f}".format(ci[0] * k, ci[1] * k) if ci else "no CI")


def pc(v):
    return "NA" if v is None else "{:.1f}".format(100 * v)


def table(rows):
    L = ["| 智能体 | D | P1 修复 | review 修复 | review−P1 | review−P2 | reflect 修复 | reflect−P1 | reflect−P2 | budget2−P1 | review 伤害 / P1 伤害 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, b in rows:
        if b is None:
            L.append("| {} | 缺失 | | | | | | | | | |".format(name)); continue
        m = b["methods"]
        if not m:
            L.append("| {} | 0 | | | | | | | | | |".format(name)); continue
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} / {} |".format(
            name, b["degraded"], fmt(b["P1_repair"]), fmt(m["review"]["repair"]), fmt(m["review"]["minus_P1"]),
            fmt(m["review"]["minus_P2"]), fmt(m["reflect"]["repair"]), fmt(m["reflect"]["minus_P1"]),
            fmt(m["reflect"]["minus_P2"]), fmt(m["budget2"]["minus_P1"]),
            pc(m["review"]["harm_on_stable"]), pc(m["review"]["P1_harm_on_stable"])))
    return L


def harm_table(rows):
    """Harm on stable states (clean right, perm_1 right): each method vs its token-matched P2 on the same state."""
    L = ["| 智能体 | 稳定态数 | P1 伤害 | review 伤害 | P2 伤害 | review−P2 | reflect 伤害 | P2 伤害 | reflect−P2 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, b in rows:
        if not b or not b["methods"]:
            L.append("| {} | 缺失 | | | | | | | |".format(name)); continue
        rv, rf = b["methods"]["review"], b["methods"]["reflect"]
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, b["stable"], pc(rv["P1_harm_on_stable"]), pc(rv["harm_on_stable"]), pc(rv.get("P2_harm_on_stable")),
            fmt(rv.get("harm_minus_P2")), pc(rf["harm_on_stable"]), pc(rf.get("P2_harm_on_stable")), fmt(rf.get("harm_minus_P2"))))
    return L


def markdown(r):
    L = ["# H1 论文表：恢复方法 vs 各同状态安慰剂（自动生成，供回传）", "",
         "单位：百分点；[] 为按题聚类 bootstrap 95% CI（合并行把同一道题在所有智能体上的状态放在一个簇里）。",
         "D = 退化状态数（clean 对、perm_1 错）。P1 = 再换一次顺序（perm_2）；P2 = 与方法等长的中性文字。", "",
         "## 语义口径（主，格式翻转已校正）", ""]
    L += table([(k, v and v["sem"]) for k, v in r["agents"].items()] + [("**合并**", r["pooled"]["sem"])])
    L += ["", "## 原始 EM 口径", ""]
    L += table([(k, v and v["raw"]) for k, v in r["agents"].items()] + [("**合并**", r["pooled"]["raw"])])
    L += ["", "## 稳定状态上的伤害（语义口径；方法 vs 同状态、等长的 P2）", "",
          "“方法−P2”= 方法比 P2 多造成的伤害（正值表示方法更伤）。", ""]
    L += harm_table([(k, v and v["sem"]) for k, v in r["agents"].items()] + [("**合并**", r["pooled"]["sem"])])
    L += ["", "## 合并、分数据集（语义口径）", ""]
    L += table(list(r["pooled_by_dataset"].items()))
    n = r["pooled"]["sem"]["noise"]
    L += ["", "## 双向翻转噪声（合并，语义口径）", "",
          "perm_1 在 clean 正确状态上的伤害率 {}%；在 clean 错误状态上的救回率 {}%。".format(
              pc(n["perm1_harm_on_clean_correct"]), pc(n["perm1_rescue_on_clean_wrong"])), ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u1-root", type=Path, default=Path("outputs/u1"))
    p.add_argument("--agents", type=Path, nargs="+", default=[Path("configs/u1_agents.json")],
                   help="one or more roster files (e.g. configs/u1_agents.json configs/u1x_agents.json)")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--exclude", nargs="*", default=[], help="labels left out of the table and the pooled rows (reported in the output)")
    a = p.parse_args()
    try:
        if a.output.exists():
            raise ValueError("Output exists; use a new --output")
        labels = [x["label"] for f in a.agents for x in load_json(f)["agents"] if x["label"] not in a.exclude]
        report = dict(analyze(a.u1_root, labels, a.resamples), excluded_agents=a.exclude)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    a.output.mkdir(parents=True)
    write_json(a.output / "h1_pooled.json", report)
    (a.output / "h1_pooled.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

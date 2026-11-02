"""Exploratory robustness of the pooled H1 numbers (U1L protocol section 8; written after the pooled table was seen).

No new inference and no new arms: the states of paper_h1, cut by model size and recipe family, per agent and
leave-one-agent-out. Adds the unselected comparison over ALL states (method vs filler P2, method vs re-perturbation P1),
which needs no failure selection, and the method effect on clean-wrong states, which the D/stable split leaves out.
Also puts the E0-mismatch states back. paper_h1 stays the table of record; this one only says how far its pooled rows
depend on who is in the pool. Question-cluster bootstrap, stratified by dataset, on per-question sums so the many cuts stay cheap. CPU only.
"""
import argparse
import random
import re
from pathlib import Path

from .common import git_output, load_json, write_json
from .datasets import read_rows
from .paper_h1 import agent_rows, fmt
from .pcr_analysis import derive

METHODS = {"review": "perm1_neutral", "reflect": "perm1_neutral_reflect"}
NEED = ("perm_1", "perm_2", "perm1_review", "perm1_neutral", "perm1_reflect", "perm1_neutral_reflect")
COLUMNS = (("repair_minus_P2", "修复−P2"), ("repair_minus_P1", "修复−P1"), ("harm_minus_P2", "伤害−P2"),
           ("rescue_minus_P2", "clean 错状态−P2"), ("all_minus_P2", "全状态−P2"), ("all_minus_P1", "全状态−P1"))


def boot(rows, value, resamples, seed=42):
    """Mean of value(r); all of a question's states (across agents) are resampled together, within its dataset."""
    if not rows:
        return None
    clusters = {}
    for r in rows:
        c = clusters.setdefault((r["dataset"], r["sample_id"]), [0.0, 0])
        c[0] += value(r); c[1] += 1
    groups = [[v for k, v in clusters.items() if k[0] == ds] for ds in sorted({k[0] for k in clusters})]
    rng, draws = random.Random(seed), []
    for _ in range(resamples):
        total = count = 0
        for g in groups:
            for t, c in rng.choices(g, k=len(g)):
                total += t; count += c
        draws.append(total / count)
    draws.sort()
    return {"n_states": len(rows), "n_questions": len(clusters), "estimate": sum(v[0] for v in clusters.values()) / len(rows),
            "ci95": [draws[int((resamples - 1) * .025)], draws[int((resamples - 1) * .975)]]}


def effects(rows, resamples, view="sem"):
    y = lambda r, a: r[view][a]
    rows = [r for r in rows if all(a in r[view] for a in NEED)]
    degraded = [r for r in rows if r["y0"] == 1 and y(r, "perm_1") == 0]
    stable = [r for r in rows if r["y0"] == 1 and y(r, "perm_1") == 1]
    wrong = [r for r in rows if r["y0"] == 0]
    out = {"states": len(rows), "degraded": len(degraded), "stable": len(stable), "clean_wrong": len(wrong), "methods": {}}
    for m, p2 in METHODS.items():
        arm = "perm1_" + m
        minus_p2 = lambda r, a=arm, p=p2: y(r, a) - y(r, p)
        minus_p1 = lambda r, a=arm: y(r, a) - y(r, "perm_2")
        out["methods"][m] = {"repair_minus_P2": boot(degraded, minus_p2, resamples),
                             "repair_minus_P1": boot(degraded, minus_p1, resamples),
                             "harm_minus_P2": boot(stable, lambda r, a=arm, p=p2: y(r, p) - y(r, a), resamples),
                             "rescue_minus_P2": boot(wrong, minus_p2, resamples),
                             "all_minus_P2": boot(rows, minus_p2, resamples),
                             "all_minus_P1": boot(rows, minus_p1, resamples)}
    return out


def size_of(label):
    return int(re.search(r"(\d+)b", label).group(1))


def family_of(label):
    return "Search-R1" if label.startswith("v0") else "ReSearch" if label.startswith("research") else "other"


def run_rows(run_dir):
    """Usable rows (receipts verified by paper_h1) and the E0-mismatch rows the analyses exclude."""
    back = [dict(derive(s), controls_agree=s["control_mismatch"]["controls_agree"])
            for s in read_rows(run_dir / "states.jsonl") if not s.get("controls_exact", True)]
    return agent_rows(run_dir), back


def analyze(root, labels, resamples):
    usable, excluded, missing = {}, {}, []
    for label in labels:
        d = root / label / "pcr"
        if not (d / "states.jsonl").is_file():
            missing.append(label)
            continue
        usable[label], excluded[label] = run_rows(d)
    if not usable:
        raise ValueError("No completed PCR runs under " + str(root))
    pick = lambda keep: [r for k, rows in usable.items() if keep(k) for r in rows]
    everything = pick(lambda k: True)
    sizes = ("<=7B", 14, 32)
    size_rows = {("<=7B" if s == "<=7B" else "{}B".format(s)):
                 pick(lambda k, s=s: size_of(k) <= 7 if s == "<=7B" else size_of(k) == s) for s in sizes}
    families = sorted({family_of(k) for k in usable})
    report = {"analysis_commit": git_output("rev-parse", "HEAD"), "agents_used": list(usable), "missing": missing,
              "bootstrap": {"unit": "question (all agents' states of a question together)", "strata": "dataset",
                            "resamples": resamples, "seed": 42},
              "pooled": {"sem": effects(everything, resamples), "raw": effects(everything, resamples, "raw")},
              "by_size": {k: effects(v, resamples) for k, v in size_rows.items() if v},
              "by_family": {f: effects(pick(lambda k, f=f: family_of(k) == f), resamples) for f in families},
              "agents": {k: effects(v, resamples) for k, v in usable.items()},
              "leave_one_out": {x: effects(pick(lambda k, x=x: k != x), resamples) for x in usable}}
    y = lambda r, a: r["sem"][a]
    per = {k: {"states": len(v), "degraded": sum(r["y0"] == 1 and y(r, "perm_1") == 0 for r in v),
               "stable": sum(r["y0"] == 1 and y(r, "perm_1") == 1 for r in v), "clean_wrong": sum(r["y0"] == 0 for r in v),
               "controls_agree": sum(bool(r["controls_agree"]) for r in v)} for k, v in excluded.items() if v}
    back = [r for v in excluded.values() for r in v]
    report["mismatch"] = {"per_agent": per, "pooled_put_back": effects(everything + back, resamples),
                          "agents_put_back": {k: effects(usable[k] + excluded[k], resamples) for k in per}}
    return report


def table(rows):
    head = ["分组", "状态", "D", "稳定", "clean 错"]
    for m in METHODS:
        head += ["{} {}".format(m, name) for _, name in COLUMNS]
    L = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for name, b in rows:
        cells = [name, str(b["states"]), str(b["degraded"]), str(b["stable"]), str(b["clean_wrong"])]
        cells += [fmt(b["methods"][m][key]) for m in METHODS for key, _ in COLUMNS]
        L.append("| " + " | ".join(cells) + " |")
    return L


def markdown(r):
    L = ["# H1 稳健性（探索性，U1L 协议第 8 节；自动生成，供回传）", "",
         "所用智能体：{}".format(", ".join(r["agents_used"])), "没有结果的智能体：{}".format(", ".join(r["missing"]) or "无"), "",
         "单位：百分点；[] 为按题聚类 bootstrap 95% CI（{} 次，按数据集分层）。语义口径，另有注明除外。".format(r["bootstrap"]["resamples"]),
         "修复−P2 / 修复−P1 在退化状态 D 上；伤害−P2 在稳定状态上（正值 = 方法更伤）；clean 错状态−P2 在 clean 答错的状态上；",
         "全状态−P2 / 全状态−P1 在全部状态上（不做任何失败筛选，正值 = 方法答对更多）。", "",
         "## 合并", ""]
    L += table([("语义", r["pooled"]["sem"]), ("原始 EM", r["pooled"]["raw"])])
    L += ["", "## 按模型大小", ""] + table(list(r["by_size"].items()))
    L += ["", "## 按配方家族", ""] + table(list(r["by_family"].items()))
    L += ["", "## 每个智能体", ""] + table(list(r["agents"].items()))
    L += ["", "## 去掉一个智能体", ""] + table([("去掉 " + k, v) for k, v in r["leave_one_out"].items()])
    L += ["", "## E0 不复现的状态放回去", "", "| 智能体 | 状态 | 其中 D | 稳定 | clean 错 | 两次 E0 彼此一致 |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in r["mismatch"]["per_agent"].items():
        L.append("| {} | {} | {} | {} | {} | {} |".format(k, v["states"], v["degraded"], v["stable"], v["clean_wrong"], v["controls_agree"]))
    L += [""] + table([("合并（放回）", r["mismatch"]["pooled_put_back"])] +
                      [(k + "（放回）", v) for k, v in r["mismatch"]["agents_put_back"].items()]) + [""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u1-root", type=Path, default=Path("outputs/u1"))
    p.add_argument("--agents", type=Path, nargs="+", required=True, help="roster files")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--exclude", nargs="*", default=[], help="labels left out (reported in the output)")
    p.add_argument("--validity", type=Path, help="arm_diagnose JSON; agents with insertion_valid false are excluded too")
    a = p.parse_args()
    try:
        if a.output.exists():
            raise ValueError("Output exists; use a new --output")
        exclude = list(a.exclude)
        if a.validity:
            exclude += [k for k, v in load_json(a.validity).items() if v.get("insertion_valid") is False and k not in exclude]
        labels = [x["label"] for f in a.agents for x in load_json(f)["agents"] if x["label"] not in exclude]
        report = dict(analyze(a.u1_root, labels, a.resamples), excluded_agents=exclude)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    a.output.mkdir(parents=True)
    write_json(a.output / "h1_robust.json", report)
    (a.output / "h1_robust.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

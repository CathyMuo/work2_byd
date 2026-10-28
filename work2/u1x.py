"""U1x fidelity: greedy EM of each added agent vs the EM its own paper reports (E5 + 2018 Wikipedia, top-3).

References were read from the papers before any U1x run (research/U1X_PROTOCOL.md section 3). Rule:
macro-average gap over datasets that have a reference; gap < -10 pp marks the agent "possibly mis-harnessed"
(reported separately, not counted toward the generality claim). Format/length failures >= 10% exclude the
agent, as in U1. ZeroSearch's paper evaluates with Google search, so it has no comparable reference.
"""
import argparse
import json
from pathlib import Path

from .common import load_json, write_json
from .datasets import read_rows

REFERENCE_EM = {
    # AutoRefine (arXiv 2505.11277) Table 1, Qwen2.5-7B-Base, AutoRefine row.
    "autorefine_7b": {"nq": 0.484, "hotpotqa": 0.451, "2wiki": 0.405, "musique": 0.187, "bamboogle": 0.512},
    # StepSearch (arXiv 2505.15107) main table, Qwen2.5-7b-Base, StepSearch-base row.
    "stepsearch_7b": {"hotpotqa": 0.380, "2wiki": 0.385, "musique": 0.216, "bamboogle": 0.467},
    # ReSearch (arXiv 2503.19470) Table 2, EM, E5 + 2018 Wikipedia, top-5 (U1L, read before any run).
    "research_32b": {"hotpotqa": 0.4673, "2wiki": 0.4490, "musique": 0.2640, "bamboogle": 0.5680},
    "research_7b": {"hotpotqa": 0.4352, "2wiki": 0.4759, "musique": 0.2230, "bamboogle": 0.4240},
}
FAIL = ("malformed_action", "generation_limit", "context_limit")


def agent_fidelity(root, label):
    path = Path(root) / label / "sources" / "sources.jsonl"
    if not path.is_file():
        return None
    rows = read_rows(path)
    em = {}
    for ds in sorted({r["dataset"] for r in rows}):
        xs = [r["em"] for r in rows if r["dataset"] == ds]
        em[ds] = sum(xs) / len(xs)
    ref = REFERENCE_EM.get(label, {})
    gaps = {ds: em[ds] - v for ds, v in ref.items() if ds in em}
    gap = sum(gaps.values()) / len(gaps) if gaps else None
    fail = sum(r["status"] in FAIL for r in rows) / len(rows)
    return {"questions": len(rows), "em": em, "em_all": sum(r["em"] for r in rows) / len(rows), "format_fail_rate": fail,
            "reference_em": ref or None, "gap_by_dataset": gaps or None, "macro_gap": gap,
            "excluded_for_format_failure": fail >= .10,
            "possibly_misharnessed": bool(gap is not None and gap < -.10)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/u1")); p.add_argument("--roster", type=Path, default=Path("configs/u1x_agents.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = {x["label"]: agent_fidelity(a.root, x["label"]) for x in load_json(a.roster)["agents"]}
    write_json(a.output, out)
    L = ["# U1x 保真度（自动生成，供回传）", "", "| agent | 题数 | EM | 格式失败 | 论文 EM 宏平均差 | 排除 | 可能 harness 不符 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    f = lambda x: "NA" if x is None else "{:.3f}".format(x)
    for k, v in out.items():
        if v is None:
            L.append("| {} | 未运行 | | | | | |".format(k)); continue
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(k, v["questions"], f(v["em_all"]), f(v["format_fail_rate"]), f(v["macro_gap"]),
                                                                v["excluded_for_format_failure"], v["possibly_misharnessed"]))
    L += ["", "分数据集 EM：", "```json", json.dumps({k: v and {"em": v["em"], "gap": v["gap_by_dataset"]} for k, v in out.items()}, indent=1), "```", ""]
    a.output.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

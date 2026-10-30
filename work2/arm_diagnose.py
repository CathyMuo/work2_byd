"""What does an agent generate right after a PCR arm's change? CPU only, reads a completed PCR run.

Per arm: terminal status counts, answered rate, EM, mean searches after the target, mean generated
tokens, and the first characters of the first post-target generation for a few stable states
(clean right, perm_1 right) - the states where any failure is harm caused by the arm itself.
Used for U1x (2026-10-28) to tell "the method hurts" from "inserting any text breaks the agent".
"""
import argparse
import collections
import json
from pathlib import Path

from .datasets import read_rows
from .placebo import usable_states

# Insertion validity (fixed 2026-10-28 before looking at any agent other than AutoRefine and v0.3 7B PPO):
# with content-free text inserted (P2), the agent must still produce a well-formed answer on >= half of its stable states.
VALIDITY_MIN_ANSWERED = 0.5
P2_ARMS = ("perm1_neutral", "perm1_neutral_reflect")
ARMS = ("perm_2", "perm1_review", "perm1_neutral", "perm1_reflect", "perm1_neutral_reflect", "perm1_budget2")


def diagnose(run_dir, examples=4, chars=300):
    states = usable_states(read_rows(Path(run_dir) / "states.jsonl"))
    stable = [s for s in states if s["arms"]["clean_a"]["em"] == 1 and s["arms"]["perm_1"] and s["arms"]["perm_1"]["em"] == 1]
    out = {"states": len(states), "stable_states": len(stable), "arms": {}}
    for arm in ARMS:
        rows = [s["arms"][arm] for s in stable if s["arms"].get(arm)]
        if not rows:
            continue
        n = len(rows)
        out["arms"][arm] = {
            "stable_n": n, "statuses": dict(collections.Counter(r["status"] for r in rows)),
            "answered": sum(r["status"] == "answered" for r in rows) / n, "em": sum(r["em"] or 0 for r in rows) / n,
            "searches_after": sum(r.get("search_calls_after_target", 0) for r in rows) / n,
            "tokens_after": sum(r.get("generated_tokens_after_target", 0) for r in rows) / n,
            "examples": [{"sample_id": s["sample_id"], "status": s["arms"][arm]["status"],
                          "final_answer": s["arms"][arm]["final_answer"], "gold": s["gold_answer"],
                          "first_generation": (s["arms"][arm]["post_target_texts"] or [""])[0][:chars]}
                         for s in stable[:examples] if s["arms"].get(arm)]}
    rates = [out["arms"][a]["answered"] for a in P2_ARMS if a in out["arms"]]
    out["insertion_valid"] = bool(rates) and min(rates) >= VALIDITY_MIN_ANSWERED
    return out


def validity_table(report):
    L = ["## 插入有效性（P2 无内容文字下，稳定状态的正常作答率 ≥ {:.0%} 才有效）".format(VALIDITY_MIN_ANSWERED), "",
         "| 智能体 | P2（对应复查）作答率 | P2（对应反思）作答率 | 有效 |", "| --- | --- | --- | --- |"]
    for label, d in report.items():
        r = [d["arms"].get(a, {}).get("answered") for a in P2_ARMS]
        L.append("| {} | {} | {} | {} |".format(label, *["NA" if x is None else "{:.3f}".format(x) for x in r], d["insertion_valid"]))
    return "\n".join(L) + "\n"


def markdown(label, d):
    L = ["## {}（{} 个状态，其中稳定状态 {} 个）".format(label, d["states"], d["stable_states"]), "",
         "| 臂 | 稳定态 n | 正常作答率 | EM | 目标后检索次数 | 目标后生成 token | 终止状态 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm, a in d["arms"].items():
        L.append("| {} | {} | {:.3f} | {:.3f} | {:.2f} | {:.0f} | {} |".format(arm, a["stable_n"], a["answered"], a["em"], a["searches_after"],
                                                                       a["tokens_after"], json.dumps(a["statuses"], ensure_ascii=False)))
    for arm in ("perm1_neutral", "perm1_review"):
        if arm in d["arms"]:
            L += ["", "**{} 的前几个例子（目标之后的第一段生成）**".format(arm), ""]
            for e in d["arms"][arm]["examples"]:
                L += ["- `{}` 状态 {}，答案 {!r}，标准答案 {!r}：".format(e["sample_id"], e["status"], e["final_answer"], e["gold"]),
                      "  ```", "  " + e["first_generation"].replace("\n", "\n  "), "  ```"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/u1")); p.add_argument("--labels", nargs="*", default=[])
    p.add_argument("--agents", type=Path, nargs="*", default=[], help="roster files; every label with a PCR run is diagnosed")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    labels = list(a.labels) + [x["label"] for f in a.agents for x in json.loads(Path(f).read_text(encoding="utf-8"))["agents"]
                               if x["label"] not in a.labels and (a.root / x["label"] / "pcr" / "states.jsonl").is_file()]
    if not labels:
        p.error("no labels")
    report = {label: diagnose(a.root / label / "pcr") for label in labels}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    text = "# PCR 插入臂诊断（自动生成，供回传）\n\n" + validity_table(report) + "\n" + "\n".join(markdown(k, v) for k, v in report.items())
    a.output.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

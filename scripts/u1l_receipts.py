#!/usr/bin/env python3
"""U1L receipts for the paper appendix and the decision log: what each run froze, when it ran, and every E0 mismatch verbatim. CPU only."""
import datetime
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/u1")
roster = json.load(open("configs/u1l_agents.json", encoding="utf-8"))["agents"]
FIRST_FAILURE = "2wiki:f643f9180bd911eba7f7acde48001122"


def stamp(p):
    return datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M") if p.exists() else "NA"


print("## 各运行冻结的配置和时间\n")
print("| 智能体 | 运行 | 提交 | 工作区有改动 | GPU | 显存比例 | E0 模式 | 开始 | 结束 |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for a in roster:
    for kind in ("sources", "pcr"):
        d = root / a["label"] / kind
        plan = json.load(open(d / "run_plan.json", encoding="utf-8")) if (d / "run_plan.json").exists() else {}
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            a["label"], kind, str(plan.get("repository_commit", "NA"))[:7], plan.get("working_tree_dirty", "NA"),
            ",".join(plan.get("gpus", [])), plan.get("gpu_memory_utilization", "NA"), plan.get("control_mismatch", "NA"),
            stamp(d / "run_plan.json"), stamp(d / ("states.jsonl" if kind == "pcr" else "sources.jsonl"))))

print("\n## research_7b 第一次运行\n")
first = root / "research_7b" / "pcr_attempt1_e0stop"
for f in sorted(first.glob("workers/*/run_manifest.json")):
    print("{}  结束 {}  {}".format(f.parent.name, stamp(f), {k: v for k, v in json.load(open(f, encoding="utf-8")).items() if k in ("status", "recorded_states", "expected_states")}))

print("\n## E0 不复现的状态（原文）\n")
for a in roster:
    states = root / a["label"] / "pcr" / "states.jsonl"
    if not states.exists():
        continue
    for line in open(states, encoding="utf-8"):
        s = json.loads(line)
        if not s.get("controls_exact", True):
            print("{} {} t{}{}  clean EM={} perm_1 EM={}\n{}\n".format(
                a["label"], s["sample_id"], s["target_turn"], "  <-- 第一次运行停下的状态" if s["sample_id"] == FIRST_FAILURE else "",
                s["arms"]["clean_a"]["em"], s["arms"]["perm_1"]["em"], json.dumps(s["control_mismatch"], ensure_ascii=False)))

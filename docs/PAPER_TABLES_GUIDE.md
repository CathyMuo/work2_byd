# 论文表格：厂内执行指导（只用 CPU，约 10 分钟）

更新：2026-10-27　分支：`feat/credit-testbed`
背景：`research/U1_U2_HR2_RESULTS_AND_DECISION.md`

目的：
1. 把 U1 的 H1 结果变成论文主表：每种安慰剂各自的差值和 CI，以及 7 个智能体的合并结果。
2. 补齐 U2 回传里缺的数字。

**不用 GPU，不跑模型**，只读已有产物。

## 0. 规则

- 不改代码；不修改 `research/DECISION_LOG.md` 和 `research/STATE.json`。
- 只新增 `outputs/paper/`，不改动 `outputs/u1`、`outputs/u2`、`outputs/hr2_*`。

## 1. 拉代码、测试

```bash
cd /root/develop/workspace/work2_byd
git status --short && git fetch origin && git switch feat/credit-testbed && git pull --ff-only
export PY=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
"$PY" -m unittest discover -s tests -q
```

如果厂内有本地提交（例如 `0f74626` 之后的 messages.md），`pull --ff-only` 会失败。这时先 `git log --oneline -3` 看清楚，再用 `git rebase origin/feat/credit-testbed` 把本地提交接到最新代码后面。只要求代码与公开分支一致；本地提交不必推送。

## 2. H1 合并表

```bash
"$PY" -m work2.paper_h1 --u1-root outputs/u1 --output outputs/paper/h1_pooled
```

会打印并保存 `outputs/paper/h1_pooled/h1_pooled.md`，包含：
- 每个智能体和合并后的结果，分语义口径（主）和原始 EM 口径；
- 合并后分数据集的结果；
- 双向翻转噪声。

用时约几分钟（bootstrap 2000 次）。

## 3. 回传清单

1. `git rev-parse HEAD`。
2. `outputs/paper/h1_pooled/h1_pooled.md` **全文**。
3. 下面三个文件的**全文**（上次只回传了摘要，缺 CI 和 MR-Search 在 S_wrong 上的修复率）：
   - `outputs/u2/reseek_reseek_3b/analysis/u2_report.md`
   - `outputs/u2/mr_v03_7b_ppo/analysis/u2_report.md`
   - `outputs/u2/mr_v03_3b_ppo/analysis/u2_report.md`
4. HR2 GRPO 的源轨迹状态分布：

```bash
"$PY" - <<'PY'
import json, collections
for r in ("ppo", "grpo"):
    c = collections.Counter()
    for line in open("outputs/hr2_%s/records.jsonl" % r):
        c[json.loads(line)["trajectories"]["R"]["status"]] += 1
    print(r, dict(c))
PY
```

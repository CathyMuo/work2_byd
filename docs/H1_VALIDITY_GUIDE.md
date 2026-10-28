# H1 插入有效性与主表：厂内执行指导（只用 CPU，约 5 分钟）

更新：2026-10-28　分支：`feat/credit-testbed`
背景：`research/U1X_RESULTS_AND_DECISION.md` §5

要做三件事：
1. 对 U1 + U1x 全部 11 个智能体做插入有效性诊断。判据：在无内容文字 P2 下，稳定状态的正常作答率 ≥ 50%。
2. 按这个判据生成 H1 主表。
3. 生成两张敏感性表：一张排除 AutoRefine 和 14B GRPO（9 个智能体）；另一张是之前的全部 11 个和排除 AutoRefine 的 10 个，这两张已经有了。

**不用 GPU，不跑模型**，只读已有产物。

## 0. 规则

- 当前没有运行中的实验，可以拉取代码。不改代码，不提交，不修改 `research/DECISION_LOG.md` 和 `research/STATE.json`。
- 只新增 `outputs/paper/` 下的文件。

## 1. 拉代码、测试

```bash
cd /root/develop/workspace/work2_byd
git status --short && git fetch origin && git log --oneline -3
git pull --ff-only      # 有厂内本地提交时，改用 git rebase origin/feat/credit-testbed
export PY=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
"$PY" -m unittest discover -s tests -q
```

## 2. 全部智能体的插入有效性诊断

```bash
"$PY" -m work2.arm_diagnose --agents configs/u1_agents.json configs/u1x_agents.json --output outputs/paper/arm_diagnose_all.json
```

输出开头是一张“插入有效性”表（每个智能体一行），后面是每个智能体的分臂明细。

## 3. 主表（按判据排除）与 9 个智能体的敏感性表

```bash
"$PY" -m work2.paper_h1 --u1-root outputs/u1 --agents configs/u1_agents.json configs/u1x_agents.json \
  --validity outputs/paper/arm_diagnose_all.json --output outputs/paper/h1_pooled_valid
"$PY" -m work2.paper_h1 --u1-root outputs/u1 --agents configs/u1_agents.json configs/u1x_agents.json \
  --exclude autorefine_7b v03_14b_grpo --output outputs/paper/h1_pooled_9
```

如果第一条命令排除的正好是 AutoRefine 和 14B 两个，两张表会完全相同，这也正常，照常回传。

## 4. 回传清单

1. `git rev-parse HEAD`。
2. `outputs/paper/arm_diagnose_all.md` 里的**“插入有效性”表**，以及 `v03_14b_grpo`、`v03_3b_grpo` 两节的分臂明细。
3. `outputs/paper/h1_pooled_valid/h1_pooled.md` **全文**，开头会写明排除了哪些智能体。
4. `outputs/paper/h1_pooled_9/h1_pooled.md` 中的三节：语义口径合并行、“稳定状态上的伤害”、“智能体层面汇总”。

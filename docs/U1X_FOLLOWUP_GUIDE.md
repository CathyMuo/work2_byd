# U1x 后续统计：厂内执行指导（只用 CPU，约 5 分钟）

更新：2026-10-28　分支：`feat/credit-testbed`
背景：`research/U1X_RESULTS_AND_DECISION.md`

要做两件事：
1. 生成**排除 AutoRefine 的 10 个智能体**合并表（论文主表）。
2. 看清 AutoRefine 在“插入文字”之后到底生成了什么。

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

## 2. 10 个智能体合并表

```bash
"$PY" -m work2.paper_h1 --u1-root outputs/u1 --agents configs/u1_agents.json configs/u1x_agents.json \
  --exclude autorefine_7b --output outputs/paper/h1_pooled_10
```

## 3. 插入臂诊断（AutoRefine 对照 v0.3 7B PPO）

```bash
"$PY" -m work2.arm_diagnose --labels autorefine_7b v03_7b_ppo --output outputs/paper/arm_diagnose.json
```

## 4. 回传清单

1. `git rev-parse HEAD`。
2. `outputs/paper/h1_pooled_10/h1_pooled.md` **全文**，包括“稳定状态上的伤害”一节。
3. `outputs/paper/arm_diagnose.md` **全文**。
4. 如果手边有 `outputs/paper/h1_pooled_u1x/h1_pooled.md` 的“稳定状态上的伤害”一节（11 个智能体），也一起贴上。

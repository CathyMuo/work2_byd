# AC1 厂内执行指导：干预式审计的确认实验（新题 + 第二检查点）

更新：2026-10-19　分支：`feat/credit-testbed`
协议（**先读第 1–4 节**）：[research/AUDIT_CONFIRM_PROTOCOL.md](../research/AUDIT_CONFIRM_PROTOCOL.md)
背景：CT1 v1 已判 NO-GO。用户决定把 PCR 与 CT1 中站得住的三条发现，放到**从未用过的新题**和**第二个检查点**上做确认。
本轮**只做推理**，不训练。下面两部分可以复用旧引擎：PCR（20 臂）和 CT1（k=16）。

---

## 0. 规则与先要回报的事

1. 沿用 searchr1 环境、E5 + Wiki-18 Flat 索引和常驻 Retriever。不重建索引，不改代码，不覆盖旧 `outputs/`；本轮产物都放在 `outputs/ac1_*`。
2. **请先回报**：
   - `git log --oneline c39af3c..e279f63` 和 `git diff --stat c39af3c e279f63`：上一轮的 HEAD `e279f63` 不在公开仓库，测试数又是 170 项，与公开的 141 项不一致；
   - 厂内**离线可用的 Search-R1 检查点**清单（路径 + 名字），重点看有没有 `SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo`、`...-7b-em-grpo`、`...-3b-em-grpo`。
3. 第二检查点的选择顺序：3B PPO → 7B GRPO → 3B GRPO，取第一个能离线加载的。**请在跑任何第二检查点结果之前回报选了哪个。**都没有的话，只跑主检查点并说明。
4. 不要自己下 GO/NO-GO 结论。

## 1. 拉代码、测试

```bash
cd /root/develop/workspace/work2_byd
git status --short && git fetch origin && git switch feat/credit-testbed && git pull --ff-only
git rev-parse HEAD
export AC_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
bash scripts/run_audit.sh test        # 期望 OK（家服 146 项）
```

## 2. 冻结新题（只做一次，两个检查点共用）

```bash
bash scripts/run_audit.sh freeze
sha256sum outputs/ac1_fresh/samples.jsonl
```

脚本会自动排除这几份历史题集（存在的才排除）：
`outputs/samples/pilot.jsonl`、`outputs/a1_clean_scale500/clean_rollouts.jsonl`、`outputs/si1_samples/samples.jsonl`、`outputs/si1_dp4_samples/samples.jsonl`。
**厂内如果还有其他看过结果的题集，用 `bash scripts/run_audit.sh freeze --exclude 路径1 路径2` 一并排除**，并在回报里列出来。

**期望**：`Frozen 1000 fresh questions; excluded … historical ids …`。回传这一行、sha256，以及 `selection_manifest.json` 中的 `exclusions`。

## 3. 主检查点：7B PPO（`AC_LABEL=7b_ppo`，沿用现有 `configs/local.json`）

```bash
export AC_LABEL=7b_ppo AC_GPUS=0,1,2,3
bash scripts/run_audit.sh check
nohup bash scripts/run_audit.sh all > outputs/ac1_7b_ppo_all.log 2>&1 &
tail -f outputs/ac1_7b_ppo_all.log
```

`all` 依次执行以下步骤，每一步都可以单独重跑；中断后再执行 `all` 会跳过已完成的部分，或者自动续跑：

| 步骤 | 内容 | 预计（四卡） |
| --- | --- | --- |
| `sources` | 1000 题贪心 E0 源轨迹（严格检索漂移检查） | 10–20 min |
| `pcr` | PCR 计划 → 10 题 smoke → 全量（约 1900 个状态 × 20 臂）→ 分析 | 4–5 h |
| `ct` | CT1：500 题，k=16 → policy 信号 → 分析 | 约 1 h |
| `report` | 汇总 H1/H2/H3 的机械判定 → `outputs/ac1_7b_ppo/ac1_verdicts.md` | 秒级 |

可选：`export AC_WITH_REF=1` 会让 `ct` 步骤顺带跑 trace（需要 `local.json` 里有 `base_model`）。

## 4. 第二检查点

先复制一份 local 配置，只改 `agent_checkpoint`（以及 `checkpoint_version`）：

```bash
cp configs/local.json configs/local.second.json   # 编辑 agent_checkpoint 指向第二检查点；该文件不提交
export AC_LABEL=3b_ppo AC_LOCAL=configs/local.second.json   # 标签按实际检查点命名，例如 7b_grpo
bash scripts/run_audit.sh check
nohup bash scripts/run_audit.sh all > outputs/ac1_${AC_LABEL}_all.log 2>&1 &
```

新题（`outputs/ac1_fresh`）两个检查点共用，**不要重新冻结**。

## 5. 停止条件

- `sources` 或 `pcr` 报 `Live retriever differs` / `RetrieverDriftError`：
  这两步故意沿用严格的检索一致性检查，因为 PCR 的 E0/E0 控制依赖它。先确认 Retriever 服务正常，然后**直接重跑同一条命令**（两步都支持续跑）。
  同一个 query 反复失败时停下回报，附上日志最后 50 行。
- `E0 control does not reproduce source continuation`：说明当前环境与生成源轨迹时不同。停止并回报。
- 任何 worker 非零退出：回传 `outputs/ac1_${AC_LABEL}/*/logs/*.log` 的最后 50 行。
- CT 部分的停止条件与 `docs/CT1_FACTORY_GUIDE.md` 相同（检索并列问题已在代码侧处理，自动重试）。

## 6. 回传清单（每个检查点一份）

1. `git rev-parse HEAD`、检查点路径与名字、`AC_LABEL`、GPU / 显存设置、是否设置了 `AC_WITH_REF`。
2. `outputs/ac1_${AC_LABEL}/ac1_verdicts.md` **全文**。
3. `outputs/ac1_${AC_LABEL}/pcr/analysis/pcr_report.md` 全文。
4. `outputs/ac1_${AC_LABEL}/ct/analysis/ct1_report.md` 全文。
5. 三个 `run_manifest.json` 中的耗时字段（`sources`：`wall_seconds`；`pcr`：`wall_seconds`；`ct`：`mc_wall_seconds`），以及 `ct` 的 `retrieval` 字段。
6. `bash scripts/run_audit.sh pack` 生成的 feedback 包路径。

## 7. 输出的含义

| 判定 | 确认条件（预登记） | 对应的论文主张 |
| --- | --- | --- |
| H1 | review 或 reflect 中至少一个：安慰剂修复率 / 方法修复率 ≥ 1/3，或者校正前后净值方向相反 | 恢复方法的收益主要是均值回归 |
| H2 | 删支持文档 − 删无关文档的 95% CI 上界 < 0 | 证据有因果作用，审计方法有分辨力 |
| H3 | E 的信度 ≥ 0.8，且在“有信息”轮次上 ig、ealr、fact_hit 与 E 的 ρ 都 < 0.3、CI 上界 < 0.4 | 证据类信用信号不测证据 |

`ac1_verdicts` 只是机械值；继续与否由家服在 `DECISION_LOG.md` 中决定。

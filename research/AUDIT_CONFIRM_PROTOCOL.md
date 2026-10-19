# AC1 预登记协议：搜索智能体干预式审计的确认实验

登记：2026-10-19　分支：`feat/credit-testbed`　配置：`configs/audit_confirm_v1.json`、`configs/pcr_audit_v1.json`、`configs/credit_v1_k16.json`
用户决定（2026-10-19）：启动方向 (a) 的确认实验，同时并行为 (b) 查新，两周后再定。
本文件在**看到任何 AC1 结果之前**冻结。下列假设都来自 PCR 和 CT1 的开发集（SI1 dev 题，已经看过结果）；
AC1 只用**从未用过的新题**，并且至少在两个检查点上检验。

## 1. 要确认的三条主张

| 编号 | 主张 | 开发集来源 |
| --- | --- | --- |
| H1 | 在“按失败挑出”的状态上，恢复方法（review、reflect）的修复率，至少有 1/3 能被安慰剂修复率解释，或者校正后净值的方向与未校正相反 | PCR Exp-2（G2） |
| H2 | 删支持文档比删无关文档更伤：drop_support − drop_nonsupport 的 95% CI 上界 < 0 | PCR Exp-3（G3） |
| H3 | 在“有信息”的轮次上，ig、ealr、fact_hit 与证据价值 E 的 Spearman ρ 都 < 0.3，且 CI 上界 < 0.4；前提是 E 的信度 ≥ 0.8（k=16） | CT1 k=8/k=16 |

另外只描述、不设门槛的量：lucky share、mEM / rEM、A 的均值与信度、answer_hit 的一致性、trace。

## 2. 题目与检查点

- **新题**：从本地 HotpotQA dev 和 2WikiMultihopQA dev 各取 500 题，按 `sha256(20261019:sample_id)` 排序，不看结局。
  按 **sample_id 和规范化题面**两种方式排除所有历史题集：A1 pilot、A1 scale、SI1（含旧 240 题）及其衍生（PCR、CT1 都是 SI1 dev 的子集）。跨数据集的重复题面也剔除。
- **E0 源轨迹**：每个检查点用贪心解码重新生成 1000 条（四卡，严格的检索漂移检查）。PCR 的 E0/E0 控制会再次逐 token 验证。
- **检查点**：
  - 主检查点：Search-R1 7B PPO（与历史 cohort 相同）；
  - 第二检查点：厂内可离线使用的 Search-R1 官方检查点，优先级为 3B PPO → 7B GRPO → 3B GRPO。路径写在单独的 `local` 配置里，必须在看到结果之前写入 DECISION_LOG。
  - 没有第二检查点时，只做主检查点，并在报告中写明“未在第二检查点确认”。

## 3. 每个检查点上的实验

| 实验 | 引擎 | 规模 | 臂 |
| --- | --- | --- | --- |
| PCR-A | `work2.pcr`，arm group `audit` | 1000 题中有检索的全部；目标轮 1/2 | PCR v1 的 16 臂，加上 `reflect`、`neutral_reflect`、`perm1_reflect`、`perm1_neutral_reflect`，共 20 臂 |
| CT-A | `work2.credit_testbed`，k=16 | 每个数据集 250 题（从新题中按 CT 的种子排序） | V / N / C 三类分支；信号含 policy（ig、ealr）和可选的 reference（trace） |

`reflect` 是一种比 review 更具体的反思提示（要求找出尚未解决的部分、检查证据、必要时发更具体的查询），文本固定在 `work2/placebo.py`。
`neutral_reflect` 是与它 token 数完全相同的语义空文本。

## 4. 判定

- 每个检查点、每条主张分别判定：**确认 / 未确认**。H1 对 review 和 reflect 分别判定，至少一个确认才算 H1 确认。
- H1 另外报告 Δ = repair(perm1_m) − repair(perm_2) 在退化层 D 上的 bootstrap CI（按题聚类），只描述、不设门槛。
- **继续写合稿 (a) 的条件**：主检查点上 H1、H2、H3 至少两条确认，并且第二检查点上（如果有）同方向的至少一条确认。
  否则，(a) 只按确认通过的部分写，或者并入 (b) 的结果再决定。
- `work2.audit_confirm report` 只输出机械值；最终决定由人记录在 `DECISION_LOG.md`。

## 5. 不做的事

- 不在开发集（SI1 dev、PCR、CT1 的题）上改估计量或门槛。
- 不加 RL 训练。
- AC1 结果出来之后，不再新增恢复方法或信号来补救。

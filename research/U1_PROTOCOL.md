# U1 预登记协议：多智能体、多数据集的干预式审计（冲主会升级 U1）

登记：2026-10-22　分支：`feat/credit-testbed`
配置：`configs/u1_protocol.json`、`configs/u1_agents.json`、`configs/pcr_u1_{bounded,official}.json`、`configs/credit_u1_{bounded,official}.json`
用户决定（2026-10-22）：目标 ACL 2027 主会；确认升级计划，先做 U1 和 U2 查新，U3 等训练流程的 pilot 再定。
本文件在看到任何 U1 结果前冻结；唯一留待之后决定的是提示词变体（见 §2），它必须在任何 U1 运行之前、按 FC1 的规则定下来，并记进决策日志。

## 1. 目的

AC1 已在 2 个检查点、2 个数据集上确认 H1 与 H3。U1 把同一套测量扩到：
- 一份智能体名单（见 §3），覆盖不同规模、PPO / GRPO、base / instruct 底座；
- 5 个数据集：HotpotQA、2Wiki、NQ、MuSiQue、Bamboogle。

目的是回答审稿人“只测了两个模型、两个数据集”的质疑。

## 2. 设置

- **题目**：
  - HotpotQA 与 2Wiki 直接复用 AC1 的新题（各 500 道），与 AC1 结果可以逐题对照；
  - NQ 500、MuSiQue 500 道，Bamboogle 全部（125 道）取自 FlashRAG 格式的测试集，按 `sha256(20261022:sample_id)` 排序，不看结局，并按题面排除所有历史题集；
  - 某个数据集文件在厂内不存在时，跳过并记录，不换成其他数据。
- **多答案**：NQ 等数据集有多个标准答案，EM 取最匹配的那个（Search-R1 / FlashRAG 的惯例）；IGPO 信号用第一个答案打分。
- **没有支持事实的数据集**（NQ、MuSiQue、Bamboogle 的 FlashRAG 版本）：`fact_hit` 和删除臂不可用，H3 的判定跳过 `fact_hit`，但 `ig` 与 `ealr` 必须可用；H2 在 U1 中不评估。
- **提示词**：只能是 `bounded`（我们的）或 `official`（Search-R1 官方原文）。**按 FC1 规则 D1 决定**：只要 FC1 显示官方提示词对 (a) 的任一检查点高 ≥ 3 pp（CI 下界 > 0），就用 `official`，否则用 `bounded`。决定写入 DECISION_LOG 后才能运行（脚本要求设置 `U1_PROMPT`）。
- **每个智能体的测量**：
  1. 贪心 E0 源轨迹：全部 U1 题；
  2. **PCR（H1）**：每个数据集按 hash 取最多 400 道有检索的题，目标轮 1 和 2。只跑 H1 需要的 9 个臂：`clean_a/b`、`perm_1`、`perm_2`、`perm1_review`、`perm1_neutral`、`perm1_budget2`、`perm1_reflect`、`perm1_neutral_reflect`；
  3. **CT（H3）**：每个数据集 150 道（Bamboogle 全部），k=16，温度 1.0；信号为 ig、ealr（policy 模型），以及 fact_hit、answer_hit（模型无关）。
- 解码和检索设置与 AC1 相同（E5 + Wiki-18、Top-3、最多 4 次检索）。

## 3. 智能体名单（`configs/u1_agents.json`）

| 标签 | 官方仓库 | 说明 |
| --- | --- | --- |
| `v01_7b_ppo` | `…qwen2.5-7b-em-ppo` | 旧版，与 PCR / CT1 / AC1 衔接 |
| `v03_3b_ppo` | `…qwen2.5-3b-em-ppo-v0.3` | AC1 第二检查点 |
| `v03_3b_grpo` | `…qwen2.5-3b-em-grpo-v0.3` | |
| `v03_7b_ppo` | `…qwen2.5-7b-em-ppo-v0.3` | |
| `v03_7b_grpo` | `…qwen2.5-7b-em-grpo-v0.3` | |
| `v03_7b_it_grpo` | `…qwen2.5-7b-it-em-grpo-v0.3` | instruct 底座 |
| `v03_14b_grpo` | `…qwen2.5-14b-em-grpo-v0.3` | 更大规模 |

镜像上拿不到的仓库，跳过并记录。非 Search-R1 系的智能体（如 R1-Searcher）动作格式不同，需要适配器，等 U2 查新后再决定，**不在本协议的判定范围内**。

## 4. 判定（机械值由 `work2/u1.py summary` 输出，人工记录决定）

- **纳入**：格式 / 长度失败率 < 10% 的智能体；其余的排除并报告。
- **每个智能体（汇总全部数据集）**：
  - H1 确认：review 或 reflect 的 placebo explained ≥ 1/3，或者校正前后净值方向相反；
  - H3 确认：E 的信度 ≥ 0.8，且在“有信息”轮次上，可用的证据类信号（ig、ealr，以及可用时的 fact_hit）与 E 的 ρ 都 < 0.3、CI 上界 < 0.4。
- **主张成立**：H1（以及 H3）在 ≥ ⌈0.8 × 纳入智能体数⌉ 个智能体上确认（例如 7 个中 ≥ 6 个，6 个中 ≥ 5 个）。
- **分数据集结果只作描述**：如果某个数据集上普遍不成立，如实写出，不删数据集。

处置：
- 两条主张都成立 → 进入论文主表；
- 只有一条成立 → 论文主张相应收窄；
- 都不成立 → 停下重新评估 (a)。

## 5. 不做的事

- 不在 AC1 的开发数据上改估计量；不在看到 U1 结果后增删智能体或数据集（拿不到的除外，并且要记录）。
- 不在 U1 里训练模型（U3 另行登记）。

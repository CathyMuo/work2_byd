# CT1 预登记协议：搜索智能体逐轮信用信号的干预式检验（pilot，第 1–2 周）

登记：2026-10-16　分支：`feat/credit-testbed`　配置：`configs/credit_v1.json`
选题依据：`WORK2_IDEA_SURVEY_20261016.md`（方向 A）。本文件在看到任何 CT1 结果前冻结；
要修改就另开 v2，并在 `DECISION_LOG.md` 写明理由。

## 1. 问题

2026 年大量 turn-level reward / credit 方法（IGPO、IG-Search、TRACE、CIPO、KnowGPO、STAMP、E-GRPO……）
声称在奖励“这一轮搜索的贡献”或“证据的作用”，但都没有用**真实重新执行**的因果量去验证。
QVal（2606.32034）只做了非搜索环境、只干预动作，也没有下游 RL 验证。
CT1 pilot 回答：**在搜索 QA 上，能否便宜而可靠地测出逐轮的干预真值？主流信号与真值的一致性如何？**

## 2. Cohort 与轨迹

- 策略：Search-R1 7B PPO（与 A1/SI1/PCR 相同的检查点、E5 + Wiki-18 Flat、Top-3 / Top-20、预算 4、prompt `work2_search_r1_bounded_v1`）。
- **on-policy 采样**：temperature 1.0、top_p 1.0（Search-R1 训练时的 rollout 设置），每个请求使用确定性种子。
- 题目：从 SI1 dev 源文件里，HotpotQA 和 2Wiki 各取 150 题，按 `sha256(20261017:sample_id)` 排序取前 150；**不看结局**。
- 每题采样 1 条**被打分轨迹** τ（种子固定），再用同种子重采一次，检查能否复现（只记录复现率，不中止运行）。

## 3. 干预真值（每题批量、同前缀重新执行）

记 s_j 为 τ 在第 j 次观测之后的 token 前缀（s_0 = 初始 prompt），共 T 次搜索。V(·) 表示从该前缀起、用同一策略采样 k=8 次续写的平均 EM（已用掉的搜索次数照常计入预算）。

| 量 | 定义 |
| --- | --- |
| V_j | V(s_j)，j = 0..T |
| V∅_j | 把第 j 次观测换成**语义空观测** o∅ 之后的 V，j = 1..T（查询文本不变，之后的检索正常进行） |
| V_cb | 从 s_0 起，**所有**检索都返回 o∅（“无证据世界”），用来衡量这道题不靠证据能不能答对 |
| **A_t** = V_t − V_{t−1} | 第 t 轮（查询 + 观测）的 on-policy 优势 |
| **E_t** = V_t − V∅_t | 第 t 次观测内容的因果价值 |

o∅ 的构造：取同批另一道题（按冻结顺序取第 i + n/2 题）的题面作为查询，检索 Top-20，
依次取前 3 篇“不含本题 gold 答案串、没有支持事实词法命中”的文档，按标准格式渲染。
o∅ 不够 3 篇时，这道题记为 ineligible。

## 4. 被检验的信号（都在同一条 τ 上离线计算）

| 名称 | 定义 |
| --- | --- |
| `ig` | IGPO 式 Eq.3–4：在 s_t 后接 `<think>Now there's enough information to answer</think><answer>` + gold + `</answer>`，取 gold token 的平均 log-prob；ig_t = 该值(s_t) − 该值(s_{t−1}) |
| `ig_deadzone` | IG-Search 式：\|ig_t\| < 0.05 时置 0 |
| `trace` | TRACE 式：同样的 gold 平均 log-prob，但在冻结的参考模型（`base_model`，即 Qwen2.5-7B-Instruct）下取 TD 差。可选；参考模型不可用时记为 missing |
| `ealr` | CIPO 式：在观测 t 之后的下一段生成上，计算正常注意力与“目标 token 屏蔽观测 t 区段”（4D 加性掩码，token 和位置都不变）两种条件下的平均 log-prob 差。没有下一段生成的轮次记为 missing |
| `fact_hit` | STAMP / E-GRPO 式：第 t 次观测**首次**出现的支持事实数（标题+句子词法代理） |
| `answer_hit` | 第 t 次观测首次出现 gold 答案串，取 0/1 |
| `random` | 与结局无关的均匀随机数（`sha256(sample_id|t)`） |

LLM 评论器和提示式打分（QVal 中最强的一族）留到第 3–4 周，**不进入**本 pilot 的门槛。

## 5. 估计量

- 一致性：轮级 pooled Spearman ρ(signal, A) 与 ρ(signal, E)，按整题重抽、数据集内分层做 bootstrap（2000 次，seed 42）。
- AUROC：标签为 E_t ≥ 0.25（观测确实有用）时，各信号的区分能力。
- 分层：
  - **记忆可答**（V_cb ≥ 0.5）vs 不可答；
  - 轮次 t = 1 / ≥ 2；
  - τ 成功 vs 失败。
- 信度：k=8 次样本按奇偶拆成两半，分别算 A、E，取两半之间的 Pearson r，再做 Spearman-Brown 校正 2r/(1+r)。
- 描述量：E_t 的分布，A_t 与 E_t 的关系，V_cb 的分布，V_T − V_cb。

## 6. Go / No-Go（pilot 结束后判定一次）

- **G-a 真值可靠**：A 和 E 的 Spearman-Brown 信度都 ≥ 0.6。
- **G-b 存在值得报告的差异**：满足任一条即可——
  1. `ig`、`ealr`、`fact_hit` 中至少一个的 ρ(·, E) < 0.3，且 CI 上界 < 0.5；
  2. `ig` 或 `ealr` 在“记忆可答层”与“不可答层”之间的 ρ(·, E) 差值，CI 不含 0；
  3. `ig` 与 `ealr` 之间的 ρ < 0.3（主流信号彼此不一致）。
- **G-c 成本可控**：本 pilot 的 MC 墙钟时间 × (600/300) ≤ 8 h。

处置：
- G-a 不过 → 先把 k 增到 16 重测一次；仍不过就 NO-GO。
- G-a 过、G-b 不过 → 主流信号已经与真值一致，结论影响力低，降级为 Findings（只保留预测效度实验）。
- G-c 不过 → 正式实验降到 k=8 / 400 题，或把 o∅ 分支改为只在 t ≤ 2 上做。
- 全部通过 → 进入第 3–4 周：3 个检查点、全部信号（含 LLM 评论器、提示式打分），另立协议；第 4–8 周进行 3B RL 预测效度实验。

`credit_analysis` 只输出机械值；GO 由人在 `DECISION_LOG.md` 中记录。

## 7. 已知局限

- E_t 的语义空观测只是一种“无信息”操作化（用无关题的真实检索结果），结论限定在这个定义下；附录可补“明确的无结果”变体。
- k=8 的二值平均有采样噪声（按 PCR F5，单实例噪声明显），所以必须报告信度，并且只做总体相关，不对单轮下结论。
- `trace` 用 Instruct 模型代替原文的参考模型，属于近似；`ig` 的 gold 拼接方式按 IGPO 原文 Eq.3 的格式。

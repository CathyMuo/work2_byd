# U1L 预登记协议：H1 在更大的智能体上是否成立（32B）

登记：2026-10-28　分支：`feat/credit-testbed`　名单：`configs/u1l_agents.json`　脚本：`scripts/run_u1l.sh`
用户决定（2026-10-28）：补“更大、更强的智能体”这一项，并把实验做充分。本文件在任何 U1L 运行之前冻结。

## 1. 为什么做

目前 H1 主表只到 7B。14B GRPO 因为插入文字后给不出答案而被判为不可测。SearchAuditor 在强模型上得到相反结论（定向修复明显胜过对照）。审稿人最可能的质疑是“结论只适用于小模型”。U1L 直接检验这一点：
1. 在 32B 的 RL 搜索智能体上 H1 是否仍然成立；
2. 同一训练配方从 7B 到 32B，内容效应和伤害怎么变化；
3. 不做搜索 RL、只靠提示的同尺寸模型是否一样。

## 2. 智能体（运行前定死）

**查新结论**：
- 32B、公开权重、能接入我们 E5 + Wiki-18 本地检索的 RL 搜索智能体，只找到 ReSearch（NeurIPS 2025）一家。
- Search-R1、ZeroSearch、StepSearch、AutoRefine 最大只到 14B 或 7B。
- ASearcher-Local 需要页面访问工具和它自己的知识库，不兼容；ASearcher-Web、WebSailor 等是网页环境。

| 标签 | 权重 | 类型 | 提示与格式 |
| --- | --- | --- | --- |
| `research_32b` | `agentrl/ReSearch-Qwen-32B-Instruct` | RL（GRPO，MuSiQue 训练） | `research` |
| `qwen25_32b_inst` | `Qwen/Qwen2.5-32B-Instruct` | 只靠提示，没有搜索 RL | `official`（Search-R1 原文模板） |
| `research_7b` | `agentrl/ReSearch-Qwen-7B-Instruct` | RL，同一配方的 7B | `research` |
| `qwen25_7b_inst` | `Qwen/Qwen2.5-7B-Instruct` | 只靠提示 | `official` |

- 运行顺序就是表中的顺序：最重要的 32B RL 先跑。
- **只靠提示的两个模型不并入 RL 主表**，单独报告，作为“规模与 RL 的参照”。

## 3. 格式与 harness

- **ReSearch 格式**（从 Agent-RL/ReCall 仓库 `7fcf4d3~1` 读出原文，即 ReSearch 权重发布时的代码，用 ast 读取，没有执行；sha256 记录在 `tests/test_u1l.py`）：
  - chat template，系统消息为 `re_search_template_sys` 原文，用户消息就是问题本身；
  - 观测为 ` <result>\n{各文档 contents 用空行连接}\n</result>`；
  - 答案取 `<answer>` 内最后一个 `\boxed{}` 的内容，与上游 `remove_boxed(last_boxed_only_string(...))` 相同；没有 `\boxed{}` 时答案为空。
- **ReSearch 的环境按上游设置**：E5 + Wiki-18、**Top-5**、**最多 8 次检索**（上游不设轮数上限，只限制 8192 token 的回复长度）、上下文 10240 token、每轮最多生成 1024 token。为此 PCR 的检索预算检查放宽为“4 或 8”，只有这个格式用 8。
- **两个只靠提示的模型**用 U1 的标准环境：Top-3、最多 4 次检索、上下文 8192。
- 在真实的 Qwen2.5-Instruct 分词器上核对过：渲染结果与上游的 `apply_chat_template` 一致。
- **题目**：U1 冻结的 2125 道题。PCR 每个数据集最多 400 道可检索的题，第 1、2 轮，`h1` 臂组，与 U1 相同。

## 4. 判定

- **排除**：源轨迹格式或长度失败 ≥ 10% 的智能体排除（同 U1）。
- **保真度**（参考值在运行前从 ReSearch 论文 Table 2 读出，记录在 `work2/u1x.py`）：

  | | HotpotQA | 2Wiki | MuSiQue | Bamboogle |
  | --- | --- | --- | --- | --- |
  | 32B-Instruct EM | 46.7 | 44.9 | 26.4 | 56.8 |
  | 7B-Instruct EM | 43.5 | 47.6 | 22.3 | 42.4 |

  宏平均差 < −10 pp 时，标为“可能 harness 不符”，单独报告，不计入结论。
- **插入有效性**：沿用 2026-10-28 写定的格式判据，即无内容文字 P2 下稳定状态的正常作答率 ≥ 50%。不满足的智能体不计入 H1，如实报告它的失效形态。
- **每个智能体的 H1 判定**：与 U1 相同（AC1 写法）。另外报告效应量：方法 − P2（修复）、方法 − P1、方法 − P2（伤害），以及 CI。
- **预登记的问题与读法**：
  - **Q1（规模）**：如果 `research_32b` 有效，并且 H1 成立、方法 − P1 的 CI 全在 0 以下，就写“结论延伸到 32B 的 RL 智能体”。如果它的内容效应（方法 − P2 修复）CI 全在 0 以上，并且 ≥ 10 pp，就如实写“在 32B 上内容开始起作用，结论随规模变化”。
  - **Q2（同配方，7B → 32B）**：并列报告两个 ReSearch 的内容效应和伤害（不同模型、不同状态，不做配对检验，只描述）。
  - **Q3（RL vs 只靠提示）**：同尺寸下，ReSearch 和 Qwen-Instruct 的效应并列，只描述。
  - **主表更新**：RL 主表 = 已有的 9 个有效智能体 + U1L 中有效的 RL 智能体。只靠提示的模型单独成表。
- 机械值由脚本给出，决定由人写进 `DECISION_LOG.md`。

## 5. 资源（4×H800，只推理）

- 新下载：两个 32B 各约 65 GB，ReSearch-7B 约 15 GB。Qwen2.5-7B-Instruct 在 HR1 时已经下载。
- 32B 在单卡上运行需要 `U1_GPU_MEM=0.92`（权重约 65 GB，留出约 8 GB 给 KV 缓存）。
- 估算：
  - 32B：源轨迹约 3–4 小时，PCR 约 20–24 小时。PCR 为了保证 E0 控制逐字复现，是逐条贪心生成的。
  - 7B：每个约 3–5 小时。
  - 合计约 2.5–3 天。每跑完一个智能体回报一次，`research_32b` 最先出结果。

## 6. 局限

- 32B 的 RL 搜索智能体只有 ReSearch 一家。
- ReSearch 的环境与其他智能体不完全相同（Top-5、8 次检索），但 H1 是同一智能体、同一环境内部的比较。
- 只靠提示的模型不属于“RL 训练的搜索智能体”，只作参照。

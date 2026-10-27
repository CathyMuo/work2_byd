# U1x 预登记协议：H1 在其他训练配方上的普适性

登记：2026-10-27　分支：`feat/credit-testbed`　名单：`configs/u1x_agents.json`　脚本：`scripts/run_u1x.sh`
用户决定（2026-10-27）：做完整方案（ZeroSearch 两个 + AutoRefine-7B-Base + StepSearch-7B-Base），只跑 H1。
本文件在任何 U1x 运行之前冻结。候选查新见家服本地文档《工作二_非SearchR1智能体查新.md》。

## 1. 目的

U1 的 7 个智能体都来自 Search-R1 一家的训练配方。审稿人很可能问：H1（恢复收益主要是均值回归）是不是这一配方特有的现象。U1x 加入另外三种配方和一个非 Qwen 家族，检验 H1 的普适性。

## 2. 智能体（运行前定死，不按结果增删）

| 标签 | 权重 | 底座 | 配方 | 提示词变体 |
| --- | --- | --- | --- | --- |
| `zs_llama32_3b` | `Alibaba-NLP/ZeroSearch_wiki_V2_Llama_3.2_3B` | Llama-3.2-3B | ZeroSearch：用 LLM 模拟搜索引擎做 RL，课程式加噪；论文默认用 REINFORCE | `official`（与 Search-R1 官方模板逐字相同） |
| `zs_qwen25_7b` | `Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B` | Qwen2.5-7B | ZeroSearch | `official` |
| `autorefine_7b` | `yrshi/AutoRefine-Qwen2.5-7B-Base` | Qwen2.5-7B | AutoRefine：GRPO + 检索专用奖励，每次检索后在 `<refine>` 里提炼 | `autorefine` |
| `stepsearch_7b` | `Zill1/StepSearch-7B-Base` | Qwen2.5-7B | StepSearch：StePPO，基于信息增益和冗余惩罚的逐步奖励 | `stepsearch` |

## 3. 格式与 harness

- **提示词**：都用 ast 从上游仓库读出原文，没有执行上游代码；sha256 记录在 `tests/test_u1x.py`。
  - ZeroSearch：`inference.py`，与 Search-R1 官方模板相同；
  - AutoRefine：`preprocess/data_process/utils.py`，`template_type="autorefine"`；
  - StepSearch：`scripts/data_process/{nq,hotpot}_search.py` 里实际生效的第二个 prefix，与论文 Table 4 一致。
- **chat template**：都按上游做法，作为单条用户消息、tokenizer 有 chat template 时套用。ZeroSearch 的 Llama 版没有 chat template，直接用原文，并按上游 `tokenizer.encode` 的行为加 BOS（真实 Llama-3.2 分词器上已核对 token 完全一致）。
- **观测**（`work2/rollout.py` 的 `FORMATS`）：

| 变体 | 观测块 | 文档格式 |
| --- | --- | --- |
| `autorefine` | `\n\n<documents>…</documents>\n\n`，文档块 strip | `Doc i(Title: t) text` |
| `stepsearch` | `\n\n<information> … </information>\n\n`，标签内侧留空格，文档块 strip | `Doc i<## Title: t ##> text` |
| `official`（ZeroSearch） | 与 U1 中 Search-R1 智能体完全相同 | 同左 |

  `official` 沿用历史上不 strip 的写法，因为 U1 的 Search-R1 智能体都是这样跑的；ZeroSearch 与它们保持一致。
- **统一的环境**：E5 + Wiki-18，Top-3，最多 4 次检索，贪心源轨迹。与上游的偏差如实记录：
  - AutoRefine 评测用 3 次检索；
  - StepSearch 训练用 5 次检索，并在最后一轮附加警告语，我们不复现这句警告；
  - ZeroSearch 论文用 Google、Top-5。
- **题目**：U1 冻结的 2125 道题（`outputs/u1_fresh`）。H1 部分沿用 U1 的 PCR 设置：每个数据集最多 400 道可检索题，第 1、2 轮，`h1` 臂组。**不跑 H3**。

## 4. 判定

- **排除**：源轨迹格式 / 长度失败 ≥ 10% 的智能体排除（同 U1）。
- **保真度**（`work2/u1x.py`，参考值在运行前从论文读出）：
  - AutoRefine-7B-Base：NQ 0.484、HotpotQA 0.451、2Wiki 0.405、MuSiQue 0.187、Bamboogle 0.512；
  - StepSearch-7B-Base：HotpotQA 0.380、2Wiki 0.385、MuSiQue 0.216、Bamboogle 0.467；
  - ZeroSearch 没有可比参考值（论文用 Google），只报告 EM。
  - 我们的 EM 与参考值的宏平均差 < −10 pp 时，标为“可能 harness 不符”：单独报告，不计入普适性主张。
- **每个智能体的 H1 判定**：与 U1 完全相同（AC1 写法，语义口径）。
- **普适性主张**：在纳入且未标记的 U1x 智能体中，H1 确认数 ≥ ceil(0.8 × 纳入数)。另外报告 U1 + U1x 全部纳入智能体的合并效应量（`work2.paper_h1 --agents configs/u1_agents.json configs/u1x_agents.json`）。
- 机械值由脚本给出，决定由人写进 `DECISION_LOG.md`。

## 5. 资源

- 只做推理。新下载：Llama-3.2-3B 约 6.5 GB，三个 7B 各约 15 GB。
- 四卡上每个智能体约 1–2 小时（源轨迹 + PCR），合计约 5–7 小时。

## 6. 局限

- 只有一个非 Qwen 智能体（ZeroSearch-Llama 3B）。
- ZeroSearch wiki 版训练时看到的是模拟文档，在真实 E5 检索上可能存在分布偏移。
- 这些智能体的 harness 细节（检索次数、Top-k、警告语）与上游有小的偏差。H1 是在同一个智能体、同一个 harness 内部比较方法与安慰剂，所以偏差影响的是 EM 水平，不直接影响 H1 的比较。

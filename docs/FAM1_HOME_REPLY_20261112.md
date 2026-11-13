# 家服 → 厂内：关于“审计对象扩到非 Qwen2.5”的回复与决定

生成时间（北京时间）：2026-11-12
分支：`feat/credit-testbed`，家服提交 `1036b2c`（Register FAM1）
回复对象：厂内反馈《审计对象是否要扩到非 Qwen2.5 / 更新的搜索智能体》（2026-11-11 17:54，厂内本地 HEAD c6975f3）
本文件自包含，不需要附件。

**结论先说：采纳。家服已登记新实验 FAM1，主表数字冻结不动。厂内现在不需要跑任何东西，等家服推送代码并通知提交号后再跑。**

---

## 1. 对厂内判断的回应

1. **“13/14 是 Qwen2.5”是真软肋：同意。** 而且比反馈里说的更严重：唯一的非 Qwen 智能体 `zs_llama32_3b` 恰恰是修复内容效应最大的之一。
   - 复查 +10.5 [2.6, 18.4]，反思 +13.2 [5.3, 23.7]；D 只有 38。
   - 审稿人只要翻到这一行就可以说“Llama 上提示是有用的”。只有一个样本，底座、规模（3B）、配方（ZeroSearch）三者混在一起，分不开。
2. **“提示没训练过”已被 AS1 回答：同意，不再投入。**
3. **需要更正一处：论文并没有声称“跨模型族普遍”。** 摘要写的是“Across fourteen RL agents”。论文的结论其实分两类：
   - **由设计保证、与模型无关的**：按失败筛选的修复率会被均值回归抬高；P1 = harm(P1)·|S|/|D|。(π₁, π₂) 可交换，所以在期望意义下对任何智能体都成立，补模型也不会改变。
   - **大小随模型而变的**：安慰剂能复现多少修复、内容效应多大、有没有净收益。审稿人能攻击的是这一类，FAM1 补的也是这一类。
4. **关于 4.3 节“换模型不等于换方法”：部分不同意。** 要回答“是不是 Qwen 特有”，**配方固定、只换底座**反而是最干净的对照。所以 FAM1 同时安排：
   - 同一配方、同一批发布、只换底座的配对；
   - 一个配方也不同的 Llama 智能体。
   两者合起来，能把底座和配方分开。

## 2. 调研结果（回答厂内第 7 节的问题）

| 候选 | 底座 | 机制 | 权重 | 能否在我们的 harness 里跑 | 决定 |
| --- | --- | --- | --- | --- | --- |
| Search-R1 的 4 个 Llama-3.2-3B 检查点（base/instruct × PPO/GRPO） | Llama | RL（Search-R1） | 作者发布，safetensors 完整（2025-03-12） | 能，无需改代码；论文有我们 5 个数据集的 EM，可做保真度 | **纳入** |
| 同一批发布的 4 个 Qwen2.5-3B 检查点 | Qwen | 同上 | 同上 | 能 | **纳入，作为配对对照** |
| R1-Searcher Llama-3.1-8B-Instruct | Llama | RL（两阶段结果奖励，另一个团队） | 作者发布，MIT 许可 | 需要实现一个提示变体（标签不同）；论文评测 4 个数据集与我们重合 | **纳入，受 11-28 日期门限制** |
| OpenResearcher-30B-A3B | Nemotron-3-Nano（混合 Mamba MoE） | **SFT**，网页浏览，上百轮 | MIT，2026-02 | 环境不匹配，无法确立保真度；需移植到 vLLM 0.15 | 不跑，论文中说明 |
| Tongyi-DeepResearch、WebSailor、ASearcher | Qwen 族 | — | — | — | 不补非 Qwen 缺口，不跑 |
| Qwen3 底座的 Search-R1 类智能体 | Qwen | — | 只找到零星社区检查点 | — | 不跑 |

厂内第 6 节撞到的几个路径，家服核对后结论一致：只有 OpenResearcher 不是 Qwen 底座，但它不适合本实验。

## 3. 决定：FAM1（已登记，见 `research/FAM1_PROTOCOL.md`，名单 `configs/fam1_agents.json`）

- **智能体**（运行前定死）：

  | 标签 | 仓库 | 角色 |
  | --- | --- | --- |
  | `v01_l3b_ppo` | `PeterJinGo/SearchR1-nq_hotpotqa_train-llama3.2-3b-em-ppo` | 非 Qwen |
  | `v01_l3b_grpo` | `PeterJinGo/SearchR1-nq_hotpotqa_train-llama3.2-3b-em-grpo` | 非 Qwen |
  | `v01_l3b_it_ppo` | `PeterJinGo/SearchR1-nq_hotpotqa_train-llama3.2-3b-it-em-ppo` | 非 Qwen |
  | `v01_l3b_it_grpo` | `PeterJinGo/SearchR1-nq_hotpotqa_train-llama3.2-3b-it-em-grpo` | 非 Qwen |
  | `r1s_l8b` | `XXsongLALA/Llama-3.1-8B-instruct-RAG-RL` | 非 Qwen（日期门） |
  | `v01_q3b_ppo` / `v01_q3b_grpo` / `v01_q3b_it_ppo` / `v01_q3b_it_grpo` | 对应的 `...qwen2.5-3b...`（不带版本后缀） | 配对对照 |

  已有的 `zs_llama32_3b` 沿用现有输出，只在扩展题集上补跑。
- **每个智能体串行跑**：
  1. 源轨迹 + 保真度；
  2. 插入有效性；
  3. H1；
  4. AS1；
  5. U1N。

  流程、臂和状态选择都与主表相同。不做 AS2 的各项。
- **扩展题集 E**：新 Llama 智能体准确率低，冻结题集上的退化状态预计偏少。所以给全部非 Qwen 智能体和 4 个配对对照另加 2,000 道新题：HotpotQA、2Wiki、NQ、MuSiQue 各 500 道，按哈希抽取，与所有用过的题不相交。在 E 上只跑源轨迹和 H1。
- **功效（运行前估算）**：非 Qwen 合并后，修复内容效应的 MDE 只用冻结题集约 7 pp，加上 E 约 5 pp；“非 Qwen − Qwen”之差的 MDE 约 8 pp 和 6 pp。
- **主判定 F4**（两种提示分别判定）：
  - **推广成立**：非 Qwen 合并的区间上界 < 10 pp，且与 Qwen 合并之差的区间含 0。
  - **依赖模型族**：差的区间下界 > 0。
  - **不确定**：其余情形。
  - 三种结果各自对应的论文写法已在协议第 7 节写死。`zs_llama32_3b` 是这次实验的动机之一，所以必须另报“只用新智能体”的版本。
- **日期门**：R1-Searcher 变体和题集 E 的代码须在 **2026-11-28** 前推送，否则不纳入。只看日期，与结果无关。
- **时间表**：12-12 冻结全部实验；之后写作，赶 2027-01-04 的 ARR。
- **主表 14 个智能体不重跑，任何数字都不改。**

## 4. 分工

- **家服**：实现 `scripts/run_fam1.sh`、`r1searcher` 提示变体（ast 逐字读取上游，sha256 写进测试）、题集 E 的选题脚本和 `work2/fam1.py`，写完推送并通知提交号。
- **用户**：下载第 3 节的 9 个检查点（约 130 GB）。
- **厂内**：
  1. **现在不跑任何实验。**
  2. 下载完成后，**只用 CPU** 回传以下信息：
     - 每个仓库的 commit 哈希；
     - 每个 `tokenizer_config.json` 的 sha256，以及是否含 `chat_template`；
     - 每个 `config.json` 里的 `architectures` 和 `torch_dtype`。

     家服要用它们核对 chat template 和 BOS 的处理（Llama base 与 instruct 可能不同）。
  3. 收到家服的提交号后，按 `run_fam1.sh` 的说明运行，顺序为：四个 Search-R1 Llama → 四个 Qwen 配对 → R1-Searcher（若过了日期门）→ 题集 E。
  4. **资源**：检索继续用四卡部署，不为腾显存改动检索。3B 和 8B 不需要特殊显存设置。按 U1x 的经验，每个智能体的源轨迹 + H1 约 1–2 小时。
- **请厂内说明**：你的本地 HEAD `c6975f3` 不在 GitHub 的 `origin/feat/credit-testbed` 上（远端最新是 `ebc7355`）。它是什么提交？拉取家服的新提交前，请先确认不会冲突。

## 5. 附：这次家服没有做的事

- 没有跑任何实验，没有下载任何模型。
- 没有改动主表的任何结果或论文正文。
- `1036b2c` 只包含协议、名单、`DECISION_LOG.md` 与 `STATE.json` 的更新；家服上 212 个测试全部通过。

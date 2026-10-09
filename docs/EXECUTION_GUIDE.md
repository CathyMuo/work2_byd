# 工作二详细执行步骤

更新日期：2026-10-09。适用仓库：`CathyMuo/work2_byd`。
执行地点：厂内 4×H800；代码修改与推送由开发服务器完成。

## 1. 现在到底能跑什么

**当前版本可以执行 A1：clean Search-R1 baseline。不能一条命令完成全部研究实验。**

此前初始化版本只有仓库、配置和环境工具，没有工作二 rollout 程序。
本次增加本地样本选择、Retriever 启动入口、模型推理、有限轮数 rollout、
指标汇总、案例导出和结果反馈脚本。模型、数据、语料、索引由你在厂内提供，
本仓库不下载它们、不合成新问题，也不重新配置你的 Search-R1 环境。

| 实验阶段 | 本次状态 | 当前可执行内容 |
| --- | --- | --- |
| A0 仓库初始化 | 完成 | 公开拉取、依赖固定 |
| A1 clean baseline | 已实现，待厂内实际运行 | 10 题小样本、200 题 clean baseline、基础模型对照 |
| A2 相同前缀重放 | 尚未实现 | A1 已保存前缀，后续实现 E0/E0、E0/E1 续写 |
| A3 E0–E4 干预 Pilot | 尚未实现 | 后续实现事实核验、单轮干预与配对指标 |
| B 恢复基线 | 尚未实现 | 后续比较预算、过滤、Prompt、Harness-G |
| C 新 RL 方法 | 尚未实现，研究 GO 尚未判定 | 不在当前版本执行 |

已完成 CPU 协议/逻辑测试；尚未在厂内 H800、真实模型及 Wiki-18 上运行。
“代码可执行”表示运行入口已提供，不表示 GPU 兼容性或研究结论已得到验证。

## 2. 为什么先跑 A1：对应我们的研究问题

研究问题是：某一轮真实检索观测变化，是否会改变后续搜索子目标、导致偏移，
以及 Agent 是否能在有限预算内恢复关键事实。

要回答这个问题，先要确认基线能正确调用工具、形成足够的多轮轨迹，并保存
可重放的检索前状态。否则模型本来就不会搜索或上下文已截断，后续失败无法
归因于检索干预。

本轮 A1 只回答：

1. Search-R1 checkpoint 能否产生有效工具调用和规范答案？
2. 是否有实际执行第 2 轮及后续检索的轨迹？
3. 在固定预算内，答案、工具调用次数和终止情况是什么？
4. 是否留下后续同前缀分叉所需的 token 前缀和候选文档？

**A1 的 EM 变化、两个模型的答案差异、Query 字符串差异，都不能直接证明
检索诱导的因果偏移，也不能计算本研究的 Recovery 指标。**

## 3. 厂内需要已有的资源清单

| 资源 | 要求 | 本仓库是否提供/下载 |
| --- | --- | --- |
| Search-R1 环境 | 你已配置的可用环境 | 否，直接沿用 |
| Agent checkpoint | 优先已训练 Search-R1 Qwen2.5-3B；本地 Hugging Face 格式，包含权重、config 和 tokenizer | 否 |
| 基础模型 | Qwen2.5-3B-Instruct，同样为本地完整模型目录；对照可稍后运行 | 否 |
| E5 模型 | 本地 E5-base-v2 完整目录 | 否 |
| 检索语料 | Wiki-18 JSONL，供现有检索服务读取 | 否 |
| FAISS 索引 | 与同一份 Wiki-18 passage 行号/ID 对齐 | 否 |
| HotpotQA | 本地 development split，原始 JSON 数组或 JSONL | 否 |
| 2WikiMultiHopQA | 本地 development split，原始 JSON 数组或 JSONL | 否 |

数据行要求：`_id` 或 `id`、`question`、字符串 `answer`、
`supporting_facts: [[title, sentence_index], ...]`；
`context: [[title, [sentence0, sentence1, ...]], ...]` 用于提取对应事实句。
缺少可定位事实句会记录为未知，不能只靠标题相同宣称证据命中。
没有 ID 的行用问题文本哈希标识；重复 ID 或重复题目会停止选择，要求核对原文件。

checkpoint 若是 veRL/FSDP 分片训练存档，需你已有流程导出为可加载的完整
Hugging Face 模型目录。当前脚本不合并权重、不下载模型、不训练补齐 checkpoint。

计划目标为 HNSW。若现有索引实际是 Flat，可先用它验证工程，并记录为 Flat；
正式实验组之间必须固定相同索引。不要把 Flat 的结果描述为 HNSW 结果。

## 4. 更新代码，填写本地路径

首次拉取：

```bash
git clone --recurse-submodules https://github.com/CathyMuo/work2_byd.git
cd work2_byd
```

已经拉取过：

```bash
cd /你的仓库路径/work2_byd
bash scripts/update.sh
```

以下命令均从 `work2_byd` 根目录执行。更新脚本遇到本地代码修改会停止，
避免覆盖；`outputs/` 和 `configs/local.json` 不提交，也不会被正常 pull 覆盖。

填写路径：

```bash
cp configs/local.example.json configs/local.json
# 用你熟悉的编辑器修改 configs/local.json
```

已有 `configs/local.json` 就直接编辑，不要用模板覆盖自己的配置。

至少填写这些键：

```json
{
  "agent_checkpoint": "/本地模型目录/SearchR1-3B",
  "base_model": "/本地模型目录/Qwen2.5-3B-Instruct",
  "retriever_model": "/本地模型目录/e5-base-v2",
  "corpus_file": "/本地语料目录/wiki-18.jsonl",
  "index_file": "/本地索引目录/e5_HNSW.index",
  "hotpotqa_file": "/本地数据目录/hotpot_dev_distractor_v1.json",
  "two_wiki_file": "/本地数据目录/2wiki/dev.json",
  "retrieval_url": "http://127.0.0.1:8000/retrieve",
  "checkpoint_version": "填写 checkpoint 来源版本或已有校验清单标识",
  "corpus_version": "填写固定语料版本标识",
  "index_version": "填写固定索引版本标识",
  "actual_index_type": "填写真实索引类型，如 HNSW 或 Flat"
}
```

上面的路径都是占位符，必须换成你的真实路径。
`inference_url` 在当前本地 vLLM/HF 入口中不使用；无需另起 OpenAI 推理服务。
未跑基础模型对照时，`base_model` 无需已就绪；A1 主模型入口读取 `agent_checkpoint`。

版本标识需要实际填写，不能原样保留说明文字。记录文件大小和配置哈希只是一种
工程溯源，不等于权重/语料/索引的完整内容校验。现有服务加载的资源也要和记录一致。

## 5. 一次性固定 200 题 Pilot 样本

在已有 Search-R1 环境执行：

```bash
conda activate searchr1
python -m work2.prepare_pilot \
  --local configs/local.json \
  --output outputs/samples \
  --count-per-dataset 100 \
  --seed 42
```

这里仅从你的本地文件读取、选择和整理现有题目，不补数据。
每个数据集按唯一 ID 排序，再按固定种子打乱选择，各取 100 题。
选择发生在任何模型结果之前，不挑选“模型答对的题”。

产物：

| 文件 | 内容 |
| --- | --- |
| `outputs/samples/pilot.jsonl` | HotpotQA 100 + 2Wiki 100 条规范化题目及已有标签 |
| `outputs/samples/selection_manifest.json` | 源文件哈希、选定 ID、种子、样本文件哈希 |

脚本不会覆盖已有样本目录。后续小样本、主基线、基础模型对照及未来干预应沿用
同一个 `pilot.jsonl`。不要因为某轮结果不好就重新抽题。
这批题目属于探索 Pilot；尚未建立独立确认集，也未完成与训练题/测试题的重叠审计。

## 6. 复用或启动本地 Retriever

### 6.1 已有服务能用时直接复用

把 `retrieval_url` 改成实际地址。服务需要接受：

```json
{"queries": ["example query"], "topk": 20, "return_scores": true}
```

返回 Search-R1 schema：`{"result": [[{"document": {"contents": "title\ntext"}, "score": 0.1}, ...]]}`。
`contents` 为必需字段，文档 ID 缺失时记录内容哈希，并标注 ID 的来源。
模型只看到前 3 篇；20 篇候选留作审计及未来干预候选，不全部注入模型。

### 6.2 需要启动服务时

在另一个终端，进入同一仓库，激活你已有的 Retriever 环境；环境名按你的实际情况改。

```bash
conda activate retriever
CUDA_VISIBLE_DEVICES=1 python -m work2.launch_retriever \
  --local configs/local.json
```

这个入口只传递路径并启动固定版 Search-R1 服务，不安装任何依赖。
默认 FAISS 索引在 CPU 上，但**上游 E5 编码器仍调用 CUDA**，因此这里留出 GPU 1。
如需要沿用 GPU FAISS，可追加 `--faiss-gpu`；不要为 HNSW 强行转 GPU。
本包装器沿用上游固定端口 8000；若已有其他端口服务，直接复用并填写对应 URL。
启动失败或端口被占用时先解决，不要同时重复启动两个服务。

GPU 0 留给 Agent；GPU 2、3 先不占用。四卡并非四个实验必须同时并行的要求。

## 7. 检查本次输入和服务接口

```bash
conda activate searchr1
python -m work2.run_baseline \
  --local configs/local.json \
  --samples outputs/samples/pilot.jsonl \
  --per-dataset-limit 5 \
  --check-only
```

这一步仅检查样本结构、本地模型目录和一次检索返回格式，不加载 Agent、不做模型推理。
看到 `Input/service check passed` 后进入下一步。
它也不校验模型权重是否完整、不确认服务是否真的加载了你声明的索引。
该检查针对本次实验输入，**不是要求重新配置环境或反馈环境检查报告**。

## 8. 先跑 10 题小样本：每个数据集前 5 题

推荐先用已有 vLLM：

```bash
conda activate searchr1
mkdir -p outputs/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a1.sh smoke \
  --backend vllm \
  --local configs/local.json \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  2>&1 | tee outputs/logs/a1_smoke.log
```

`smoke` 固定读取已选 Pilot 中每个数据集前 5 题，默认写到 `outputs/a1_smoke/`。
这是实际模型实验，不是模拟运行。所有模型加载仅使用本地目录。

如果 vLLM 与你已有环境存在兼容问题，可在原有 Transformers 环境使用：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a1.sh smoke \
  --backend hf \
  --run-dir outputs/a1_smoke_hf \
  2>&1 | tee outputs/logs/a1_smoke_hf.log
```

HF 入口使用本地模型、BF16、greedy decoding；不实现 vLLM tensor parallelism。
vLLM/HF 数值输出不保证相同；后续成对实验必须固定一种 backend，不能交叉拼接配对。

### 8.1 小样本后先看哪些文件

```bash
python -m json.tool outputs/a1_smoke/baseline_metrics.json
python -m work2.inspect_baseline --run-dir outputs/a1_smoke --count 6
```

采用 HF 时把目录换成对应 HF 目录。`inspect_baseline` 只导出真实案例，优先失败案例，
不是随机研究样本，也不能用它估计总体比例。

先核对：

1. `run_manifest.json` 的 `status` 为 `completed`，题目数为 10，基础设施错误为 0。
2. 能否看到正常 `<search>...</search>` 和 `<answer>...</answer>`。
3. 有多少题执行至少 1 次搜索、至少 2 次搜索。
4. `generation_limit`、`context_limit` 是否频繁，模型是否耗尽搜索预算。
5. 文档标题/正文是否正常，检索内容是否来自预期语料。
6. 每个有效搜索是否存下原始 token 前缀、实际 Query、20 候选和 3 可见文档。

10 题只能检查工程和发现明显故障，不能用来决定创新性、效果显著性或是否进入 RL。
有失败时把日志和指标反馈给我，我修代码后推送；不要求你在厂内改源代码。

## 9. 小样本确认可用后跑 200 题 clean baseline

```bash
conda activate searchr1
set -o pipefail
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a1.sh pilot \
  --backend vllm \
  --local configs/local.json \
  2>&1 | tee outputs/logs/a1_clean_200.log
```

`pilot` 在已冻结的样本文件上运行全部 200 题，默认输出到 `outputs/a1_clean_200/`。
这里的 Pilot 仍是 **E0 clean**，没有实施 E1–E4。
若使用 HF，小样本和 200 题均使用 `--backend hf`，保持模型、Retriever、预算一致。
脚本顺序执行题目，优先保证审计性；当前没有多卡分片或批量异步提速。

运行参数来自 `configs/diagnostic.json`：

| 参数 | 初始值 |
| --- | --- |
| 解码 | temperature 0；seed 42 |
| 可见文档 / 候选数 | 3 / 20 |
| 成功搜索预算 | 最多 4 次，第四次返回后仍允许生成答案 |
| 上下文 | 8192 tokens，达到边界会记录终止，不静默裁掉历史 |
| 每段生成 | 最多 512 tokens，到上限未形成有效动作会记录失败 |

Prompt 参考 Search-R1 `infer.py` 的工具标签协议，但明确加入 4 次搜索预算，
属于本仓库受控诊断 harness，不是完全复刻官方训练评测流水线。
不要把本次分数直接等同于上游论文分数。

如果要改变预算、生成上限或索引声明，应复制设计配置到一个忽略的本地文件，
用 `--design configs/local.diagnostic.json` 指定，另建 run 目录并注明原因。
不要对同一组结果偷偷更改配置，也不要在看到效果后自动搜索更好的设置。

## 10. 可选：未做 Search RL 的基础模型对照

在主基线能正常执行后，基础模型已在本地时可运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m work2.run_baseline \
  --local configs/local.json \
  --samples outputs/samples/pilot.jsonl \
  --checkpoint-key base_model \
  --backend vllm \
  --run-dir outputs/a1_base_model_200 \
  2>&1 | tee outputs/logs/a1_base_model_200.log
```

两组使用同一批问题、Retriever、预算和 backend。
比较 EM/F1、有效答案率、多轮率，辨别问题是否只是模型缺少工具调用能力。
该对照不需要新训练，也不能替代之后的 E0/E1/E2/E3/E4 配对干预。

## 11. 输出内容及指标解释

| 文件 | 用途 |
| --- | --- |
| `run_manifest.json` | 本次 Git commit、依赖 commit、配置、样本 ID/哈希、资源标识、backend、包版本和完成状态 |
| `clean_rollouts.jsonl` | 逐题答案、EM/F1、终止原因、生成片段、tokens、实际搜索及检索前缀 |
| `baseline_metrics.json` | 总体和各数据集的答案/工具/预算指标及样本分母 |
| `review_cases.json` | 执行案例导出命令后产生的有限真实案例 |

`prefixes` 中保存 `prefix_text`、完整 `prefix_token_ids`、token 哈希、Query、
候选文档、可见文档和 observation。后续 replay 必须沿用保存的 token 前缀，
不能从头重跑同一道题当作“同前缀”实验。

EM/F1 使用大小写、标点、英文冠词和空白标准化；yes/no/noanswer 按严格匹配处理。
脚本按原始数据中的单个字符串答案评估，不自动增加别名答案。
统计以全部模型有效执行的题为分母；没有合规答案的模型失败计 0 分，不删除难题。
基础设施错误单列并停止批次，部分结果不能当作完整 200 题结果。
`multi_turn_rate` 的分母是该组可评估题目，不是仅已经搜索过的题。

`visible_lexical_fact_ids` 只表示“标题匹配且已有支持句文本出现在可见文档中”的
保守文本代理。不是完整语义证据验证，不证明语料覆盖率、桥接因果效用或 Recovery。
候选中的事实命中也不表示 Agent 实际看到了该事实。

## 12. 结果如何反馈

直接反馈指标、run manifest、日志和少量真实案例即可，不用 git push。
无需运行环境检查脚本。可选打包命令：

```bash
python -m work2.inspect_baseline --run-dir outputs/a1_clean_200 --count 6
python scripts/pack_feedback.py \
  --include outputs/a1_clean_200/run_manifest.json \
  --include outputs/a1_clean_200/baseline_metrics.json \
  --include outputs/a1_clean_200/review_cases.json \
  --include outputs/samples/selection_manifest.json \
  --include outputs/logs/a1_clean_200.log
```

压缩包位于 `feedback/`。每个选定文件上限 10 MiB；超大日志请直接反馈关键报错。
全文 `clean_rollouts.jsonl` 留在厂内，当前打包工具不包含 JSONL 原始轨迹，
需要时从中导出具体案例。模型、数据、凭据和本地路径配置不提交公开仓库。

## 13. 失败后怎么处理

| 情况 | 处理 |
| --- | --- |
| `run-dir already exists` | 保留原目录，用 `--run-dir outputs/a1_smoke_v2` 指定新目录 |
| 模型加载失败 | 反馈完整报错及 manifest；检查是否本地完整 HF 模型、tokenizer 和权重 |
| OOM | 先确认 Agent 与 Retriever 不争同一卡；报告后再决定降低占用/调整运行参数 |
| 连接拒绝或超时 | 检查已配置的服务地址、端口与服务终端；可用 `--request-timeout` 增大合理等待时间 |
| response schema 不匹配 | 反馈实际检索响应结构；由我改适配器，不要求厂内改代码 |
| 工具标签格式错误 | 查看真实生成文本；不加规则伪造工具调用或答案 |
| 多轮率极低 | 检查 checkpoint 是否确实完成 Search-R1 训练，再分析任务/预算；不直接进入扰动实验 |
| 频繁 `generation_limit` / `context_limit` | 反馈比例和案例，先审计截断，再决定是否修改初始预算 |
| Ctrl-C、服务或推理异常 | manifest 记为 failed，保留已写出的轨迹；当前不提供自动续跑，修复后用新目录重新执行 |

自动续跑可能把不同服务状态的轨迹混在一个 run 中，首版选择显式新建目录。
没有合规 `<answer>` 时不把自由文本猜成答案。

## 14. A1 之后怎样验证研究问题（后续实施计划）

以下是研究执行顺序，不是当前已存在的可运行命令。

### A2：同前缀重放自检

从 A1 保存的第一/第二轮检索前状态恢复生成。先对 10 题做 E0/E0：同一 token
前缀、同一 observation 和解码配置，核对续写差异；再做 E0/E1，保持同一文档
集合，仅重排一次。第二轮只纳入真正到达该轮的轨迹，分母单列。
需要实现 replay 入口后才能运行；A1 保存前缀本身不代表 replay 已验证。

### A3：单轮干预 Pilot

固定前缀后，各条件只改一次工具 observation，后续 Retriever 恢复正常。
E2 从同 Query 第 4–20 候选选真实相关干扰文档替换非关键文档；
E3 仅在原 Top-3 里有经事实核验的桥接证据时移除/替换；
E4 引入冗余且保留关键事实。无法判定事实或无法找到合格候选时记录 skipped，
不强行给标签。对文档长度、实际文本和集合差异保留审计。

分别报告 E0→E1/E2/E3/E4 的答案退化及反向改善、后续搜索子目标、
Recovery@1/2/3、成本和预算耗尽率。OffTarget 需要独立事实/实体子目标核验，
不能只看 Query 文本差异。采用配对不确定性估计和案例复核，另用独立确认集验证。

### B：检查简单恢复方法能否解决

比较 4→6 次预算、去重/过滤、回顾未完成子目标的 Prompt、可比的 Harness-G，
以及明确标注的 Oracle 分析上界。比较时报告 tokens、检索访问量和有效证据，
不能把不同工具接口的一次调用简单当成等成本。

### C：只有 GO 后进入 RL

必须有证据支持：E2/E3 的问题超过 E1 排序效应、存在跨轮行为差异、
简单基线仍不能消除主要差距、扰动状态中确实存在有效恢复动作。
再固定训练/确认/测试隔离及对照组，研究成对轨迹的恢复动作信用分配。
单纯跑完 A1 或建好环境都不构成 GO。

**当前厂内顺序：更新代码 → 填已有路径 → 固定题目 → 复用/启动 Retriever
→ 10 题小样本 → 200 题 clean baseline → 反馈 → 我实现下一阶段并 push。**

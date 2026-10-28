**2026-10-27 U1 / U2 / HR2 已完成**：H1 在 7/7 个智能体上确认，H3 未确认（降为次要分析），U2 两个已发表方法都确认 H1，HR2 只有 PPO 有效。见 [结果与决策](research/U1_U2_HR2_RESULTS_AND_DECISION.md)。
**进度看板**：`bash scripts/run_board.sh` 启动后打开 `http://127.0.0.1:8765/`，可以看研究说明（给导师汇报用）和实时实验进度。见 [看板使用说明](docs/BOARD_GUIDE.md)。

**2026-10-28 U1x 完成**：H1 在 3/3 个可测的非 Search-R1 智能体上成立；AutoRefine 无法用插入文字的方法测量。见 [U1x 结果与决策](research/U1X_RESULTS_AND_DECISION.md)。
厂内当前任务：[U1x 后续统计（只用 CPU）](docs/U1X_FOLLOWUP_GUIDE.md)。已完成：[论文表格](docs/PAPER_TABLES_GUIDE.md)、[U1x](docs/U1X_FACTORY_GUIDE.md)。

**2026-10-23 U1 之后的两项（代码与指导已就绪）：**
- **U2 方法审计**：用同状态安慰剂检验 ReSeek（强制 judge）和 MR-Search（反思回合）自己的恢复机制。厂内从 [U2 厂内执行指导](docs/U2_FACTORY_GUIDE.md) 开始，协议见 [U2 协议](research/U2_PROTOCOL.md)。
- **HR2 交接重放**：在 v0.3 7B PPO / GRPO 上重做 HR1，新增 `evonly_placebo` 臂。厂内从 [HR2 厂内执行指导](docs/HR2_FACTORY_GUIDE.md) 开始，协议见 [HR2 协议](research/HR2_PROTOCOL.md)。
- 顺序：U1 → U2 → HR2。之前回传里的“GRPO 训练”是用户的显卡保活程序，不是研究任务；开跑前确认四卡空闲即可。

**2026-10-22 冲主会升级 U1（目标 ACL 2027 主会）：** 把 H1 / H3 的审计扩到 7 个官方 Search-R1 智能体 × 5 个数据集（HotpotQA、2Wiki、NQ、MuSiQue、Bamboogle）。
FC1 已完成（10-23）：U1 用 `bounded` 提示词；主 7B 换成 v0.3 7B PPO，详见 [FC1 结果与决策](research/FC1_RESULTS_AND_DECISION.md)。厂内从 [U1 厂内执行指导](docs/U1_FACTORY_GUIDE.md) 开始，协议见 [U1 协议](research/U1_PROTOCOL.md)。
**2026-10-20 当前任务：FC1 评测保真度检查。** HR1 交接重放判 NO-GO（R 相对 Instruct 无增益），并暴露出我们的 7B 是 Search-R1 v0.1、提示词也偏离了官方模板。
写 (a) 之前，先比较各检查点（v0.1 / v0.3、PPO / GRPO、Instruct）在两种提示词下的 EM。厂内从 [FC1 厂内执行指导](docs/FC1_FACTORY_GUIDE.md) 开始，协议见 [FC1 协议](research/FIDELITY_PROTOCOL.md)。
HR1 结果见 [HR1 结果与决策](research/HR1_RESULTS_AND_DECISION.md)。
# Work 2: Interventional Replay Audits of RL-Trained Search Agents（AC1）

**2026-10-19 并行任务：HR1 交接重放 pilot（方向 (b)-A）。** Qwen2.5-7B-Instruct 与 Search-R1 7B PPO 从同一前缀互相交接，
把 RL 增益拆成“获取证据 / 利用证据 / 与证据无关”三部分。厂内从 [HR1 厂内执行指导](docs/HR1_FACTORY_GUIDE.md) 开始，协议见 [HR1 协议](research/HANDOFF_PROTOCOL.md)。
AC1 主检查点结果：H1、H3 确认，H2 未确认，见 [AC1 结果与决策](research/AC1_RESULTS_AND_DECISION.md)。
**2026-10-19 当前任务，分支 `feat/credit-testbed`：AC1 确认实验。** 在从未用过的新题和第二个检查点上，确认三条干预式审计主张：
恢复收益主要是均值回归、证据有因果作用、证据类信用信号不测证据。
厂内 agent 从 [AC1 厂内执行指导](docs/AC1_FACTORY_GUIDE.md) 开始；协议见 [AC1 协议](research/AUDIT_CONFIRM_PROTOCOL.md)。

```bash
bash scripts/run_audit.sh test && bash scripts/run_audit.sh freeze
AC_LABEL=7b_ppo bash scripts/run_audit.sh all
AC_LABEL=3b_ppo AC_LOCAL=configs/local.second.json bash scripts/run_audit.sh all
```

CT1（逐轮信用检验）已于 2026-10-19 判 NO-GO（A 两次信度不足；且被 2608.19760 / 2609.02417 抢先），结果见 [CT1 结果与决策](research/CT1_PILOT_RESULTS_AND_DECISION.md)。
PCR（路径运气）已于 2026-10-16 按预登记 G1 NO-GO 归档，结果见 [PCR 结果与决策](research/PCR_PHASE1_RESULTS_AND_DECISION.md)。
以下 SI1 / E / A 系列为已归档的历史方向，保留供复核。

---

**2026-10-13 SI1 CPU 诊断已报告完成：当前 ridge 原型归档 NO-GO。**
选择器 gain 3 / harm 5，同触发率随机 gain 4 / harm 5；未建立选择优势。
没有下一轮必跑任务，不调已观察验证集、不启动 SI2、确认或 RL。
详见 [CPU 诊断与归档决策](report/SI1_CPU_DIAGNOSTICS_AND_NO_GO_20261013.md)。
以下四卡命令和执行文档为历史协议，保留供复核，不要求重跑。

**四卡扩容更新：默认1200训练+800验证，四个独立TP=1 worker并行。**
直接看[四卡2000题运行清单](docs/SI1_DP4_QUICKSTART.md)。
全部原样控制通过后才启动干预，四卡完整核验后统一合并；旧结果保留。

**2026-10-12 新分支 `feat/selective-intervention`：用户授权启动独立SI1开发探针。**
方向为自然检索状态上的相对gain/harm学习与选择性干预，提供样本冻结、四臂续跑、
CPU审计、轻量选择器拟合、先选择后评价和反馈打包。
从[SI1厂内执行指导](docs/SI1_FACTORY_GUIDE.md)开始；
方法阅读见[近邻阅读与取舍](research/SI1_LITERATURE_AND_DIRECTION.md)，
协议见[SI1计划](research/SI1_PROTOCOL.md)。旧约束方法与E分支仍归档，下面为历史进度。

工作二独立研究仓库：多轮搜索智能体中的检索反馈偏移与自适应恢复。
实验服务器：厂内 **4×H800**。厂内已报告 1000 题 A1、2000 个 A3 目标状态及 B
基线结果；新实验与代码分支 `f4a4bb5` 尚未提供原始源码/轨迹。
厂内最新 `8ceeda1` 已报告 E v2 收尾全部通过：8 案例机械审计、5 个事件案例
人工解释完成，无新 GPU。当前方法分支 **归档 NO-GO**，没有未完成的 E 必跑任务。
不重跑审计、不补无事件解释、不进入 F/RL，不新增提示变体或分片实验。
详见 [归档后安排](docs/POST_ARCHIVE_NEXT_STEPS.md) 与
[结果深度分析](research/E_V2_RESULTS_AND_DIRECTION.md)。
全部尝试性试验和有效结论已汇总至 [report/](report/README.md)。

A1 的原 200 题 clean baseline 已由厂内报告完成，
实际 cohort 是官方 Search-R1 **7B + FAISS Flat**，EM 0.335、多轮率 0.975。
详见 [A1 报告分析](research/A1_REPORTED_RESULTS.md)；原始数据尚未在开发服务器核验。
厂内报告 A2 的 20/20 精确控制通过，见 [A2 分析](research/A2_REPORTED_RESULTS.md)。
当前已实现独立审查后执行的 E2–E4 **10 题工程 Pilot**；完整诊断扩展、恢复基线和 RL 待后续。

**历史约束方法已归档；SI1不重跑其E实验，保留厂内备份与所有原始产物。**
原 [A3 工程文档](docs/A3_EXECUTION_GUIDE.md) 保留为历史工程入口。

详细步骤见 [工作二实验实施文档](docs/EXECUTION_GUIDE.md)：资源清单、路径填写、
Retriever 启动、10 题小样本、200 题基线、结果含义及反馈方式均有命令和说明。

## 厂内首次拉取

```bash
git clone --recurse-submodules https://github.com/CathyMuo/work2_byd.git
cd work2_byd
```

厂内已经配好 Search-R1 环境，直接沿用。环境由你维护，我负责实验代码、
运行说明与推荐配置清单，见 [推荐环境配置](docs/RECOMMENDED_ENVIRONMENT.md)。
后续代码开发不以环境检查或报告回传为前置条件。

运行实验时填写本机模型、语料和索引路径：

```bash
cp configs/local.example.json configs/local.json
# 编辑 configs/local.json 的真实路径；该文件不会提交
```

结果和日志直接反馈到对话，不需要厂内 git push。仓库中保留的
`check_environment.py` 和 `pack_feedback.py` 仅供需要排障时自愿使用，
不属于必跑步骤。

## 当前状态：归档完成

没有必跑 E 命令。旧脚本保留作历史复核，勿将下面的历史入口视为新实验指令。
下一步见 [归档后安排](docs/POST_ARCHIVE_NEXT_STEPS.md)：旧 B 记录可定位时做一次
46/52/3 状态 ID 集合对账，否则保持 unresolved；新候选先做独立文献/假设/协议
评审，不自动启动预算控制、F 或 RL。generic 与 replay 平台原样保留。

## 历史 CPU 配对与语义审查入口

首次从厂内分叉切换主线需按 [对齐步骤](docs/FACTORY_ALIGNMENT_AND_NEXT_STEPS.md)，
普通 git pull 无法快进覆盖已提交分支。之后使用实际 raw 文件：

```bash
python -m work2.audit_pairs \
  --a1-rollouts outputs/a1_clean_scale500/clean_rollouts.jsonl \
  --a3-rollouts outputs/a3_scale500_merged/paired_rollouts.jsonl \
  --output-dir outputs/semantic_audit
```

A3 实际文件名/字段以厂内为准，先用 `--schema-only` 核对。格式不一致明确停止，
不猜字段、不伪造控制。新程序给同资格状态 E1 配对、整题聚类区间、全部退化
事件的证据审查表；语义判断仍由独立审查完成。B 使用显式 compact 转移格式
核对完整 gain/harm/net，不自动读取未知旧 schema。

## 原 A3 工程 Pilot 入口

沿用现有 `configs/local.json`、常驻 Retriever 和 A1 产物：

```bash
python -m work2.prepare_interventions
cp outputs/a3_review/proposals.json outputs/a3_review/reviewed.json
# 按 A3 文档逐条完成 reviews，不能保留 pending 或直接全部勾选：
python -m work2.prepare_interventions --validate outputs/a3_review/reviewed.json
python -m work2.run_interventions --check-only
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a3.sh
```

选择 A1 冻结顺序中每个数据集前 5 题，目标轮 1/2；资格不足保留 skip 和分母。
原始 Top-3 只替换一篇，使用同 Query 原始 rank 4–20 的真实候选。先跑精确
控制，失败整批停止。旧 A2 的 Flat→hnsw 标签误判只在新产物中勘误，不改历史。
语义 OffTarget/Recovery 和正式统计仍需下一轮协议；此轮不是研究/RL GO。
A2 入口仍保留在 [A2 文档](docs/A2_EXECUTION_GUIDE.md)，本 cohort 无需重跑。

## A1 入口：保留供独立新 cohort 使用

填好已有资源路径后，按实施文档复用或启动检索服务，再执行：

```bash
python -m work2.prepare_pilot --local configs/local.json
python -m work2.run_baseline --check-only --per-dataset-limit 5
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a1.sh smoke
# 检查小样本结果后执行：
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a1.sh pilot
```

这两批运行都是 E0 clean，不包含干预。正式判断多轮搜索偏移与恢复需要后续
阶段的同前缀配对实验；普通 clean baseline 分数不能替代这些证据。

## 后续更新

```bash
cd work2_byd
bash scripts/update.sh
```

我在当前开发服务器修改、检查并 push；你在厂内 pull、按当次运行说明执行，
再反馈日志和结果。我据此修改后继续 push。每次反馈保留 Git commit，
这样结果能对应到确定版本。厂内只需公开仓库读取权限。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `third_party/Search-R1` | 主底座，包含上游修改的 veRL、检索服务与训练代码 |
| `third_party/Stable-RAG` | 文档排序敏感性对照参考 |
| `third_party/Harness-G` | 结构化搜索强基线参考 |
| `third_party/SearchAtlas` | 搜索轨迹解析与诊断参考 |
| `upstreams.lock.json` | 四个上游仓库的固定 commit 和用途 |
| `configs/diagnostic.json` | 阶段 A 的设计配置 |
| `configs/local.example.json` | 厂内路径模板 |
| `research/` | 实验计划、状态、决策记录 |
| `report/` | 各轮试验结果、结论修正、NO-GO 与来源/产物总档案 |
| `scripts/` | 环境检查、更新和反馈打包工具 |
| `work2/` | 本地数据读取、A1 rollout、推理适配、指标与案例导出 |
| `tests/` | CPU 协议、边界和运行产物测试 |
| `docs/EXECUTION_GUIDE.md` | 详细厂内实施步骤与后续研究路线 |
| `docs/A2_EXECUTION_GUIDE.md` | A2 的重放步骤、控制与反馈要求 |
| `docs/A3_EXECUTION_GUIDE.md` | 原 E2–E4 工程资格审查、冻结和单轮干预步骤 |
| `docs/FACTORY_ALIGNMENT_AND_NEXT_STEPS.md` | 当前厂内分叉备份/对齐、CPU 审计与后续机制协议 |
| `docs/POST_ARCHIVE_NEXT_STEPS.md` | 当前归档状态、一次性历史对账与独立立项评审 |
| `research/E_V2_RESULTS_AND_DIRECTION.md` | 方法 NO-GO、结果分析与已完成的 CPU 收尾协议 |
| `docs/E_V2_STRATIFIED_PROBE.md` | 历史 v2 两层问题、来源/前缀审查与共用执行引擎 |
| `docs/E_EXECUTION_GUIDE.md` | 历史 v1 充分配对、实际扰动起点与成本口径 |
| `docs/CD_REVIEW_AND_MECHANISM_PROTOCOL.md` | 最新 C/D 正文诊断、审查/机制收据与 E/F 的范围 |
| `outputs/` | 本机环境报告与未来实验产物，不提交 |

上游以 Git submodule 固定版本，均使用公开 HTTPS URL。不要在厂内执行
`git submodule update --remote`，否则会偏离本次实验的固定版本。
第三方代码遵循各自仓库许可证；新代码在本仓库实现，依赖默认保持不变。

## 当前研究路线

来自[之前的讨论](https://chatgpt.com/share/6ac84bc4-3cd4-83ea-8a65-ec7b03c325c5)：
先确认 Search-R1 checkpoint 的有效工具调用和多轮轨迹，随后做相同前缀的
E0/E0 与 E0/E1 重放，再做 HotpotQA、2Wiki 各 100 题的受控干预 Pilot。
完成恢复基线和 GO/NO-GO 后，才考虑新 RL 方法。

模型、Wiki-18、FAISS 索引和数据集需要厂内本地准备。四个上游的依赖版本
可能不同，初始化阶段不把它们全部安装到同一环境。

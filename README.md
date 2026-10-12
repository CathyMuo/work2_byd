# Work 2: Search Feedback Drift and Recovery

工作二独立研究仓库：多轮搜索智能体中的检索反馈偏移与自适应恢复。
实验服务器：厂内 **4×H800**。厂内已报告 1000 题 A1、2000 个 A3 目标状态及 B
基线结果；新实验与代码分支 `f4a4bb5` 尚未提供原始源码/轨迹。
当前已接受厂内 `0ea168a` 的 **8/175 事件部分校准**，保留 167 pending，
已实现唯一公开的 **E 实际扰动历史续写入口**。先独立审查至少 1 对失败/稳定
案例，跑精确控制优先的 smoke，再最多 4 对工程扩展。
完整命令、起点规则与最新计数修正见 [E 实施文档](docs/E_EXECUTION_GUIDE.md)。
[C/D 审查协议](docs/CD_REVIEW_AND_MECHANISM_PROTOCOL.md) 保留为审计入口，
正文重叠不自动证明充分性；研究与 RL 仍未 GO。
详见 [扩大结果审查](research/FACTORY_SCALE_REVIEW.md) 与
[厂内代码对齐步骤](docs/FACTORY_ALIGNMENT_AND_NEXT_STEPS.md)。

A1 的原 200 题 clean baseline 已由厂内报告完成，
实际 cohort 是官方 Search-R1 **7B + FAISS Flat**，EM 0.335、多轮率 0.975。
详见 [A1 报告分析](research/A1_REPORTED_RESULTS.md)；原始数据尚未在开发服务器核验。
厂内报告 A2 的 20/20 精确控制通过，见 [A2 分析](research/A2_REPORTED_RESULTS.md)。
当前已实现独立审查后执行的 E2–E4 **10 题工程 Pilot**；完整诊断扩展、恢复基线和 RL 待后续。

**本轮先保存厂内分叉再对齐主线，准备/核验 E 配对；不重跑规模实验。**
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

## 当前入口：E 配对与工程 smoke

按 [E 实施文档](docs/E_EXECUTION_GUIDE.md) 对齐公开主线后：

```bash
python -m work2.prepare_e
cp outputs/e_review/proposals.json outputs/e_review/reviewed.json
# 独立核验实际截断前缀，填写所有组的 run/skip 与必要 reviews：
python -m work2.prepare_e --validate outputs/e_review/reviewed.json
python -m work2.run_e --pair-limit 1 --check-only
CUDA_VISIBLE_DEVICES=0 bash scripts/run_e.sh --pair-limit 1 --run-dir outputs/e_smoke
```

至少一组完整失败/稳定配对，保留原干预和已用预算；全批原样控制与历史后缀
一致后才运行一般回顾/原问题约束核对。现有校准不能替代稳定控制的独立审查。

## CPU 配对与语义审查入口

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
| `scripts/` | 环境检查、更新和反馈打包工具 |
| `work2/` | 本地数据读取、A1 rollout、推理适配、指标与案例导出 |
| `tests/` | CPU 协议、边界和运行产物测试 |
| `docs/EXECUTION_GUIDE.md` | 详细厂内实施步骤与后续研究路线 |
| `docs/A2_EXECUTION_GUIDE.md` | A2 的重放步骤、控制与反馈要求 |
| `docs/A3_EXECUTION_GUIDE.md` | 原 E2–E4 工程资格审查、冻结和单轮干预步骤 |
| `docs/FACTORY_ALIGNMENT_AND_NEXT_STEPS.md` | 当前厂内分叉备份/对齐、CPU 审计与后续机制协议 |
| `docs/E_EXECUTION_GUIDE.md` | 当前 E 候选审查、实际扰动起点、全批控制优先和成本/结果反馈 |
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

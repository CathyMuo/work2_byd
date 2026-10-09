# Work 2: Search Feedback Drift and Recovery

工作二独立研究仓库：多轮搜索智能体中的检索反馈偏移与自适应恢复。
实验服务器：厂内 **4×H800**。A1 的 200 题 clean baseline 已由厂内报告完成，
实际 cohort 是官方 Search-R1 **7B + FAISS Flat**，EM 0.335、多轮率 0.975。
详见 [A1 报告分析](research/A1_REPORTED_RESULTS.md)；原始数据尚未在开发服务器核验。
当前新增 A2 同前缀 E0/E0 与 E0/E1 入口，待厂内运行；E2–E4、恢复基线和 RL 尚未实现。

**本轮请执行 [A2 实施文档](docs/A2_EXECUTION_GUIDE.md)，保持原模型/索引/题目，
不要重新准备数据或重跑 A1。**

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

## 当前实验入口：A2

沿用现有 `configs/local.json`、常驻 Retriever 和 A1 产物：

```bash
python -m work2.run_replay --source-run outputs/a1_clean_200 --check-only
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a2.sh
```

默认选择 A1 冻结顺序中每个数据集前 5 题，目标轮 1/2。第二轮不可达单列，
不筛掉答错题。源记录、资源或服务不一致会停止；E0/E0 不一致时不继续 E1。
通过仅表示可提交 A3 工程审查，不能视为检索偏移已证实或 RL GO。

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
| `docs/A2_EXECUTION_GUIDE.md` | 当前 A2 的重放步骤、控制与反馈要求 |
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

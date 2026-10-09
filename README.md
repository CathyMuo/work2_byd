# Work 2: Search Feedback Drift and Recovery

工作二独立研究仓库：多轮搜索智能体中的检索反馈偏移与自适应恢复。
实验服务器：厂内 **4×H800**。当前版本完成仓库、依赖与协作流程初始化；
尚未实现正式诊断实验，也没有实验结果。

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

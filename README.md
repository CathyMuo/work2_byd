# Work 2: Search Feedback Drift and Recovery

工作二独立研究仓库：多轮搜索智能体中的检索反馈偏移与自适应恢复。
实验服务器：厂内 **4×H800**。当前版本完成仓库、依赖与协作流程初始化；
尚未实现正式诊断实验，也没有实验结果。

## 厂内首次拉取

```bash
git clone --recurse-submodules https://github.com/CathyMuo/work2_byd.git
cd work2_byd
python3 scripts/check_environment.py
```

上述检查只依赖 Python 标准库，不安装包、不下载模型、不启动 GPU 实验。
把 `outputs/environment/environment.json` 和终端中的检查结论反馈给我。
缺少依赖时也会生成报告。先根据报告确定版本，再安装实验依赖。

如已有模型和语料，请填写本机路径后再检查：

```bash
cp configs/local.example.json configs/local.json
# 编辑 configs/local.json 的真实路径；该文件不会提交
python3 scripts/check_environment.py --config configs/local.json
python3 scripts/pack_feedback.py
```

`feedback/` 中的压缩包可以直接交回当前对话，不需要厂内 git push。
默认只包含环境报告；实验结果生成后可通过 `--include` 指定汇总 JSON 或日志。
原始数据、模型、凭据和本地配置不打包。

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

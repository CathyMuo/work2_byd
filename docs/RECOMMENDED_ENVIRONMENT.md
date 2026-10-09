# 推荐环境配置清单

厂内已有可用的 Search-R1 环境，优先直接沿用。下面是本仓库固定版本的
兼容参考，不是重装要求，也不要求提交环境检查报告。执行环境由用户维护。

## Search-R1 推理 / 训练环境

| 项目 | 推荐参考 | 说明 |
| --- | --- | --- |
| Python | 3.9 | 固定版 Search-R1 的主环境示例 |
| PyTorch | 2.4.0，CUDA 12.1 构建 | 与下述 vLLM 参考组合配套 |
| vLLM | 0.6.3 | 固定版上游参考；已有可运行版本优先保留 |
| Transformers | 已有兼容版本，`<4.48` | 遵循固定版 Search-R1 requirements |
| veRL | Search-R1 自带修改版 | 不用最新 veRL 覆盖已有可用底座 |
| FlashAttention | 与现有 torch/CUDA 匹配的 FlashAttention 2 | 保留厂内已验证的构建 |
| 其他依赖 | 已有 Search-R1 依赖 | 包括 ray、datasets、hydra-core、tensordict 等；固定版要求 `tensordict<0.6` |
| NVIDIA 驱动 | 支持现有 CUDA 运行时 | 系统 Toolkit、torch 运行时、vLLM 编译版本需兼容，不按版本显示字符串简单判断 |
| GPU | 厂内 4×H800 | 诊断阶段初始 TP=1；四卡不要求同时占满 |

上述主环境组合来自[本仓库固定的 Search-R1 安装说明](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/README.md#installation)。
依赖边界见[固定版 requirements.txt](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/requirements.txt)。
vLLM 二进制与 CUDA/PyTorch 配套说明见[vLLM 0.6.3 文档](https://docs.vllm.ai/en/v0.6.3/getting_started/installation.html)。
这些是上游兼容参考，尚未在厂内执行验证；已有可用环境无需为了对齐清单而升级或降级。

## Retriever 参考

| 项目 | 推荐参考 |
| --- | --- |
| Python | 独立 Retriever 环境可参考 3.10；已有环境沿用 |
| 编码器 | 本地 E5-base-v2，固定模型版本 |
| PyTorch | 如重建 Retriever 环境，上游参考 2.4.0 / CUDA 12.1 |
| FAISS | CPU HNSW 可用于诊断；已有 GPU FAISS 也可沿用，上游 GPU 参考为 1.8.0 |
| 服务依赖 | transformers、FastAPI、uvicorn；与现有服务版本兼容 |
| 服务接口 | Search-R1 `/retrieve`，本地地址可配置 |

分环境建议及 GPU FAISS 版本来自上述 Search-R1 安装说明。
CPU HNSW 是工作二诊断计划中的工程选择；使用既有 Flat 索引时必须明确
记录索引类型，并在实验对比中固定，不能将其标成 HNSW。

## 本地实验资源和初始参数

| 项目 | 工作二初始配置 |
| --- | --- |
| Agent | 优先 Search-R1 已训练的 Qwen2.5-3B checkpoint |
| 对照模型 | Qwen2.5-3B-Instruct |
| 语料 | Wiki-18，与 FAISS 索引的 passage ID 一致 |
| 数据 | HotpotQA、2WikiMultiHopQA 的本地数据 |
| 检索 | 可见 Top-3；候选 Top-20；最多搜索 4 轮 |
| 解码 | temperature=0，seed=42 |
| Token 预算 | 上下文 8192；每轮生成上限初始为 512 |
| 精度 | 初始建议 BF16；由已有服务支持情况决定 |

模型和语料路径在 `configs/local.json` 中填写；实验设计见
`configs/diagnostic.json`。这部分属于待实现、待运行的实验设置，
不是已经测得的性能结论。

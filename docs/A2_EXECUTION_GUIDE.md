# A2：同前缀 E0/E0 与 E0/E1 详细执行步骤

日期：2026-10-09。A1 200 题已由厂内报告完成；分析见
[A1 工程验收记录](../research/A1_REPORTED_RESULTS.md)。
本次仅运行 A2 工程自检，E2–E4、恢复基线、RL 不在本次执行范围。

更新：该 cohort 的 A2 已报告完成。当前下一轮见
[A3 执行文档](A3_EXECUTION_GUIDE.md)；本文件保留 A2 复现步骤，不要求重跑。

## 1. A2 要验证什么

**我们需要证明分支来自同一真实 Agent 状态，而且只改目标轮的一次检索观测。**

每个选定问题、每个可达目标轮创建三个分支：

| 分支 | 输入 | 用途 |
| --- | --- | --- |
| clean_a | A1 原始 token 前缀 + A1 原始 Top-3 观测 | 当次重放的 clean 比较基准 |
| clean_b | 完全相同的前缀和观测，再续写一次 | E0/E0 控制，检测重放非确定性 |
| permuted_a | 相同前缀，把原 Top-3 按 `[1,2,0]` 轮换一次 | E0/E1 排序对照 |

前三份文档的 ID、正文、score 和来源都保持不变，只改变模型观察到的顺序。
之后的所有 Query 仍调用正常 Retriever，不连续施加干预、不修改索引、不缓存工具响应。
先完成两个 clean 分支；若其续写不完全一致，则停止，不执行该状态的 permutation。
不同题交替 clean_a/clean_b 执行顺序，permutation 在通过 control 后执行。

当前 E0/E0“完全一致”包括续写 token ID、生成文本、后续 Query、后续观测、终止状态
和答案。仅答案相同而其他续写不同，也记录控制失败。
另报告 clean_a 与原 A1 历史续写的一致性，作为额外溯源检查；当次配对基准始终是
新的 clean_a，不拿旧 A1 结果冒充新配对分支。

## 2. 保留哪些资源，哪些工作不用再做

沿用 A1 的官方 **7B** checkpoint、修复后的本地权重索引、E5-base-v2、Wiki-18、
**Flat** 索引、vLLM 0.6.3、BF16、temperature 0、seed 42、TP=1。
不要换成 3B，不要改成 HNSW，不要重抽 Pilot，也不要升级推理包。
不要覆盖或编辑 A1 结果目录、原 run manifest 或冻结样本文件。

A2 从 **A1 manifest** 继承全部 Agent/retrieval 数值参数，不重新套用当前
`configs/diagnostic.json`。若旧 manifest 的设计字段仍写 3B/HNSW，保留原文件作为
历史声明；实际模型来自其 checkpoint 身份，实际索引类型来自其
`declared_asset_versions.actual_index_type`。不能因修改新默认配置而改写旧实验。

需要存在的本地文件：

- `outputs/a1_clean_200/run_manifest.json`
- `outputs/a1_clean_200/baseline_metrics.json`
- `outputs/a1_clean_200/clean_rollouts.jsonl`
- A1 manifest 所指向的原 `pilot.jsonl`
- 现有 `configs/local.json`、相同模型/语料/索引，和现有正常 Retriever

这些已经由 A1 产生或在厂内存在；本次不下载、不转换、不补资源。
被中断的另一项训练不需要为本实验重启。

## 3. 选题与第二轮分母

固定取 A1 manifest 原顺序中每个数据集的前 5 题，总计 10 题。
先选题，再判断能否到达目标轮，**不筛掉答错、预算耗尽或最终没有规范答案的题**。
默认目标轮为 1 和 2，因此最多 20 个目标状态、60 条分支续写。

原轨迹未到第二次搜索时，写 `unreachable`；不会补选另一个“容易”的问题。
统计分别给出各数据集、各目标轮的计划可达分母、实际记录的分母、控制对数量、
排序对数量和未到达数量。

## 4. 更新代码，先做源文件检查

在厂内原仓库目录执行：

```bash
bash scripts/update.sh
conda activate searchr1
python -m work2.run_replay \
  --source-run outputs/a1_clean_200 \
  --per-dataset-limit 5 \
  --target-turns 1 2 \
  --check-only
```

`--check-only` 不加载模型、不调用检索服务。它核对：

1. A1 是 completed 的 E0 clean，原样本顺序和题目/标签一致。
2. 冻结样本哈希未变，原始轨迹的计数/EM/F1 与原 metrics 一致。
3. 目标前缀 token 哈希、Query、生成历史、原始文档与观测记录一致。
4. 选定 10 题及可达目标轮打印出来，便于保留选择记录。

成功时显示 `A2 source-record check passed`。完整 token 拼接重建及资源/服务验证
会在实际运行时进行，这个检查成功不代表 GPU 重放已通过。
旧路径若搬迁导致原 pilot 不可访问，保留原数据和 manifest，反馈具体错误；
不要随手改源 manifest 来绕过哈希或身份检查。

## 5. 执行 A2

继续复用现有常驻 Retriever，不重复启动、不杀进程。Agent 沿用 A1 的 GPU0。

```bash
mkdir -p outputs/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a2.sh \
  --local configs/local.json \
  2>&1 | tee outputs/logs/a2_sanity.log
```

脚本默认写到 `outputs/a2_sanity/`。目录已经存在时不会覆盖；重跑请指定新目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a2.sh \
  --local configs/local.json \
  --run-dir outputs/a2_sanity_v2 \
  2>&1 | tee outputs/logs/a2_sanity_v2.log
```

运行时严格验证：

- 模型/tokenizer/修复后的权重索引元数据哈希与 A1 一致，权重分片大小和 mtime 一致。
- 本地语料、索引 stat 和 Retriever 模型身份与 A1 一致，推理包版本一致。
- `local.json` 实际索引类型与 A1 记录一致；端口地址沿用原服务。
- 对所选题历史 Query 做正常检索预探测，检查 Top-3 ID/内容/顺序是否仍与 A1 一致。
- 用当前 tokenizer 重建原初始 Prompt，并按**原始生成 token ID**和历史 observation
  的编码拼接出每个目标前缀；不把整个文本重新 tokenize 后当成原始前缀。

服务预探测只审计环境，不计入 Agent 搜索预算；单独记录其调用次数。
后续相同 Query 再次调用服务时也会核对可见结果。如果结果变化，会记录
`RetrieverDriftError` 并停止；不会用缓存掩盖变化。score 的细微变化和隐藏候选
顺序不进入当前 Top-3 观测一致性判定，但原始实际响应仍保存。

完整语料/权重文件没有重新计算全内容哈希；stat 检查不能证明全内容相同。
身份检查和服务探测是工程取证，不能宣称覆盖所有未来 Query 的全局 Retriever 稳定性。

## 6. 全局预算的含义

从第 1 轮状态恢复时，第 1 次搜索已完成，后续最多还能实际执行 3 次搜索。
从第 2 轮状态恢复时，前 2 次搜索已完成，后续最多还能执行 2 次搜索。
第四次返回后仍可以生成最终答案；再提出搜索则记为 `search_budget_exhausted`。

上下文 8192、每段生成 512 等沿用原 A1 数值。
各分支记录总成本及 `search_calls_after_target`、`generated_tokens_after_target`，
避免把分叉之前已经花掉的成本重复当作新增成本。

## 7. 输出、退出码和检查方法

| 文件 | 内容 |
| --- | --- |
| `run_manifest.json` | 当前/source commit、A1 文件哈希、继承参数、选题、资源检查和完成状态 |
| `replay_sanity.jsonl` | 每题/目标轮的 2–3 个分支、完整续写、原始前缀、干预及比较结果 |
| `replay_metrics.json` | 工程 gates、分轮/分数据集分母、控制差异、描述性的配对结果 |
| `retrieval_audit.json` | 全部实际预探测/后续调用，是否匹配历史或已有同 Query 观测 |
| `review_cases.json` | 执行下方命令后导出的有限真实案例 |

```bash
python -m json.tool outputs/a2_sanity/replay_metrics.json
python -m work2.inspect_replay --run-dir outputs/a2_sanity --count 4
```

退出码 0：全部计划状态已处理，所要求各轮存在有效控制和排序对，控制一致；
退出码 2：E0/E0 不一致、源记录校验失败或自检未具备所需有效对；
其他异常通常是资源/服务/推理失败。最终同时看日志及 manifest：

- `completed`：完成计划；仍要检查工程 gate，不能只看这个字符串。
- `control_failed`：两条 clean 续写不同，停止 E1 和后续状态；这是待诊断工程问题，
  不是研究方向的科学 NO-GO。
- `failed`：异常中止；保留已写出的案例，不把部分结果当作全部 10 题自检。

`ready_for_a3_engineering_review=true` 只表示可提交 A3 的工程审查，**不会自动启动
A3，也不是研究/RL GO**。我会审查真实案例后再实现下一阶段。

E0/E1 的 EM 翻转和调用次数差异仅为描述性 Pilot 结果。Query 字符串不同不能
直接记为 OffTarget；本阶段不计算 Recovery，也不基于 10 题声称显著或创新。
同一问题的 t=1/t=2 分叉相关，不能把最多 20 个状态当成 20 个独立问题。

## 8. 反馈命令

```bash
python -m work2.inspect_replay --run-dir outputs/a2_sanity --count 4
python scripts/pack_feedback.py \
  --include outputs/a2_sanity/run_manifest.json \
  --include outputs/a2_sanity/replay_metrics.json \
  --include outputs/a2_sanity/retrieval_audit.json \
  --include outputs/a2_sanity/review_cases.json \
  --include outputs/logs/a2_sanity.log
```

直接反馈生成的 `feedback/` 压缩包或上述文件，不用厂内 git push。
原始 JSONL 留在厂内。失败时同样反馈 manifest、metrics、日志和已有案例；
如果异常发生在生成分支前而没有 JSONL，跳过案例导出即可。

## 9. 常见问题

- **元数据/mtime 与 A1 不一致**：先核对是否仍是原模型和资源。若只是路径迁移或
  重新复制导致 stat 改变，提供具体差异供审查；不要改旧 manifest 或静默放行。
- **A1 没有完整资源身份字段**：保留原实验，反馈缺失字段；需要补充独立旁证，
  不能假装旧运行记录里已有这些信息。
- **本地模型索引 JSON 再次损坏**：脚本报错，不自动修复/下载。保留错误及文件，
  按厂内已有资源维护流程处理，并审计对来源身份的影响。
- **RetrieverDriftError**：确认常驻服务和资源未换；保留 audit 中首次变化的 Query。
  若确有索引变化，不能把新旧结果混为本 cohort 的同前缀比较。
- **E0/E0 输出不同**：先检查具体 token/Query/观测，定位推理或服务非确定性。
  不自动切换 backend、调种子，或把不一致的题删掉后继续 E1。
- **第二轮不可达**：保留 unreachable，照实报告条件分母，不补选“更好”的题。
- **排序后观测相同**：记录 permutation_unchanged，控制仍保留，该状态不计为有效 E1 对。
- **资源忙或 OOM**：反馈即可；本脚本不杀训练进程，也不恢复其他训练任务。

后续仍按：厂内运行/反馈 → 开发服务器审查、改代码并 push → 厂内拉取下一阶段。

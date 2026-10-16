# CT1 厂内执行指导：逐轮信用信号的干预式检验（pilot）

更新：2026-10-16　分支：`feat/credit-testbed`
协议和门槛：[research/CREDIT_PROTOCOL.md](../research/CREDIT_PROTOCOL.md)（先读第 3、4、6 节）。
选题依据：家服调研报告（方向 A）。旧 PCR 方向已按 G1 NO-GO 归档，**不要再跑 PCR**。

本轮只做 pilot：Search-R1 7B PPO，HotpotQA 150 题 + 2Wiki 150 题，温度 1.0 的 on-policy 采样，k=8。
**不训练**、不改协议参数，也不根据结果调参。遇到下文的停止条件就停下回报，不要自行修补代码。

---

## 0. 规则

1. 沿用厂内现有的 searchr1 环境、7B PPO 检查点、E5 + Wiki-18 Flat 索引、常驻 Retriever，以及现有的 `configs/local.json`。
   `local.json` 里的 `base_model`（Qwen2.5-7B-Instruct）只在可选的第 6 步用到。
2. 不覆盖旧的 `outputs/`；本轮产物放在 `outputs/ct1_*`。
3. 不修改仓库代码。有报错就停下，回传完整报错、`git rev-parse HEAD` 和 `pip show vllm torch transformers`。
4. 不要自己下“GO”结论；门槛只是机械值。
5. GPU 需要独占 4 张卡（每卡一个 vLLM，显存占用 0.8）；如果卡被占用，先确认能否停掉再开始。

## 1. 拉代码与测试

```bash
cd /root/develop/workspace/work2_byd          # 以实际路径为准
git status --short                            # 如有本地修改先 git stash
git fetch origin
git switch -c feat/credit-testbed --track origin/feat/credit-testbed 2>/dev/null || git switch feat/credit-testbed
git pull --ff-only
git rev-parse HEAD
export CT_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
bash scripts/run_credit.sh test
```

**期望**：`OK`（家服为 136 项）。

## 2. 冻结题目（CPU，几秒）

```bash
export CT_SOURCES=outputs/si1_dp4_dev/sources.jsonl
bash scripts/run_credit.sh prepare
sha256sum outputs/ct1_plan/plan.json
```

**期望**：`Frozen 300 questions (smoke 8).`。回传这一行和 sha256。目录已存在时会拒绝运行；不要删除，改用 `CT_PLAN=outputs/ct1_plan_b` 并说明原因。

## 3. 检查与 smoke（8 题，约 5–15 分钟）

```bash
export CT_GPUS=0,1,2,3
bash scripts/run_credit.sh check     # 检查 GPU 列表、plan，以及 Retriever 的 **批量** 查询接口；不加载模型
bash scripts/run_credit.sh smoke     # MC 真值 → policy 信号（HF）→ 分析
```

smoke 依次执行三步：

1. **MC 真值**（vLLM）：每题先采样 1 条被打分轨迹（同种子再采一次，检查能否复现），然后批量重新执行
   V（每个观测后的状态）、N（把观测换成语义空观测）、C（全程无证据）三类分支，每类 k=8 次。
2. **policy 信号**（HF transformers，每卡加载一个 7B）：
   - IGPO 式信息增益：gold 答案的平均 log-prob；
   - CIPO 式 EALR：用 4D 注意力掩码让目标 token 看不到观测。
3. **分析**：smoke 只有 8 题，数值没有意义，只看工程是否通过。

**停止条件**：
- `check` 报 `Retriever batch interface check failed`，或 Retriever 返回格式错误：当前检索服务不支持一次发多个 query。停止并回报服务启动命令（不要改服务）。
- ~~`Live retriever drift`~~：**2026-10-16 已在代码侧处理，不再是停止条件。** 厂内 smoke 发现检索服务在并发批量负载下，会把分数并列或几乎并列的文档换顺序（例如 `[0.8303, 0.8251, 0.8251]`）。现在的处理：
  1. 客户端统一排序：先按分数（保留 4 位小数）降序，并列时按 `document_id` 排；
  2. 同一 worker 内，同一个 query 只取第一次的结果并缓存复用，所以所有分支看到的观测一致；
  3. 每命中缓存 50 次，抽一次重新检索，比较 Top-3 的 id **集合**，变化只计数，不中止；
  4. 检索返回 HTTP 500、超时这类临时错误时，自动退避重试，最多 4 次；仍然失败才停止，并回报服务日志。

  回传时附上 `run_manifest.json` 里的 `retrieval` 字段（`audit_change_rate`）。如果这个比例 > 5%，说明服务本身不稳定，需要在报告里写明。
- **不需要**改检索服务、重建索引或豁免 decoy 查询。
- 信号阶段报 `Explicit 4D causal mask is not equivalent ...` 或 `Evidence mask had no effect`：当前 transformers 或注意力实现没有正确执行自定义掩码。
  默认已是 `eager`（家服在 transformers 4.47.1 上验证过：eager 严格执行 4D 掩码；sdpa + bf16 一传入掩码就会换 kernel，单 token 的 log-prob 最多差约 0.9，因此不可用）。
  报错时停止，并回报 transformers 版本和 `CT_ATTN` 的值；**不要**改成 sdpa。
- 任何 worker 非零退出：回传 `outputs/ct1_smoke/logs/*.log` 的最后 50 行。
- `CUDA out of memory`：MC 阶段在命令后追加 `--gpu-memory-utilization 0.7` 重试；信号阶段（eager、8k 上下文）如果 OOM，先确认卡上没有其他进程。都要在回报里说明。
- `FlashAttention only supports Ampere GPUs or newer`：只会出现在 H800/A800 以外的卡上（例如 RTX 5090 上的 vLLM 0.6.3）。设置 `export VLLM_ATTENTION_BACKEND=XFORMERS` 后重跑；H800 上不需要。

**smoke 回传**：
- `outputs/ct1_smoke/run_manifest.json`；
- 终端打印的 `Merged ... questions; ... branch rollouts; trajectory reproduced x/8 ...` 一行；
- 每个 `logs/mc_worker_*.log` 的最后 3 行（含每题耗时）；
- `outputs/ct1_smoke/signals/policy/worker_0_manifest.json`（其中有 `mask_self_check`）。

## 4. 正式 pilot（300 题）

```bash
nohup bash scripts/run_credit.sh run > outputs/ct1_pilot_mc.log 2>&1 &
tail -f outputs/ct1_pilot_mc.log
```

- 预计时长：根据 smoke 中每题耗时 × 75 题/卡来估算（家服估计 0.5–2 h）。**请在回报里写实际墙钟时间**，它就是门槛 G-c 的输入。
- 中断后重跑**同一条命令**即可续跑：已完成的题不会重跑。要求代码 commit、GPU 映射和题目都不变。跑的过程中不要 `git pull`。
- 完成标志：`outputs/ct1_pilot/run_manifest.json` 中 `"status": "completed"`，并且有 `records_sha256`。

## 4b. 2026-10-16 补充：按协议做 k=16 重测（当前要做的事）

k=8 pilot 的 G-a 未通过（A 的信度 0.259；E 为 0.740，已达标）。按协议 §6，在**同一批 300 题**上用 k=16 重测一次。
前 8 个样本按种子与 k=8 完全相同，只是再追加 8 个。分析见 `research/CT1_PILOT_RESULTS_AND_DECISION.md`。

```bash
git pull --ff-only && bash scripts/run_credit.sh test          # 期望 141 项 OK
export CT_PROTOCOL=configs/credit_v1_k16.json CT_PLAN=outputs/ct1_plan_k16 CT_RUN=outputs/ct1_pilot_k16
export CT_SOURCES=outputs/si1_dp4_dev/sources.jsonl CT_GPUS=0,1,2,3
bash scripts/run_credit.sh prepare
"$CT_PYTHON" -c "import json;a=json.load(open('outputs/ct1_plan/plan.json'));b=json.load(open('outputs/ct1_plan_k16/plan.json'));print('same_questions', [q['sample_id'] for q in a['questions']]==[q['sample_id'] for q in b['questions']])"
nohup bash scripts/run_credit.sh run > outputs/ct1_pilot_k16_mc.log 2>&1 &   # 复用已通过的 outputs/ct1_smoke；预计约 40 min
bash scripts/run_credit.sh signals          # 新增探索性信号 cf_ig，约 2 min
bash scripts/run_credit.sh signals-ref      # 可选：configs/local.json 的 base_model 存在时才跑；不存在就跳过并说明
bash scripts/run_credit.sh analyze
bash scripts/run_credit.sh pack
```

- `same_questions` 必须为 `True`，否则停下回报。
- k=8 的 `outputs/ct1_pilot` 保持不动。
- 回传：`outputs/ct1_pilot_k16/analysis/ct1_report.md` 全文（含新增的“探索性诊断”一节）、`ct1_report.json` 中的 `gates`、`reliability`、`exploratory`，以及 `run_manifest.json` 中的 `mc_wall_seconds` 和 `retrieval`。另外写明是否跑了 `signals-ref`。
- 探索性诊断不进入门槛；同样不要自行下 GO/NO-GO。
## 5. 信号与分析

```bash
bash scripts/run_credit.sh signals          # policy：IG + EALR，四卡
bash scripts/run_credit.sh analyze          # 生成 outputs/ct1_pilot/analysis/
bash scripts/run_credit.sh pack
```

## 6. 可选：TRACE 式参考模型信号

`configs/local.json` 的 `base_model` 指向本地 Qwen2.5-7B-Instruct 时运行：

```bash
bash scripts/run_credit.sh signals-ref
CT_RUN=outputs/ct1_pilot bash scripts/run_credit.sh analyze --output-dir outputs/ct1_pilot/analysis_with_ref
```

本地没有这个模型就跳过，并在回报中写明。trace 不进入 pilot 门槛。

## 7. 回传清单（直接贴进对话）

1. `git rev-parse HEAD`、`git status --short`、`CT_SOURCES`、GPU 与显存设置、`CT_ATTN`。
2. `outputs/ct1_pilot/analysis/ct1_report.md` **全文**。
3. `ct1_report.json` 中的 `gates`、`reliability`、`truth`、`memory_stratum_difference`、`ig_vs_ealr`。
4. `run_manifest.json` 中的 `mc_wall_seconds`、`trajectory_reproduced`、`null_eligible`、`branch_rollouts`、`retrieval`；信号阶段的耗时（看 `signals/policy/worker_*_manifest.json` 里的 `seconds`）。
5. 任意 3 题的 `ct1_turns.json` 条目（同一题的所有轮次一起给出）。
6. pack 生成的 feedback 包路径。

## 8. 输出字段的含义

| 字段 | 含义 | 健全性参考（不是预期值） |
| --- | --- | --- |
| `truth.mean_E` | 观测内容的平均因果价值 V_t − V∅_t | 应该大于 0；接近 0 或为负时要报告 |
| `reliability.A/E.spearman_brown` | k=8 拆半后的信度 | G-a 要求 ≥ 0.6 |
| `signals.ig.vs_E.spearman` | IGPO 信息增益与证据价值的一致性 | — |
| `signals.ealr.vs_E.spearman` | CIPO EALR 与证据价值的一致性 | — |
| `memory_known_questions` | 无证据世界里 V_cb ≥ 0.5 的题数 | 太少（< 20）时，分层结论不可靠 |
| `trajectory_reproduced` | 同种子重采样的复现数 | 不是门槛，但明显偏低时要报告 |

参考值只用来发现明显的工程异常，不能拿来调参。

## 9. 本轮不做

第 3–4 周的多检查点、LLM 评论器 / 提示式打分，以及第 4–8 周的 3B RL 预测效度实验，都要等家服根据 pilot 的 G-a/b/c 记录决定之后，另出协议和代码。

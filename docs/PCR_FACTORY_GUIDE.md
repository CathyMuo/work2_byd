# PCR v1 厂内执行指导（给厂内 agent / 操作者）

更新：2026-10-16　分支：`feat/path-luck-pcr`
研究协议与预登记门槛：[research/PCR_PROTOCOL.md](../research/PCR_PROTOCOL.md)（先读第 3–5 节）

本轮只做 **Phase 0（CPU）+ Phase 1（Exp-1/2/3，GPU 推理）**。不训练、不做 RL，不改协议参数，也不根据结果调参。
每完成一步就把指定的输出原样回传；遇到“停止条件”立即停下回报，不要自己修补、放宽或绕过。

---

## 0. 必须遵守的规则

1. 只用厂内现有的 Search-R1 环境和资源（同一个 7B PPO 检查点、E5、Wiki-18 Flat 索引、常驻 Retriever 服务）。不下载、不装包、不重建索引。
2. `configs/local.json` 沿用已有的那份，不要提交它。
3. **不覆盖任何旧的 `outputs/`**。新产物全部放在 `outputs/pcr_*`。
4. 不修改仓库代码。代码有问题就停下报告（贴出完整报错和 `git rev-parse HEAD`）。
5. 跑之前和跑完都要记录：`git rev-parse HEAD`、`git status --short`（应为空）、`nvidia-smi`。
6. 不要把任何结论写成“GO”。门槛只是机械值，GO/NO-GO 由家服在 `DECISION_LOG.md` 中决定。

## 1. 拉取代码

厂内只读公开仓库，不需要 push。

```bash
cd /root/develop/workspace/work2_byd      # 以厂内实际路径为准
git status --short                        # 有本地修改先 git stash，不要丢
git fetch origin
git switch feat/path-luck-pcr 2>/dev/null || git switch -c feat/path-luck-pcr --track origin/feat/path-luck-pcr
git pull --ff-only
git rev-parse HEAD
```

指定解释器（与 SI1 相同的 Search-R1 环境）：

```bash
export PCR_PYTHON=/path/to/searchr1/env/bin/python   # 若 SI1_PYTHON 已设置会自动沿用
bash scripts/run_pcr.sh test
```

**期望**：最后一行为 `OK`（2026-10-16 家服为 124 项）。不是 OK → 停止，回传完整输出。

## 2. Phase 0：已有产物的 CPU 重分析（约 1–5 分钟，不需要 GPU）

```bash
export PCR_A1=outputs/a1_clean_scale500/clean_rollouts.jsonl
export PCR_A3=outputs/a3_scale500_merged/paired_rollouts.jsonl   # 用实际文件名
bash scripts/run_pcr.sh phase0
```

脚本先打印 A3 第一条记录的格式（`--schema-only`），再运行 `work2.phase0_luck`。

- 如果报 `Unrecognized intervention branches ...` 或 `Ambiguous branch`：按打印出的分支名追加映射后重跑，例如
  `bash scripts/run_pcr.sh phase0 --branch E1_permutation=permuted_a --branch E2_distractor=e2`
  （失败时不会生成输出目录；成功后如果要再跑一次，设置 `PCR_PHASE0_OUT=outputs/pcr_phase0_v2`）。
- 如果报 `Pre-target history changed`、`E0/E0 control mismatch` 或分数不一致：**停止**，回传报错和 schema 输出。不要改代码去绕过检查。

**回传**：
- 终端最后的 E1/E2/E3/E4 汇总行；
- `outputs/pcr_phase0/phase0_report.json` 全文；
- `format_flips_for_review.jsonl` 的行数，以及前 20 行；
- `semantic_flips.jsonl` 的行数。

Phase 0 不阻塞 Phase 1，可以先跑 Phase 1 的 prepare/smoke。

## 3. Phase 1：冻结状态

源轨迹默认用 SI1 四卡开发运行的 `outputs/si1_dp4_dev/sources.jsonl`。

```bash
ls -la outputs/si1_dp4_dev/sources.jsonl
export PCR_SOURCES=outputs/si1_dp4_dev/sources.jsonl
export PCR_PER_DATASET=300          # 预登记默认值；不要改
bash scripts/run_pcr.sh prepare
```

**期望输出**：`Frozen 600 questions / ~1150 states (unreachable by turn: {'2': ...}).`
回传这一行和 `outputs/pcr_plan/plan.json` 的 sha256（`sha256sum outputs/pcr_plan/plan.json`）。

- SI1 文件不存在时，改用 `PCR_SOURCES=outputs/a1_clean_scale500/clean_rollouts.jsonl`，并在回报里写明。
- `outputs/pcr_plan` 已存在时会拒绝运行。不要删除旧的 plan，换一个 `PCR_PLAN=outputs/pcr_plan_b`，并说明原因。

## 4. Phase 1：检查与 smoke（10 题，四卡，约 5–15 分钟）

先确认 Retriever 服务正在运行（与 SI1 相同的 `retrieval_url`），然后：

```bash
export PCR_GPUS=0,1,2,3
bash scripts/run_pcr.sh check       # 不加载模型，只检查 GPU/服务/plan
bash scripts/run_pcr.sh smoke       # 10 题 × 目标轮 1/2 × 16 臂，之后自动跑一次分析
```

每张卡一个独立的 vLLM（TP=1，显存占用 0.75）。如果卡上还有其他作业，用 `--gpu-memory-utilization 0.6` 追加在命令后，并在回报里写明。

**停止条件（smoke 与正式运行相同）**：
- `E0 control does not reproduce source continuation`：同一前缀的原样重放和源轨迹不一致，说明当前环境（vLLM/权重/检索服务）和产生源轨迹时不同。**停止**，回传日志 `outputs/pcr_smoke/logs/worker_*.log` 的最后 50 行、`pip show vllm torch transformers` 和 `nvidia-smi`。不要换源、不要放宽。
- `Live retriever differs` / `RetrieverDriftError`：检索服务的返回和源轨迹不一致。停止并回报。
- 任何 worker 非零退出：协调器会停掉其余 worker；回传对应日志。

**smoke 期望**：终端最后是 `Merged N states; all E0/E0 controls exact; retrieval drift 0.`，随后打印分析摘要。
smoke 只有 10 题，数值没有意义，只看工程是否通过。回传 `outputs/pcr_smoke/run_manifest.json`，以及每个 worker 日志的最后 5 行。

## 5. Phase 1：正式运行（600 题，约 1150 状态 × 16 臂，四卡预计 2–4 小时）

```bash
nohup bash scripts/run_pcr.sh run > outputs/pcr_exp1_coordinator.log 2>&1 &
tail -f outputs/pcr_exp1_coordinator.log        # 每 60 秒打印各卡进度
```

- 中途失败或被中断时：先看日志、修复外部原因（服务、显存），再执行 `bash scripts/run_pcr.sh resume`。
  续跑要求代码 commit、GPU 映射和题目完全相同，已完成的状态不会重跑。**不要**删除 `workers/` 下的任何文件。
- 跑的过程中不要 `git pull`（worker 会检查 commit，一旦变化就拒绝运行）。

完成标志：`outputs/pcr_exp1/run_manifest.json` 中 `"status": "completed"`、`"all_controls_exact": true`、`"retrieval_drift": 0`。

## 6. 分析（CPU，约 3–10 分钟）

```bash
bash scripts/run_pcr.sh analyze          # 会自动用 SI1 cases.jsonl 做 turn-1 review 的复现交叉检查
bash scripts/run_pcr.sh pack             # 只打包汇总，不含原始轨迹/权重/local.json
```

输出在 `outputs/pcr_exp1/analysis/`：
- `pcr_report.md`：中文摘要（路径运气、逐臂翻转、Exp-2、Exp-3、门槛机械值）；
- `pcr_report.json`：完整数字（含分层：轮次/数据集/答案类型/clean 搜索次数/证据代理/目标轮是否含支持事实）；
- `flip_cases.jsonl`：前 60 个“只换顺序就发生语义翻转”的案例，用于论文首图。

## 7. 回传清单（直接贴进对话）

1. `git rev-parse HEAD`、`git status --short`、使用的 `PCR_SOURCES`、GPU 和显存设置。
2. Phase 0：第 2 节列出的内容。
3. Phase 1：`outputs/pcr_exp1/analysis/pcr_report.md` **全文**。
4. `pcr_report.json` 里的 `gates`、`flip_table`、`recovery.semantic.selected_fragile`、`deletion.semantic.all_matched`、`si1_review_crosscheck`。
5. `flip_cases.jsonl` 的前 5 条。
6. 运行耗时：`run_manifest.json` 中的 `wall_seconds`。
7. pack 生成的 feedback 包路径（如果能传文件）。

## 8. 输出的含义（方便厂内 agent 自查）

| 字段 | 含义 | 立项线索给出的参考量级 |
| --- | --- | --- |
| `flip_table.perm_k.degraded_semantic / clean_correct` | 单个排列造成的语义退化率 | E1 约 12.9%（含格式翻转） |
| `luck.semantic.lucky_share_of_accuracy` | clean 答对、但 6 个安慰剂里过半答错的比例 | 假设 15–35%，G1 门槛 ≥10% |
| `luck.semantic.mEM` vs `clean_em` | 对路径取平均后的准确率 vs 单条轨迹的准确率 | — |
| `recovery...methods.review.placebo_explained_fraction` | review 的修复率里，有多少能被换排列/中性文本解释 | G2 门槛 ≥ 1/3 |
| `deletion...support_minus_nonsupport` | 删支持文档比删无关文档多伤多少 | G3 要求 CI 上界 < 0 |
| `si1_review_crosscheck` | turn-1 review 臂与 SI1 的同臂是否逐字一致 | 应接近全部一致；明显不一致时要报告 |

参考量级只用来发现明显的工程异常（例如 lucky share = 0 或 = 1），**不是**预期结果，也不能拿来调参。

## 9. 本轮不做的事

Phase 2（多检查点、MuSiQue/Bamboogle/NQ、BGE/BM25、温度 0.6 × 4 种子）和 Phase 3（LDR 训练）
都要等家服根据 Phase 1 回报记录 G1–G3 的决定之后，另出协议和代码。厂内不要提前开始。

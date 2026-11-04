# AS2 厂内执行指导

更新：2026-11-05（含修订 1）　协议：`research/AS2_PROTOCOL.md`（第 1–7 节和第 8 节修订 1，都在运行前冻结）　脚本：`scripts/run_as2.sh`、`scripts/run_u2.sh`

**为什么又有 GPU 实验**：AS1 指导第 7 节说过“之后不再安排 GPU 实验”。之后论文收到三份审稿意见，用户决定补实验来回应。本批次一次做完：

| 部分 | 内容 | 智能体 | 资源 |
| --- | --- | --- | --- |
| A. CPU 分析 | 在**已有**的 H1、AS1、U1N 输出上做五项分析；回传 `arm_diagnose` 全文和 MR-Search 报告全文 | 14 个 | 只用 CPU，约半小时 |
| B. `as2` | 5 种无意义文字变体；把复查提示和无意义文字放进观测块内部；分级阳性对照（“第几篇有答案”“多给一篇含答案的文档”） | 14 个 | GPU |
| C. `as2s` | 同一前缀上温度 1.0 采样 4 次；另外贪心跑完 5 种文档顺序 | 3B PPO、7B GRPO、14B PPO、32B GRPO | GPU |
| D. MR-Search 贪心复核 | 原审计用的是采样，这次按原论文的设置用贪心解码再跑一遍 | 7B PPO、3B PPO | GPU，较小 |

状态与 AS1 完全相同（同一选择规则、同一种子）。只用推理，不训练，不重新生成源轨迹。GPU 部分预计 4×H800 上 1.5–2 天，顺序：B → C → D。

## 0. 规则（同前）

- 不改代码。
- 不修改 `research/DECISION_LOG.md`、`research/STATE.json`。
- 运行期间不提交、不拉取代码。
- `configs/local*.json` 不提交。
- 报错就停下回报，不要自行重试或绕过。

## 1. 开跑前

```bash
cd /root/develop/workspace/work2_byd
export U1_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
git status --short                      # 运行目录以外不应有改动
git fetch origin && git merge origin/feat/credit-testbed
bash scripts/run_as2.sh test            # 期望没有 FAIL / ERROR
ls configs/local.u1_*.json              # 14 个智能体都要有
ls outputs/u1/*/as1/states.jsonl | wc -l   # 14：B 要拼接 AS1 的结果
ls outputs/u2/plan/plan.json            # D 沿用 U2 的冻结计划
```

**检索服务**：必须是这 14 个智能体源轨迹用的那一套 **4 卡**检索服务。AS1 收官后如果已经停掉，按原命令用 4 卡重启，参数一字不改。每次检索都会和源轨迹逐条比对，配置不对会报 `RetrieverDriftError`。

## 2. A：只用 CPU（先做，可与检索服务启动并行）

```bash
cd /root/develop/workspace/work2_byd
# A1 五项分析：四格表、三种加权、alert 对 review、问题级策略、语义口径核查
$U1_PYTHON -m work2.review_cpu --u1-root outputs/u1 --output outputs/paper/review_cpu
# A2 插入臂的终止状态（把超额伤害拆成“格式坏了”和“答错了”）
$U1_PYTHON -m work2.arm_diagnose --agents configs/u1_agents.json configs/u1x_agents.json configs/u1l_agents.json \
  --output outputs/paper/arm_diagnose_all.json
# A3 MR-Search 原审计的报告全文（已有文件，只是没回传过全文）
ls outputs/u2/mr_v03_7b_ppo/analysis/u2_report.md outputs/u2/mr_v03_3b_ppo/analysis/u2_report.md
wc -l outputs/paper/review_cpu/semantic_cases.jsonl
```

## 3. 冒烟测试（约 40 分钟）

```bash
bash scripts/run_as2.sh smoke v03_7b_ppo      # 只有 B
bash scripts/run_as2.sh smoke v03_7b_grpo     # B 和 C
bash scripts/run_u2.sh smoke mrg v03_7b_ppo   # D
```

冒烟会先冻结计划（`as2_plan/`、`as2s_plan/`），正式运行沿用这两个计划。`*_smoke/` 是单独的目录，不影响正式运行。

**冒烟后请核对**：`outputs/u1/v03_7b_grpo/as2_plan/plan.json` 里 `selected_from.selected` 的三个数，应与同一智能体 `as1_plan/plan.json` 里的完全相同。不同就停下回报。

## 4. 正式运行

```bash
nohup bash -c 'bash scripts/run_as2.sh all && bash scripts/run_u2.sh run mrg v03_7b_ppo \
  && bash scripts/run_u2.sh smoke mrg v03_3b_ppo && bash scripts/run_u2.sh run mrg v03_3b_ppo' > outputs/as2.log 2>&1 &
tail -f outputs/as2.log
```

- 资源分配与 AS1 相同：32B 用 GPU 1、2、3，显存比例 0.85；其余用 4 张卡，显存比例 0.75。MR-Search 与原 U2 一样用 4 张卡。
- 中途断了，重新执行同一条命令即可续跑：已完成的会跳过。
- B 的两对分级臂只在“有文档含答案”的状态上运行，其余状态记为 None，所以日志里每个状态的臂数不一样是正常的。

全部完成后：

```bash
bash scripts/run_as2.sh summary
```

## 5. 停止条件

- `RetrieverDriftError`：检索服务不是源轨迹那一套，停下回报。
- `Cannot match filler token count` 或 `Cannot match neutral instruction token count`：停下回报智能体和题号。家服已在 Qwen2.5 和 Llama-3.2 的分词器上核对过 1500 段新填充文字，全部等长。
- `Observation has no closing ... tag`：停下回报智能体。
- `Sampled continuations ... require the vLLM backend`：C 必须用默认的 vLLM 后端。
- 某个运行合并时报 `agent not measurable`（E0 不复现超过 2%）：记下，继续跑其他智能体，回报。
- 显存不足：停下回报 `nvidia-smi` 和报错，不要自己改显存比例。

## 6. 回传（可以分两次：A 做完先传一次，GPU 部分完成后再传一次）

**A（CPU）**
1. `outputs/paper/review_cpu/review_cpu.md` 全文。
2. `outputs/paper/review_cpu/semantic_cases.jsonl`（作为文件发送；如果只能贴文本，先贴 `wc -l` 和前 200 行）。
3. `outputs/paper/arm_diagnose_all.md` 全文（较长，可以单独贴）。
4. `outputs/u2/mr_v03_7b_ppo/analysis/u2_report.md` 和 `outputs/u2/mr_v03_3b_ppo/analysis/u2_report.md` 全文。

**GPU**
1. `git rev-parse HEAD`；开始和结束时间；每个运行的耗时。
2. 每个智能体 `as2_plan/plan.json`（和 4 个智能体的 `as2s_plan/plan.json`）里的 `selected_from.selected`。
3. 每个运行 `run_manifest.json` 里的 `control_mismatch_states`、`control_mismatch_rate`。
4. `outputs/paper/as2/as2.md` 全文（含开头的“预登记判定”和“复现”）。
5. `outputs/u2/mrg_v03_7b_ppo/analysis/u2_report.md` 和 `outputs/u2/mrg_v03_3b_ppo/analysis/u2_report.md` 全文。
6. 冒烟测试有没有问题。

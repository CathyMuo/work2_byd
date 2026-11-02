# AS1 / U1N 厂内执行指导

更新：2026-11-02　协议：`research/AS1_U1N_PROTOCOL.md`（运行前冻结）　脚本：`scripts/run_as1u1n.sh`

**做什么**：在 H1 主表的 14 个 RL 智能体上补两项实验。

- **AS1 阳性对照**：在 H1 已有的状态上，比较“直接给出答案”“只提示当前思路可能错了”和“复查提示”这三种内容，各自减去等长的无意义文字。目的是证明我们的工具能测出内容的作用。
- **U1N 自然失败**：不换文档顺序，直接比较提示和无意义文字。

只用推理，不训练，不重新生成源轨迹。预计 4×H800 上 1–1.5 天，先跑完全部 AS1，再跑 U1N。

## 0. 规则（同前）

- 不改代码。
- 不修改 `research/DECISION_LOG.md`、`research/STATE.json`。
- 运行期间不提交、不拉取代码。
- `configs/local*.json` 不提交。
- 报错就停下回报，不要自行重试或绕过。

## 1. 开跑前

先做完 U1L 指导第 9 节（只用 CPU），并回传。两件事互不影响，但第 9 节要先拉一次代码，本批次也要用新代码，所以按下面的顺序：

```bash
cd /root/develop/workspace/work2_byd
export U1_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
git status --short                      # 运行目录以外不应有改动
git fetch origin && git merge origin/feat/credit-testbed
bash scripts/run_as1u1n.sh test         # 期望没有 FAIL / ERROR
ls configs/local.u1_*.json              # 14 个智能体都要有（U1、U1x、U1L 时建的）
```

**检索服务**：这 14 个智能体的源轨迹都是在 4 卡检索下生成的，必须继续用 10-30 15:24 起那一套 4 卡服务。

- 服务没停过：不要动它。
- 服务停过：按原命令用 4 卡重启，参数一字不改。

每次检索都会和源轨迹逐条比对，配置不对会报 `RetrieverDriftError`。

## 2. 冒烟测试（约 20 分钟）

```bash
bash scripts/run_as1u1n.sh smoke v03_7b_ppo
ls outputs/u1/v03_7b_ppo/as1_plan outputs/u1/v03_7b_ppo/as1_smoke outputs/u1/v03_7b_ppo/u1n_smoke
```

冒烟会先冻结计划（`as1_plan/`、`u1n_plan/`），正式运行沿用这两个计划。`as1_smoke/`、`u1n_smoke/` 是单独的目录，不影响正式运行。

## 3. 正式运行

```bash
nohup bash scripts/run_as1u1n.sh all > outputs/as1u1n.log 2>&1 &
tail -f outputs/as1u1n.log
```

脚本自动分配资源，不用设环境变量：
- 32B 用 GPU 1、2、3，显存比例 0.85，与 U1L 第 7.6 节相同；
- 其余用 4 张卡，显存比例 0.75。

中途断了，重新执行同一条命令即可续跑：已完成的跳过，未完成的用 `--resume` 接着跑。

全部完成后执行：

```bash
bash scripts/run_as1u1n.sh summary
```

## 4. 停止条件

- `RetrieverDriftError`：检索服务不是源轨迹那一套，停下回报。
- `Cannot match neutral instruction token count`：某道题的提示长度对不上等长文字，停下回报智能体和题号。家服已在 Qwen2.5 和 Llama-3.2 的分词器上检查过 33–101 个 token 的每种长度，预计不会出现。
- 某个运行合并时报 `agent not measurable`（E0 不复现超过 2%）：记下，继续跑其他智能体，回报。
- 显存不足：停下回报 `nvidia-smi` 和报错，不要自己改显存比例。

## 5. 回传（全文贴在一个文本里）

1. `git rev-parse HEAD`；开始和结束时间；每个智能体两项运行各自的耗时。
2. 每个智能体 `as1_plan/plan.json` 里的 `selected_from.available` 和 `selected_from.selected`。
3. 每个运行 `run_manifest.json` 里的 `control_mismatch_states`、`control_mismatch_rate`。
4. `outputs/paper/as1_u1n/as1_u1n.md` 全文（含开头的“预登记判定”）。
5. 冒烟测试有没有问题。

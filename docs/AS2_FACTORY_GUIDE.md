# AS2 厂内执行指导

更新：2026-11-04　协议：`research/AS2_PROTOCOL.md`（运行前冻结）　脚本：`scripts/run_as2.sh`

**为什么又有 GPU 实验**：AS1 指导第 7 节说过“之后不再安排 GPU 实验”。之后论文收到两份审稿意见，用户决定补三项实验来回应：

- **实验 1：换几种无意义文字**（14 个智能体）。审稿人质疑我们只用了一种无意义文字（“ok . - note line”），而且它不含提示里那种 `<search>`/`<answer>` 字样。这次加 4 种，与复查提示等长：
  - 自然的无关叙述；
  - 无意义文字后面加上提示的最后一句（带动作标签）；
  - 复查提示的词打乱；
  - 纯标点。
- **实验 2：分级阳性对照**（同一批运行）。在“告诉它错了”和“直接给答案”之间，加两档中等信息：
  - 只说“上面第几篇文档里有答案”；
  - 再给一篇含答案的检索候选文档。
- **实验 3：采样解码**（4 个 Search-R1 智能体）。主实验全是贪心解码；这次在同一前缀上用温度 1.0、每臂采样 4 次，重复主实验的几个臂。

状态与 AS1 完全相同（同一选择规则、同一种子），只用推理，不训练，不重新生成源轨迹。预计 4×H800 上 1–1.5 天：先跑完全部实验 1、2（`as2`），再跑实验 3（`as2s`）。

另有一项**只用 CPU**的回传（第 6 节），可以在开跑前或运行期间做。

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
bash scripts/run_as2.sh test            # 期望 206 个测试，没有 FAIL / ERROR
ls configs/local.u1_*.json              # 14 个智能体都要有
ls outputs/u1/*/as1/states.jsonl | wc -l   # 14：实验 2 要拼接 AS1 的结果
```

**检索服务**：必须是这 14 个智能体源轨迹用的那一套 **4 卡**检索服务。
- AS1 收官后如果已经停掉：按原命令用 4 卡重启，参数一字不改。
- 每次检索都会和源轨迹逐条比对，配置不对会报 `RetrieverDriftError`。

## 2. 冒烟测试（约 30 分钟）

```bash
bash scripts/run_as2.sh smoke v03_7b_ppo      # 只跑实验 1、2
bash scripts/run_as2.sh smoke v03_7b_grpo     # 实验 1、2，加实验 3
ls outputs/u1/v03_7b_grpo/as2_plan outputs/u1/v03_7b_grpo/as2_smoke outputs/u1/v03_7b_grpo/as2s_smoke
```

冒烟会先冻结计划（`as2_plan/`、`as2s_plan/`），正式运行沿用这两个计划。`*_smoke/` 是单独的目录，不影响正式运行。

**冒烟后请核对一项**：`as2_plan/plan.json` 里 `selected_from.selected` 的三个数，应与同一智能体 `as1_plan/plan.json` 里的完全相同。不同就停下回报。

## 3. 正式运行

```bash
nohup bash scripts/run_as2.sh all > outputs/as2.log 2>&1 &
tail -f outputs/as2.log
```

- 资源分配与 AS1 相同：32B 用 GPU 1、2、3，显存比例 0.85；其余用 4 张卡，显存比例 0.75。
- 中途断了，重新执行同一条命令即可续跑。
- 实验 2 的两对臂只在“有文档含答案”的状态上运行，其余状态记为 None。日志里每个状态打印的臂数不一样是正常的。

全部完成后执行：

```bash
bash scripts/run_as2.sh summary
```

## 4. 停止条件

- `RetrieverDriftError`：检索服务不是源轨迹那一套，停下回报。
- `Cannot match filler token count` 或 `Cannot match neutral instruction token count`：停下回报智能体和题号。家服已在 Qwen2.5 和 Llama-3.2 的分词器上核对过 1200 段新填充文字，全部等长。
- `Sampled continuations ... require the vLLM backend`：实验 3 必须用默认的 vLLM 后端，不要加 `--backend hf`。
- 某个运行合并时报 `agent not measurable`（E0 不复现超过 2%）：记下，继续跑其他智能体，回报。
- 显存不足：停下回报 `nvidia-smi` 和报错，不要自己改显存比例。

## 5. 回传（全文贴在一个文本里）

1. `git rev-parse HEAD`；开始和结束时间；每个智能体每个运行的耗时。
2. 每个智能体 `as2_plan/plan.json`（和 4 个智能体的 `as2s_plan/plan.json`）里的 `selected_from.selected`。
3. 每个运行 `run_manifest.json` 里的 `control_mismatch_states`、`control_mismatch_rate`。
4. `outputs/paper/as2/as2.md` 全文（含开头的“预登记判定”和“复现”）。
5. 冒烟测试有没有问题。

## 6. 只用 CPU：插入臂的终止状态（审稿意见 W4b）

审稿人要求把“超额伤害”拆成两部分：输出格式坏了，还是给了错误答案。`work2/arm_diagnose.py` 已经按臂统计稳定状态上的终止状态（正常作答、格式错误、长度上限……）和 EM。U1L 时只回传了其中的有效性表，这次请回传全文：

```bash
cd /root/develop/workspace/work2_byd
$U1_PYTHON -m work2.arm_diagnose --agents configs/u1_agents.json configs/u1x_agents.json configs/u1l_agents.json \
  --output outputs/paper/arm_diagnose_all.json
```

回传 `outputs/paper/arm_diagnose_all.md` **全文**（每个智能体一节，含各臂的作答率、EM、终止状态分布和例子）。文件较长，可以单独贴一次。

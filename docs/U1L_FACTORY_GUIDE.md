# U1L 厂内执行指导：在 14B / 32B 智能体上检验 H1

更新：2026-10-28（同日修订：加入 Search-R1 32B GRPO、14B PPO、14B Instruct-GRPO）；**2026-10-30 增补第 7 节：续跑方案与 E0 修订，现在从第 7 节开始做**　分支：`feat/credit-testbed`
协议（**先读第 2、4 节**）：[research/U1L_PROTOCOL.md](../research/U1L_PROTOCOL.md)

目的：检验“恢复收益主要是均值回归”这个结论是否只适用于小模型。共 7 个智能体，**只跑 H1**（源轨迹 + PCR）：

| 顺序 | 标签 | 模型 | 类型 |
| --- | --- | --- | --- |
| 1 | `v03_32b_grpo` | Search-R1 v0.3 32B GRPO | RL，与主表同一配方 |
| 2 | `research_32b` | ReSearch-Qwen-32B-Instruct | RL，另一种配方 |
| 3 | `v03_14b_ppo` | Search-R1 v0.3 14B PPO | RL |
| 4 | `v03_14b_it_grpo` | Search-R1 v0.3 14B Instruct-GRPO | RL |
| 5 | `qwen25_32b_inst` | Qwen2.5-32B-Instruct | 只靠提示（参照） |
| 6 | `research_7b` | ReSearch-Qwen-7B-Instruct | RL |
| 7 | `qwen25_7b_inst` | Qwen2.5-7B-Instruct | 只靠提示（参照） |

**只做推理**。四卡合计约 4.5–5 天，每跑完一个智能体回报一次。

---

## 0. 规则

1. 开跑前用 `nvidia-smi` 确认四张卡空闲。用户的显卡保活程序按用户自己的习惯处理，实验结束后再恢复。
2. **运行期间不要提交、不要拉取代码**。worker 发现代码变了会拒绝续跑。
3. 不改代码；不修改 `research/DECISION_LOG.md` 和 `research/STATE.json`；产物都在 `outputs/u1/<新标签>/`，不覆盖已有目录。
4. 可以开着进度看板（`bash scripts/run_board.sh`）看进度。看板只读，不影响实验。

## 1. 拉代码、测试（开跑前做一次）

```bash
cd /root/develop/workspace/work2_byd
git status --short && git fetch origin && git log --oneline -3
git pull --ff-only      # 有厂内本地提交时，改用 git rebase origin/feat/credit-testbed
export U1_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
bash scripts/run_u1l.sh test
```

期望：没有 FAIL / ERROR。

## 2. 下载模型（内部镜像），先看磁盘

```bash
df -h /data      # 需要约 280 GB 空闲
export HF_ENDPOINT=https://hf-cache.byd.com
for repo in \
  PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3 \
  agentrl/ReSearch-Qwen-32B-Instruct \
  PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3 \
  PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3 \
  Qwen/Qwen2.5-32B-Instruct \
  agentrl/ReSearch-Qwen-7B-Instruct ; do
  hf download "$repo" --local-dir "/data/models/$repo"
done
```

Qwen2.5-7B-Instruct 用 HR1 时已有的 `/root/develop/workspace/models/Qwen2.5-7B-Instruct`，不用重新下载。磁盘不够时，可以先下前两个 32B，跑完再下其余的；但不要删除已有模型。

逐个做全量校验（镜像曾把 500 错误页当作文件内容返回）：

```bash
for d in /data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3 \
         /data/models/agentrl/ReSearch-Qwen-32B-Instruct \
         /data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3 \
         /data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3 \
         /data/models/Qwen/Qwen2.5-32B-Instruct /data/models/agentrl/ReSearch-Qwen-7B-Instruct \
         /root/develop/workspace/models/Qwen2.5-7B-Instruct; do
"$U1_PYTHON" - "$d" <<'PY'
import glob, sys
from safetensors import safe_open
from transformers import AutoTokenizer
d = sys.argv[1]
files = sorted(glob.glob(d + "/*.safetensors")); assert files, "no safetensors in " + d
n = 0
for f in files:
    with safe_open(f, "pt") as h:
        for k in h.keys():
            h.get_tensor(k); n += 1
tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
print(d.rsplit("/", 1)[-1], len(files), "files", n, "tensors OK; chat_template:", bool(tok.chat_template))
PY
done
```

期望：7 行都显示 `tensors OK; chat_template: True`。镜像上缺哪个仓库就跳过它并写进回报，**不要换成别的仓库**。

## 3. 本地配置（不提交）

```bash
"$U1_PYTHON" - <<'PY'
import json
base = json.load(open("configs/local.json"))
m = "/data/models/"
paths = {"v03_32b_grpo": m + "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-32b-em-grpo-v0.3",
         "research_32b": m + "agentrl/ReSearch-Qwen-32B-Instruct",
         "v03_14b_ppo": m + "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-ppo-v0.3",
         "v03_14b_it_grpo": m + "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3",
         "qwen25_32b_inst": m + "Qwen/Qwen2.5-32B-Instruct",
         "research_7b": m + "agentrl/ReSearch-Qwen-7B-Instruct",
         "qwen25_7b_inst": "/root/develop/workspace/models/Qwen2.5-7B-Instruct"}
for label, path in paths.items():
    json.dump(dict(base, agent_checkpoint=path, checkpoint_version="/".join(path.rstrip("/").split("/")[-2:])),
              open("configs/local.u1_%s.json" % label, "w"), indent=2)
print("ok")
PY
bash scripts/run_u1l.sh check
```

期望：7 个标签都显示 `OK`。

## 4. 运行

```bash
export U1_GPUS=0,1,2,3
export U1_GPU_MEM=0.92        # 32B 在单卡上需要；14B / 7B 用同一个值也没问题
nohup bash scripts/run_u1l.sh all > outputs/u1l_all.log 2>&1 &
tail -f outputs/u1l_all.log
```

- 按上表顺序逐个跑，每个智能体依次是源轨迹、PCR、PCR 分析。
- 中断后重跑同一条命令即可续跑，已完成的部分会跳过。续跑时必须用**同样的** `U1_GPU_MEM`。
- **`v03_32b_grpo` 和 `research_32b` 的源轨迹各自跑完后**（日志里出现 `[<标签>] PCR`），先看一眼保真度。PCR 照常进行，不要停：

```bash
"$U1_PYTHON" -m work2.u1x --roster configs/u1l_agents.json --output outputs/u1l_fidelity_check.json
```

  请把这张表**立即回传**。如果格式失败 ≥ 10%，或 `possibly_misharnessed` 是 True，家服可能会决定提前处理。
- 每跑完一个智能体，回报它的 PCR 分析摘要（`outputs/u1/<标签>/pcr/analysis/`）和日志里的耗时。

全部完成后（这时没有运行中的实验，可以拉取代码）：

```bash
git pull --ff-only
bash scripts/run_u1l.sh summary
bash scripts/run_u1l.sh pack
```

## 5. 停止条件

- **32B 加载时 OOM**：把 `U1_GPU_MEM` 调到 0.95 重试一次。仍然 OOM 就停下回报 `nvidia-smi` 和报错，**不要改代码，也不要换模型**。
- `Source initial prompt differs ...` 或 `Source observation format/content mismatch`：格式配置与源轨迹不一致。停止该智能体，回报日志末尾 30 行。
- `E0 control does not reproduce source continuation`：停止该智能体并回报；其余智能体可以用 `bash scripts/run_u1l.sh agent <标签>` 继续。（2026-10-30 起新开的运行改为记录并排除，见第 7.3 节）
- `Code changed after run freezing`：运行中代码变了。不要手改任何 `run_plan.json`，回报。
- 检索漂移：重跑同一条命令；同一个 query 反复失败就停下回报。

## 6. 回传清单

1. `git rev-parse HEAD`；实际下载到的仓库；第 2 步的 7 行校验输出。
2. `outputs/u1l_fidelity.md` **全文**（两个 32B 的源轨迹跑完时各先回传一次）。
3. `outputs/u1l_summary.md` **全文**。
4. `outputs/paper/arm_diagnose_u1l.md` 中的“插入有效性”表。
5. `outputs/paper/h1_pooled_u1l/h1_pooled.md` **全文**（RL 主表，按判据排除，不含只靠提示的模型）。
6. `outputs/paper/h1_u1l_only/h1_pooled.md` **全文**（U1L 七个智能体单独成表）。
7. 每个智能体源轨迹和 PCR 的耗时。

## 7. 2026-10-30 续跑方案（家服决策，替代第 4 节的剩余部分）

**决策**：采用方案 C 的改良版。不采用 B（不改 tensor parallel）。`research_7b` **不按不可测处理**，按协议第 7 节的修订从头重跑 PCR。

总体安排：

| 阶段 | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
| --- | --- | --- | --- | --- |
| 现在 → qwen 跑完 | 检索 + `research_7b` | 检索 + `research_7b` | `qwen25_32b_inst`（不动） | `qwen25_32b_inst`（不动） |
| 两者都跑完后 | 只有检索（单卡） | 32B worker | 32B worker | 32B worker |

第二阶段先 `research_32b`，再 `v03_32b_grpo`，三卡并行，每个约 9 小时（13.8 × 2 / 3）。

### 7.1 （已被 7.5 更新）现在做（`qwen25_32b_inst` 继续跑，不要动 GPU 2/3，也不要重启检索服务）

这次**例外**允许在 qwen 运行中拉取代码。原因：worker 只在启动和续跑时核对提交，正在跑的 worker 和合并步骤已经把代码载入内存，不受影响；qwen 跑完后的 PCR 分析会用新代码，新代码与它兼容。唯一的代价是：如果 qwen 中途崩溃，就不能 `--resume`，要在第二阶段换新目录重跑（概率很低）。

```bash
cd /root/develop/workspace/work2_byd
export U1_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python

# (1) 先确认 qwen 是怎么拉起的
ps -ef | grep run_u1l | grep -v grep
```

如果看到的是 `run_u1l.sh all`：只 `kill` **`all` 那一个 bash 进程的 PID**（不要动 `run_u1l.sh agent qwen25_32b_inst`、`work2.pcr run` 和 `work2.pcr worker` 这些进程），否则 qwen 跑完后它会自己去跑 `research_7b`，和下面手动拉起的冲突。kill 之后用 `nvidia-smi` 和 `tail outputs/u1/qwen25_32b_inst/pcr/logs/worker_0.log` 确认 qwen 还在跑。如果本来就是 `agent qwen25_32b_inst`，什么都不用做。

```bash
# (2) 拉代码、测试（只用 CPU）
git status --short && git fetch origin && git log --oneline -3
git rebase origin/feat/credit-testbed   # 有厂内本地提交（如 a217e64）时；没有本地提交就用 git pull --ff-only
bash scripts/run_u1l.sh test            # 期望没有 FAIL / ERROR

# (3) 旧的 research_7b PCR 改名保留（源轨迹和 pcr_plan 不动）
mv outputs/u1/research_7b/pcr outputs/u1/research_7b/pcr_attempt1_e0stop
ls outputs/u1/research_7b/              # 应有 sources/ pcr_plan/ pcr_attempt1_e0stop/

# (4) research_7b 在 GPU 0/1 上重跑（和检索服务同卡，和上次跑 research_7b 时一样）
export U1_GPUS=0,1
export U1_GPU_MEM=0.75                  # 已更正：7B 与检索同卡用 0.75（见 7.5）
nohup bash scripts/run_u1l.sh agent research_7b > outputs/u1l_research_7b_v2.log 2>&1 &
tail -f outputs/u1l_research_7b_v2.log
```

新代码遇到 E0 不复现时不会停，日志里该行末尾会出现 `E0_MISMATCH(excluded)`。偶尔出现几条是预期内的；合并时如果比例超过 2%，会报 `agent not measurable` 并停下，这时直接回报，不要重跑。

`research_7b` 跑完后（日志出现 PCR 分析结束），做跨运行比对（只用 CPU）：

```bash
"$U1_PYTHON" -m work2.pcr crosscheck --a outputs/u1/research_7b/pcr_attempt1_e0stop --b outputs/u1/research_7b/pcr \
  --output outputs/u1/research_7b/e0_crosscheck.json
```

### 7.2 两者都跑完后（`qwen25_32b_inst` 和 `research_7b` 的 PCR 分析都结束，四卡上没有推理进程）

```bash
# (1) 检索服务改到单卡：停掉现在的检索服务（PID 见 nvidia-smi），用 CUDA_VISIBLE_DEVICES=0 按原来的命令重启
#     等服务就绪后确认：GPU 0 只有检索，GPU 1/2/3 为空
nvidia-smi

# (2) 清理 32B 之前 OOM 留下的半截目录（冻结的是旧 GPU 列表和旧提交，不能续跑）
ls outputs/u1/research_32b/ outputs/u1/v03_32b_grpo/ 2>/dev/null
#     sources/ 存在但没有 sources/sources.jsonl：mv 成 sources_oom_attempt
#     pcr/ 存在但没有 pcr/states.jsonl：mv 成 pcr_oom_attempt

# (3) 三卡跑两个 32B（依次）
export U1_GPUS=1,2,3
export U1_GPU_MEM=0.92
nohup bash -c 'bash scripts/run_u1l.sh agent research_32b && bash scripts/run_u1l.sh agent v03_32b_grpo' \
  > outputs/u1l_32b.log 2>&1 &
tail -f outputs/u1l_32b.log
```

- 每个 32B 的源轨迹跑完后，照第 4 节先跑一次保真度表并**立即回传**。
- 检索服务单卡：~~检索调用大多命中缓存~~（**已更正，见 7.5**：每次检索都实时请求并逐条比对，所以同一个智能体的源轨迹和 PCR 必须在同一套检索配置下跑）。现在没有其他实验在用检索服务。
- qwen 跑完时 `research_7b` 如果还远没跑完（预计还要 4 小时以上），不要等：先在 GPU 2/3 上跑 `research_32b`（先做上面第 (2) 步的清理；`U1_GPUS=2,3`，检索不动）。等 `research_7b` **和** `research_32b` 都结束后，再把检索改到单卡，`v03_32b_grpo` 用 `U1_GPUS=1,2,3`。**任何推理进程在跑时都不要重启检索服务**，否则正在跑的状态会因检索请求失败而中断。

### 7.3 停止条件的变化

- `E0 control does not reproduce source continuation`：只有旧运行（第 7 节之前冻结的）才会这样报错；新运行改为记录并排除（见上）。
- `E0 control mismatch rate ... agent not measurable`：停止该智能体并回报，不要重跑。
- `RetrieverDriftError`：检索服务配置与该智能体生成源轨迹时不同。停下回报，不要重跑（见 7.5）。
- 其余停止条件不变。

### 7.4 本轮回传

1. `git rev-parse HEAD`；`research_7b` 新运行开始和结束时间。
2. 每个新跑完的智能体：`outputs/u1/<标签>/pcr/run_manifest.json` 中的 `control_mismatch_states`、`control_mismatch_rate`、`control_mismatch_ids`。有不复现的状态时，从 `pcr/states.jsonl` 里摘出这些行的 `control_mismatch` 字段原文。
3. `research_7b` 的 crosscheck 输出（终端打印的那段 JSON 和前 10 个不同的状态）。
4. 照第 6 节回传各智能体的 PCR 分析摘要和耗时；全部跑完后照第 4 节末尾做 `summary` 和 `pack`。

### 7.5 2026-10-30 中午修正（厂内 11:17 回传后，**以本节为准，覆盖 7.1(4) 和 7.2**）

**两处更正**（都是家服写错了，厂内的判断是对的）：

1. **7B 的 `U1_GPU_MEM` 用 0.75**，不是 0.92。vLLM 的这个比例是按整卡算给自己用的，不会扣掉同卡检索服务的占用；`research_7b` 第一次就是用 0.75 跑的。
2. **检索不走缓存。** `AuditedRetriever` 的每一次检索都实时请求服务，再和源轨迹记下的可见文档逐条比对，不一致就立刻报 `RetrieverDriftError`。检索服务用 `shard=True` + `useFloat16=True`，换了卡数，个别查询的近似并列结果就会换位置。所以：

> **规则：同一个智能体的源轨迹和 PCR 必须在同一套检索服务配置（同样的 GPU 卡数）下跑。只在两个智能体之间、没有任何推理进程时，才可以改检索服务的卡数。**

按这条规则核对：`qwen25_32b_inst` 的源轨迹和 PCR 都在 2 卡检索下跑，没问题；`research_32b`、`v03_32b_grpo` 的源轨迹还没跑，它们会在单卡检索下跑完源轨迹和 PCR，内部一致，也没问题。已完成的智能体的 PCR 都通过了逐条比对（漂移 0），结果有效。

**决策：厂内的方案 B（不停 qwen，不重新生成源轨迹）**，顺序如下：

1. **现在**：什么都不动，等 `qwen25_32b_inst` 跑完（包括 PCR 分析）。这段时间可以做两件只用 CPU 的事：
   - `git pull`（这次只改了文档；qwen 的 worker 不受影响）；
   - 回传第一次运行的结束时间：`ls -l --time-style=full-iso outputs/u1/research_7b/pcr_attempt1_e0stop/workers/*/run_manifest.json`，以及两份 worker 日志的最后 5 行。用来确认那次 E0 失败发生在 10-29 20:35 检索服务改到 2 卡**之前**（家服判断是这样，因为检索变化会报 `RetrieverDriftError`，而不是 E0 不复现）。
2. **qwen 跑完后**：停掉检索服务，按**生成 `research_7b` 源轨迹时（10-29 上午）的原命令**用 4 卡重启（不设 `CUDA_VISIBLE_DEVICES`，或设成 `0,1,2,3`），其余参数一字不改。等服务就绪，`nvidia-smi` 应看到每张卡约 9–10 GB。
3. `research_7b` 用四卡重跑（新目录；4 卡检索每卡约 10 GB，0.75 留得下）：

```bash
export U1_GPUS=0,1,2,3
export U1_GPU_MEM=0.75
mv outputs/u1/research_7b/pcr outputs/u1/research_7b/pcr_drift_110828_b 2>/dev/null || true   # 如果残留了 pcr/ 目录
nohup bash scripts/run_u1l.sh agent research_7b > outputs/u1l_research_7b_v3.log 2>&1 &
```

   - 前几十个状态内如果仍报 `RetrieverDriftError`，说明检索服务没有恢复成源轨迹当时的配置：停下，回报重启命令和报错，不要继续试。
   - 跑完后照 7.1 做 `crosscheck`（`--a outputs/u1/research_7b/pcr_attempt1_e0stop --b outputs/u1/research_7b/pcr`）。
4. **`research_7b` 跑完后**：停掉检索服务，改成单卡（`CUDA_VISIBLE_DEVICES=0`）重启，再照 7.2 第 (2)(3) 步跑两个 32B（`U1_GPUS=1,2,3`，`U1_GPU_MEM=0.92`）。两个 32B 的源轨迹和 PCR 全程不要再动检索服务。
5. 回传时加一项：**每个智能体的源轨迹和 PCR 分别在几卡检索下跑的**（论文附录要写明检索配置）。

# U1L 厂内执行指导：在 14B / 32B 智能体上检验 H1

更新：2026-10-28（同日修订：加入 Search-R1 32B GRPO、14B PPO、14B Instruct-GRPO）　分支：`feat/credit-testbed`
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
- `E0 control does not reproduce source continuation`：停止该智能体并回报；其余智能体可以用 `bash scripts/run_u1l.sh agent <标签>` 继续。
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

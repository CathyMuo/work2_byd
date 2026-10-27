# U1x 厂内执行指导：在其他训练配方上检验 H1

更新：2026-10-27　分支：`feat/credit-testbed`
协议（**先读第 2、4 节**）：[research/U1X_PROTOCOL.md](../research/U1X_PROTOCOL.md)

目的：U1 的 7 个智能体都来自 Search-R1 的训练配方。这次加入 ZeroSearch（含一个 **Llama** 模型）、AutoRefine、StepSearch 三种配方，共 4 个智能体，**只跑 H1**（源轨迹 + PCR），检验“恢复收益主要是均值回归”是否普遍成立。
**只做推理**，不训练。四卡合计约 5–7 小时。

---

## 0. 顺序与规则

1. 如果 `docs/PAPER_TABLES_GUIDE.md`（只用 CPU）还没做，可以先做，也可以和本指导的下载并行。
2. 开跑前用 `nvidia-smi` 确认四张卡空闲（用户的显卡保活程序按用户自己的习惯处理）。
3. 规则同前：
   - 不改代码；不修改 `research/DECISION_LOG.md` 和 `research/STATE.json`；
   - **运行期间不要提交任何东西**：HR2 第一次失败就是因为运行中提交了 messages.md，worker 会拒绝在代码变化后续跑；
   - 产物放在 `outputs/u1/<新标签>/`，不会覆盖 U1 的 7 个目录。

## 1. 拉代码、测试

```bash
cd /root/develop/workspace/work2_byd
git status --short && git fetch origin && git log --oneline -3
git switch feat/credit-testbed && git pull --ff-only
export U1_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
bash scripts/run_u1x.sh test
```

- 如果厂内有本地提交，`pull --ff-only` 会失败。这时用 `git rebase origin/feat/credit-testbed` 把本地提交接到最新代码后面。
- 期望：`Ran 177 tests ... OK`（数字以实际为准，不能有 FAIL / ERROR）。

## 2. 下载 4 个模型（内部镜像）

```bash
export HF_ENDPOINT=https://hf-cache.byd.com
for repo in \
  Alibaba-NLP/ZeroSearch_wiki_V2_Llama_3.2_3B \
  Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B \
  yrshi/AutoRefine-Qwen2.5-7B-Base \
  Zill1/StepSearch-7B-Base ; do
  hf download "$repo" --local-dir "/data/models/$repo"
done
df -h /data
```

逐个确认权重能读（镜像曾把 500 错误页当作文件内容返回）：

```bash
for d in /data/models/Alibaba-NLP/ZeroSearch_wiki_V2_Llama_3.2_3B /data/models/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B \
         /data/models/yrshi/AutoRefine-Qwen2.5-7B-Base /data/models/Zill1/StepSearch-7B-Base; do
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
print(d.rsplit("/", 1)[-1], len(files), "files", n, "tensors OK; chat_template:", bool(tok.chat_template), "; bos:", tok.bos_token)
PY
done
```

期望：
- 4 行都显示 `tensors OK`；
- Llama 那行 `chat_template: False; bos: <|begin_of_text|>`；
- 三个 Qwen 那行 `chat_template: True; bos: None`。

某个仓库在镜像上没有：跳过它并在回报里写明，**不要换成别的仓库**。

## 3. 本地配置（不提交）

```bash
"$U1_PYTHON" - <<'PY'
import json
base = json.load(open("configs/local.json"))
paths = {"zs_llama32_3b": "/data/models/Alibaba-NLP/ZeroSearch_wiki_V2_Llama_3.2_3B",
         "zs_qwen25_7b": "/data/models/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B",
         "autorefine_7b": "/data/models/yrshi/AutoRefine-Qwen2.5-7B-Base",
         "stepsearch_7b": "/data/models/Zill1/StepSearch-7B-Base"}
for label, path in paths.items():
    json.dump(dict(base, agent_checkpoint=path, checkpoint_version=path.split("/data/models/")[-1]),
              open("configs/local.u1_%s.json" % label, "w"), indent=2)
print("ok")
PY
bash scripts/run_u1x.sh check
```

期望：4 个标签都显示 `OK`。

## 4. 运行

```bash
export U1_GPUS=0,1,2,3
nohup bash scripts/run_u1x.sh all > outputs/u1x_all.log 2>&1 &
tail -f outputs/u1x_all.log
```

- `all` 按名单顺序跑：`zs_llama32_3b` → `zs_qwen25_7b` → `autorefine_7b` → `stepsearch_7b`。每个智能体依次是源轨迹（2125 题，贪心）、PCR（`h1` 臂组）、PCR 分析。
- 中断后重跑同一条命令即可续跑，已完成的部分会跳过。
- **第一个智能体的源轨迹跑完后**（日志里出现 `[zs_llama32_3b] PCR`），先看一眼格式失败率：

```bash
"$U1_PYTHON" -m work2.u1x --output outputs/u1x_fidelity_check.json
```

如果某个智能体格式失败 ≥ 10%，或者 EM 与论文差很多，**不要停**，照常跑完，按协议在汇总里单独报告。

全部完成后，先拉一次代码，让汇总表带上 10-27 新增的“与 P2 匹配的伤害”列。这时已经没有运行中的实验，可以拉取；只拉取，不要提交。然后再汇总：

```bash
git pull --ff-only
bash scripts/run_u1x.sh summary
bash scripts/run_u1x.sh pack
```

## 5. 停止条件

- `Source initial prompt differs from the frozen A1 prompt/tokenizer` 或 `Source observation format/content mismatch`：格式配置与源轨迹不一致。停止该智能体，回报日志末尾 30 行。
- `E0 control does not reproduce source continuation`：停止该智能体，回报。其他智能体可以继续：`bash scripts/run_u1x.sh agent <标签>`。
- `Code changed after run freezing`：运行中代码变了。不要手改任何 `run_plan.json`，回报。
- 检索漂移（`RetrieverDriftError`）：重跑同一条命令；同一个 query 反复失败就停下回报。
- vLLM 加载 Llama 报错：回报完整报错和 `pip show vllm transformers` 的版本，不要改代码。

## 6. 回传清单

1. `git rev-parse HEAD`；实际下载到的仓库；第 2 步 safe_open 检查的 4 行输出。
2. `outputs/u1x_fidelity.md` **全文**。
3. `outputs/u1x_summary.md` **全文**。
4. `outputs/paper/h1_pooled_u1x/h1_pooled.md` **全文**（U1 + U1x 全部 11 个智能体合并）。
5. 各智能体 `outputs/u1/<标签>/sources/run_manifest.json` 里的 `wall_seconds`，以及 `pcr/run_manifest.json` 里的耗时字段。
6. pack 生成的 feedback 包路径。

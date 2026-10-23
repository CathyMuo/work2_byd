# U1 厂内执行指导：多智能体 × 多数据集的干预式审计

更新：2026-10-22　分支：`feat/credit-testbed`
协议（**先读第 2–4 节**）：[research/U1_PROTOCOL.md](../research/U1_PROTOCOL.md)

目的：把已在 2 个检查点、2 个数据集上确认的两条结论（H1 恢复收益主要是均值回归；H3 证据类信用信号不测证据），扩到 7 个官方 Search-R1 智能体 × 5 个数据集，为冲 ACL 主会扩大证据面。
**只做推理**，不训练。

---

## 0. 顺序与前置条件

> **2026-10-23 FC1 决定**：`U1_PROMPT=bounded`（官方提示词没有带来显著差异）；主 7B 检查点换成 v0.3 7B PPO（它已在名单里）。
> **开跑前必须先和用户确认 GPU 安排**：U1 要独占四卡约 1.5–2 天，不能与用户自己的 GRPO 训练同时运行（FC1 期间曾因抢显存导致训练 OOM）。
> 下载模型时，请用 safe_open 做全量校验（镜像曾把 500 错误页当作文件内容返回）。

1. **FC1 必须先完成并回传**（`docs/FC1_FACTORY_GUIDE.md`）。U1 用哪种提示词（`U1_PROMPT=bounded` 或 `official`），由家服根据 FC1 的结果决定并通知，**厂内不要自己选**。
2. 下面第 1–3 步（拉代码、下载模型、准备数据和配置）**可以和 FC1 同时做**，第 4 步要等家服给出 `U1_PROMPT`。
3. 规则同前：
   - 不改代码；不在厂内修改 `research/DECISION_LOG.md` 和 `research/STATE.json`；
   - 运行期间不要提交；
   - 不覆盖旧 `outputs/`，本轮产物放在 `outputs/u1*`。

## 1. 拉代码、测试

```bash
cd /root/develop/workspace/work2_byd
git status --short && git fetch origin && git switch feat/credit-testbed && git pull --ff-only
export U1_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
bash scripts/run_u1.sh test
```

## 2. 下载智能体（内部镜像）

```bash
export HF_ENDPOINT=https://hf-cache.byd.com
for repo in \
  SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.3 \
  SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3 \
  SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3 \
  SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3 \
  SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3 ; do
  hf download PeterJinGo/$repo --local-dir /data/models/PeterJinGo/$repo
done
df -h /data
```

- 如果 v0.3 是 fp32 存储，7B 约 30 GB、14B 约 60 GB，总共约 150–200 GB。**先确认磁盘空间**。
- 7B PPO / GRPO v0.3 在 FC1 里已经下过，就不用重复下载。
- 镜像上没有的仓库：跳过并记录，不要换成别的仓库。

## 3. 准备数据和配置

### 3.1 新数据集（FlashRAG 格式：`id`、`question`、`golden_answers`）

```bash
hf download RUC-NLPIR/FlashRAG_datasets --repo-type dataset \
  --include "nq/test.jsonl" "musique/dev.jsonl" "bamboogle/test.jsonl" --local-dir /data/datasets/FlashRAG_datasets
```

然后在 `configs/local.json` 里加三个键（下面的路径按实际修改）：

```json
"nq_file": "/data/datasets/FlashRAG_datasets/nq/test.jsonl",
"musique_file": "/data/datasets/FlashRAG_datasets/musique/dev.jsonl",
"bamboogle_file": "/data/datasets/FlashRAG_datasets/bamboogle/test.jsonl"
```

如果厂内的 Search-R1 环境里已经有这些文件，直接用已有的路径，并在回报里写明来源。

### 3.2 每个智能体一份 local 配置（不提交）

```bash
"$U1_PYTHON" - <<'PY'
import json
base = json.load(open("configs/local.json"))
paths = {
  "v01_7b_ppo": base["agent_checkpoint"],
  "v03_3b_ppo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3",
  "v03_3b_grpo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.3",
  "v03_7b_ppo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3",
  "v03_7b_grpo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3",
  "v03_7b_it_grpo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3",
  "v03_14b_grpo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-em-grpo-v0.3",
}
for label, path in paths.items():
    json.dump(dict(base, agent_checkpoint=path, checkpoint_version=path.rsplit("/", 1)[-1]),
              open("configs/local.u1_%s.json" % label, "w"), indent=2)
PY
bash scripts/run_u1.sh check      # 每个标签都应显示 OK；拿不到的模型会显示 MISSING，运行时自动跳过
```

### 3.3 冻结 U1 题目（只做一次）

```bash
bash scripts/run_u1.sh freeze
sha256sum outputs/u1_fresh/samples.jsonl
```

**期望**：`Frozen U1 questions: {'hotpotqa': 500, '2wiki': 500, 'nq': 500, 'musique': 500, 'bamboogle': 125}; missing: none`（Bamboogle 以实际题数为准）。
某个数据集显示 missing 时照常继续，并在回报里写明。

## 4. 运行（等家服给出 `U1_PROMPT` 之后）

```bash
export U1_PROMPT=bounded      # 2026-10-23 FC1 决定
export U1_GPUS=0,1,2,3
nohup bash scripts/run_u1.sh all > outputs/u1_all.log 2>&1 &
tail -f outputs/u1_all.log
```

- `all` 按名单顺序逐个智能体运行：源轨迹 → PCR（H1，9 个臂）→ CT（H3，k=16）→ 判定。
- 每一步都会跳过已完成的部分，中断后重跑 `all` 即可续跑。也可以用 `bash scripts/run_u1.sh agent v03_7b_ppo` 单独跑一个智能体。
- **预计用时**（四卡）：
  - 3B：约 2–3 h；
  - 7B：约 4–6 h；
  - 14B：约 8–10 h；
  - 7 个智能体合计约 1.5–2 天。

  请每完成一个智能体，回报一次它的 `outputs/u1/<label>/verdicts.md`。

全部完成后：

```bash
bash scripts/run_u1.sh summary
bash scripts/run_u1.sh pack
```

## 5. 停止条件

- 某个智能体的 E0/E0 控制不一致（`E0 control does not reproduce source continuation`）：停止该智能体，回报日志。其他智能体可以继续。
- 检索漂移（`RetrieverDriftError`）：重跑同一条命令；同一个 query 反复失败就停下回报。
- 信号阶段的掩码自检报错：回报 transformers 版本，不要改成 sdpa。
- 14B 在 HF 打分阶段出现 OOM：回报即可，不要改代码。

## 6. 回传清单

1. `git rev-parse HEAD`；`U1_PROMPT` 的值；实际下载到的仓库，以及缺失的仓库和数据集。
2. `outputs/u1/u1_summary.md` **全文**。
3. 每个智能体的 `outputs/u1/<label>/verdicts.md`。
4. 各智能体 `sources/run_manifest.json` 中的 `wall_seconds`，以及 `ct/run_manifest.json` 中的 `mc_wall_seconds` 和 `retrieval`。
5. pack 生成的 feedback 包路径。

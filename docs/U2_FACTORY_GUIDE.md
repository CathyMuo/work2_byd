# U2 厂内执行指导：审计两个已发表的恢复机制（ReSeek、MR-Search）

更新：2026-10-23　分支：`feat/credit-testbed`
协议（**先读第 2、4 节**）：[research/U2_PROTOCOL.md](../research/U2_PROTOCOL.md)

目的：用 H1 的同状态安慰剂，检验两篇已发表方法自己的恢复机制：
- **ReSeek**：模型每次看完检索结果写 `<judge>Yes/No</judge>`，判 No 就再搜。我们在同一个状态上把判断强制成 No 或 Yes，与“同状态重采样”和“换文档顺序”比较修复率。
- **MR-Search**：第一回合答完后插入一段反思提示，再开一个回合。我们把反思提示与“只要求再答一次”“等长无意义文字”“不带历史重开一回合”比较。

**只做推理**，不训练。整体约半天（四卡）。

---

## 0. 顺序与规则

1. **GPU**：U1 正在独占四卡时，**不要**启动 U2 的 smoke 或正式运行。第 1–3 步（拉代码、下载、配置）可以在 U1 运行期间做；第 4 步起等 U1 跑完。
   启动前用 `nvidia-smi` 确认四张卡空闲。用户的显卡保活程序不是研究任务，如果它占着卡，按用户自己的习惯先停掉，实验全部结束后再恢复。
2. U2 用的是 **U1 冻结的题目**（`outputs/u1_fresh`），所以必须先完成 U1 的 `freeze`（U1 指导 §3.3）。
3. 推荐顺序：U1 → **U2** → HR2（`docs/HR2_FACTORY_GUIDE.md`）。
4. 规则同前：不改代码；不修改 `research/DECISION_LOG.md` 和 `research/STATE.json`；运行期间不要提交；不覆盖旧 `outputs/`，本轮产物都在 `outputs/u2/`。

## 1. 拉代码、测试

**注意**：U1 还在运行时**不要**切换或拉取代码（U1 的 worker 会检查代码是否变化，变了会拒绝续跑）。等 U1 跑完再做这一步。

```bash
cd /root/develop/workspace/work2_byd
git status --short && git fetch origin && git switch feat/credit-testbed && git pull --ff-only
export U2_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
bash scripts/run_u2.sh test
```

期望：`Ran 170 tests ... OK`（数字以实际为准，不能有 FAIL / ERROR）。

## 2. 下载 ReSeek 权重（内部镜像；可以在 U1 运行期间做）

```bash
export U2_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
export HF_ENDPOINT=https://hf-cache.byd.com
hf download TencentBAC/ReSeek-qwen2.5-3b-em-grpo --local-dir /data/models/TencentBAC/ReSeek-qwen2.5-3b-em-grpo
ls -la /data/models/TencentBAC/ReSeek-qwen2.5-3b-em-grpo
```

逐个确认权重能读，并检查分词器带 chat template（镜像曾把 500 错误页当作文件内容返回）：

```bash
"$U2_PYTHON" - /data/models/TencentBAC/ReSeek-qwen2.5-3b-em-grpo <<'PY'
import glob, sys
from safetensors import safe_open
from transformers import AutoTokenizer
d = sys.argv[1]
files = sorted(glob.glob(d + "/*.safetensors"))
assert files, "no safetensors files"
n = 0
for f in files:
    with safe_open(f, "pt") as h:
        for k in h.keys():
            h.get_tensor(k); n += 1
tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
print(len(files), "files,", n, "tensors OK; chat_template:", bool(tok.chat_template), "; vocab:", len(tok))
PY
```

期望：`... tensors OK; chat_template: True`。如果镜像上没有这个仓库、或 `chat_template: False`：**停下回报**，不要换成别的仓库。

## 3. 本地配置（不提交）

ReSeek 需要一份新的 local 配置；MR-Search 直接沿用 U1 的 `configs/local.u1_v03_7b_ppo.json` 和 `configs/local.u1_v03_3b_ppo.json`。

```bash
"$U2_PYTHON" - <<'PY'
import json
base = json.load(open("configs/local.json"))
path = "/data/models/TencentBAC/ReSeek-qwen2.5-3b-em-grpo"
json.dump(dict(base, agent_checkpoint=path, checkpoint_version="TencentBAC/ReSeek-qwen2.5-3b-em-grpo"),
          open("configs/local.u2_reseek_3b.json", "w"), indent=2)
print("ok")
PY
ls configs/local.u2_reseek_3b.json configs/local.u1_v03_7b_ppo.json configs/local.u1_v03_3b_ppo.json
```

## 4. 冻结 U2 题目（只做一次）

```bash
bash scripts/run_u2.sh prepare
sha256sum outputs/u2/plan/plan.json
```

期望：`Frozen N U2 questions (smoke 10); short datasets {'bamboogle': 125}`。N 约为 300×4 + 125 = 1325（以 U1 实际冻结的数据集为准；U1 缺哪个数据集，这里也会少）。

## 5. Smoke（工程检查，约 10–20 分钟）

```bash
export U2_GPUS=0,1,2,3
bash scripts/run_u2.sh check reseek reseek_3b
bash scripts/run_u2.sh smoke reseek reseek_3b
bash scripts/run_u2.sh smoke mr v03_7b_ppo
```

每个 smoke 结束会打印一份摘要（`outputs/u2/smoke_<方法>_<标签>/analysis/u2_report.md`）。请看“保真度”一节：
- ReSeek：`source_judge_rate` 应接近 1（≥ 0.8），`source_format_fail` 应很小；
- MR-Search：`ep2_answered_rate.reflect` 应 ≥ 0.8，`ep1_format_fail` 应很小。

smoke 只有 10 道题，**这些数字只用来查工程问题**。如果 judge 率明显低于 0.8，或第二回合大量不作答，先回报 smoke 摘要再继续。

## 6. 正式运行

```bash
nohup bash -c '
bash scripts/run_u2.sh run reseek reseek_3b &&
bash scripts/run_u2.sh run mr v03_7b_ppo &&
bash scripts/run_u2.sh smoke mr v03_3b_ppo && bash scripts/run_u2.sh run mr v03_3b_ppo
' > outputs/u2_all.log 2>&1 &
tail -f outputs/u2_all.log
```

- 预计：ReSeek 3B 约 1–2 h；MR-Search 7B 约 1.5–2 h，3B 约 1 h。
- 中断后重跑同一条命令即可续跑（已完成的运行会跳过）。
- **可选**（时间允许再做）：ReSeek 提示的“只给提示”对照，用 Search-R1 v0.3 3B PPO：
  ```bash
  bash scripts/run_u2.sh smoke reseek v03_3b_ppo && bash scripts/run_u2.sh run reseek v03_3b_ppo
  ```
  它的保真度大概率不过（模型没学过 judge），这正是要看的：报告即可。

每个运行结束都会自动生成 `outputs/u2/<方法>_<标签>/analysis/u2_report.md`，请每完成一个回报一次。

## 7. 停止条件

- `MR-Search user turn needs the Qwen2.5 default-system chat template`：分词器的 chat template 不是 Qwen2.5 默认格式。停止并回报 `tokenizer_config.json` 里的 chat_template 前 200 个字符。
- `A system message needs a chat template`：ReSeek 分词器没有 chat template，回到第 2 步检查下载。
- `Code changed after run freezing`：运行期间代码变了（有人 pull 或提交）。用新的 run 目录重跑或回报，不要手改 `run_plan.json`。
- 检索服务报错：重跑同一条命令；同一个 query 反复失败就停下回报。
- 任何 OOM：回报 `nvidia-smi` 和日志末尾，不要改代码。

## 8. 回传清单

1. `git rev-parse HEAD`；ReSeek 是否下载成功、文件数与张量数。
2. `outputs/u2/plan/plan.json` 的 sha256 和第 4 步打印的题数。
3. 每个运行的 `outputs/u2/<方法>_<标签>/analysis/u2_report.md` **全文**（smoke 的也附上）。
4. 每个运行 `run_manifest.json` 里的 `wall_seconds`、`branch_rollouts`、`retrieval`，MR-Search 的还有 `user_turns`。
5. `bash scripts/run_u2.sh pack` 生成的 feedback 包路径。

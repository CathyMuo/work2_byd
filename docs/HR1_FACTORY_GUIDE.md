# HR1 厂内执行指导：交接重放 pilot（Instruct ↔ Search-R1 7B PPO）

更新：2026-10-19　分支：`feat/credit-testbed`
协议（**先读第 2–4 节**）：[research/HANDOFF_PROTOCOL.md](../research/HANDOFF_PROTOCOL.md)

目的：把 Search-R1 7B PPO（R）相对 Qwen2.5-7B-Instruct（I）的 EM 增益，拆成三部分：**获取证据**、**利用证据**、**与证据无关**。
做法是从同一前缀把轨迹交给另一个策略继续，再加两个对照臂：
- 安慰剂证据：把已检索到的文档全部换成无关文档；
- 只交接证据：去掉思考文本，只保留查询和检索结果。

**只做推理**，不训练。与 AC1 的结果互不影响；AC1 的产物不要动。

---

## 0. 规则

1. 沿用 searchr1 环境、E5 + Wiki-18 索引和常驻 Retriever；不下载、不改代码、不覆盖旧 `outputs/`。本轮产物放在 `outputs/hr1_*`。
2. 两个模型都已在厂内：
   - R：`/root/develop/workspace/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo`
   - I：`/root/develop/workspace/models/Qwen2.5-7B-Instruct`
3. 每张卡同一时间只加载一个模型。程序分三个阶段依次换模型：
   - r1：R 采样自己的轨迹并自续写；
   - i2：I 采样自己的轨迹，自续写，并接手 R 的轨迹；
   - r3：R 接手 I 的轨迹。
4. 不要自行下 GO/NO-GO 结论。

## 1. 拉代码、测试、准备 local 配置

```bash
cd /root/develop/workspace/work2_byd
git status --short && git fetch origin && git switch feat/credit-testbed && git pull --ff-only
git rev-parse HEAD
export HR_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
bash scripts/run_handoff.sh test      # 期望 OK（家服 149 项；厂内另有本地看板测试）
"$HR_PYTHON" - <<'PY'
import json
c = json.load(open("configs/local.json"))
c["instruct_checkpoint"] = "/root/develop/workspace/models/Qwen2.5-7B-Instruct"
json.dump(c, open("configs/local.handoff.json", "w"), indent=2)
print("instruct_checkpoint", c["instruct_checkpoint"], "| R", c["agent_checkpoint"])
PY
```

`configs/local.handoff.json` 不会提交（已在 `.gitignore` 里）。

## 2. 冻结题目（只做一次）

```bash
bash scripts/run_handoff.sh freeze
sha256sum outputs/hr1_fresh/samples.jsonl
```

脚本会自动排除 pilot、A1 scale、SI1 的题，以及 **AC1 的 1000 道新题**（`outputs/ac1_fresh/samples.jsonl`）。
**期望**：`Frozen 300 HR1 questions; excluded … historical ids from N files.`。厂内应该是 4 个文件：pilot、A1 scale、SI1 dp4，以及 AC1 新题。回传这一行、sha256，以及 `selection_manifest.json` 中的 `exclusions`。

## 3. 检查与 smoke（8 题，约 10–20 分钟，三个阶段都会跑一遍）

```bash
export HR_GPUS=0,1,2,3
bash scripts/run_handoff.sh check
bash scripts/run_handoff.sh smoke
```

**期望**：
- 依次打印 `phase r1 / i2 / r3`；
- 最后打印 `Merged 8 questions; … rollouts; template identical 8/8.`；
- 然后打印 smoke 分析。smoke 只有 8 题，数值没有意义。

**停止条件**：
- `Missing local instruct_checkpoint`：第 1 步的 local 配置没有生成好。
- `template identical` 不是 8/8：说明两边的统一模板没有生效，停止并回报。
- `CUDA out of memory`：在命令后追加 `--gpu-memory-utilization 0.7` 重跑，并在回报里说明。
- worker 非零退出：回传 `outputs/hr1_smoke/logs/*_worker_*.log` 的最后 50 行。中断后直接重跑同一条命令，已完成的阶段和题目会跳过。

## 4. 正式 pilot（300 题，预计约 1.5–2.5 小时）

```bash
nohup bash scripts/run_handoff.sh run > outputs/hr1_pilot.log 2>&1 &
tail -f outputs/hr1_pilot.log
bash scripts/run_handoff.sh analyze
bash scripts/run_handoff.sh pack
```

## 5. 回传清单

1. `git rev-parse HEAD`、GPU / 显存设置。
2. `outputs/hr1_pilot/analysis/hr1_report.md` **全文**。
3. `outputs/hr1_pilot/analysis/hr1_report.json` 中的 `gates`、`components`、`shares_of_gain_end`、`format_fail_rate`。
4. `outputs/hr1_pilot/run_manifest.json` 中的 `phase_wall_seconds`、`rollouts`、`template_identical`、`instruct_native_template_identical`、`retrieval_audit_changes` / `retrieval_audits`。
5. pack 生成的 feedback 包路径。

## 6. 输出的含义

| 字段 | 含义 |
| --- | --- |
| `gain_end` | 在各自轨迹走完之后，R 自续写与 I 自续写的 EM 之差，即总增益 |
| `acquisition` | 把 I 的证据换成 R 的证据，平均能多出多少 EM（获取） |
| `utilization` | 同样的证据，换 R 来读和作答，平均能多出多少 EM（利用） |
| `evidence_free` | 证据全部换成无关文档后，R 仍然比 I 多出的 EM（与证据无关的部分） |
| `G_A1` | 自重放无偏，且拿掉证据确实会让 R 掉 ≥ 5 pp |
| `G_A2` | 三个分量的 CI 半宽都 ≤ 3 pp |
| `G_A3` | 与证据无关的份额 ≥ 30%，或者获取 / 利用中有一个份额 ≥ 60% 且 CI 下界 ≥ 50% |

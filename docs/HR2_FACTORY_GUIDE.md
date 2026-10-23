# HR2 厂内执行指导：在 Search-R1 v0.3 上重做交接重放

更新：2026-10-23　分支：`feat/credit-testbed`
协议（**先读第 2、4 节**）：[research/HR2_PROTOCOL.md](../research/HR2_PROTOCOL.md)　HR1 的指导（流程相同）：`docs/HR1_FACTORY_GUIDE.md`

目的：HR1 用的 v0.1 7B PPO 不比 Instruct 强，没法拆。FC1 显示 v0.3 7B PPO 比 Instruct 高约 16 pp，所以在 v0.3 上重做：
把 RL 带来的 EM 增益拆成 **获取证据 / 利用证据 / 与证据无关** 三部分，并比较 PPO 与 GRPO 是不是从不同的地方获益。
**只做推理**，不训练，不需要新下载。

---

## 0. 顺序与规则

1. 推荐顺序：U1 → U2（`docs/U2_FACTORY_GUIDE.md`）→ **HR2**。U1 或 U2 在用 GPU 时不要启动 HR2。
   启动前用 `nvidia-smi` 确认四张卡空闲。用户的显卡保活程序不是研究任务，如果它占着卡，按用户自己的习惯先停掉，实验全部结束后再恢复。
2. HR2 的题目要排除 U1 的题，所以必须在 **U1 的 `freeze` 之后**再冻结 HR2（`outputs/u1_fresh/samples.jsonl` 要存在）。
3. 规则同前：不改代码；不修改 `research/DECISION_LOG.md` 和 `research/STATE.json`；运行期间不要提交；本轮产物都在 `outputs/hr2_*`。

## 1. 拉代码、测试

（如果刚做完 U2 的第 1 步，代码已是最新，跳过 pull，只跑 test。）

```bash
cd /root/develop/workspace/work2_byd
git status --short && git fetch origin && git switch feat/credit-testbed && git pull --ff-only
export HR_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
bash scripts/run_hr2.sh test
```

## 2. 本地配置（不提交）

两份配置，只差 R 的检查点；I 都是 HR1 用过的 Qwen2.5-7B-Instruct。

```bash
"$HR_PYTHON" - <<'PY'
import json
base = json.load(open("configs/local.json"))
inst = "/root/develop/workspace/models/Qwen2.5-7B-Instruct"
for r, path in {"ppo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3",
                "grpo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3"}.items():
    c = dict(base, agent_checkpoint=path, checkpoint_version=path.rsplit("/", 1)[-1], instruct_checkpoint=inst)
    json.dump(c, open("configs/local.hr2_%s.json" % r, "w"), indent=2)
    print(r, "R =", path, "| I =", inst)
PY
```

路径以 FC1 / U1 实际下载的位置为准；Instruct 的路径以 HR1 实际用的为准（见 `configs/local.handoff.json`）。

## 3. 冻结 HR2 题目（只做一次）

```bash
bash scripts/run_hr2.sh freeze
sha256sum outputs/hr2_fresh/samples.jsonl
"$HR_PYTHON" -c "import json;m=json.load(open('outputs/hr2_fresh/selection_manifest.json'));print([e['path'] for e in m['exclusions']])"
```

期望：`Frozen 600 HR questions; excluded ... historical ids from ... files.`；排除清单里应包括 `outputs/ac1_fresh/samples.jsonl`、`outputs/hr1_fresh/samples.jsonl`、`outputs/u1_fresh/samples.jsonl`。少了哪一个就停下回报（说明对应目录不在默认位置；可以用 `bash scripts/run_hr2.sh freeze --exclude <实际路径>` 补上，但要先删掉 `outputs/hr2_fresh` 再重冻）。

## 4. 主运行：R = v0.3 7B PPO

```bash
export HR_GPUS=0,1,2,3
bash scripts/run_hr2.sh check ppo
bash scripts/run_hr2.sh smoke ppo        # 8 道题，约 10 分钟；会打印摘要
nohup bash scripts/run_hr2.sh run ppo > outputs/hr2_ppo.log 2>&1 &
tail -f outputs/hr2_ppo.log
```

- smoke 摘要里 `template_identical` 应等于题数，四个 combo 的格式失败率应很低；摘要里会出现 `evonly_placebo` 臂的曲线。
- 正式运行约 3–4 小时，分三阶段（r1 → i2 → r3），中断后重跑同一条命令即可续跑。
- 完成后：

```bash
bash scripts/run_hr2.sh analyze ppo
```

## 5. 第二运行：R = v0.3 7B GRPO

```bash
bash scripts/run_hr2.sh smoke grpo
nohup bash scripts/run_hr2.sh run grpo > outputs/hr2_grpo.log 2>&1 &
# 完成后
bash scripts/run_hr2.sh analyze grpo
bash scripts/run_hr2.sh compare           # PPO vs GRPO，判 G-A3(iii)
bash scripts/run_hr2.sh pack
```

GRPO 平均只检索约 1.9 次，预计约 2 小时。

## 6. 停止条件

- `Code changed after run freezing`：r1 之后有人改了代码或提交（HR1 的 v2 运行就是这样被拦下的）。不要手改 `run_plan.json`，用新的 `--run-dir` 重跑或回报。
- `Previous phase ... incomplete`：上一阶段没跑完，重跑同一条 `run` 命令即可。
- `Missing local instruct_checkpoint`：第 2 步的配置没生成好。
- `No eligible null documents`：回报对应 sample_id。
- 任何 OOM：回报 `nvidia-smi` 和日志末尾，不要改代码。

## 7. 回传清单

1. `git rev-parse HEAD`；`outputs/hr2_fresh/samples.jsonl` 的 sha256 和排除清单。
2. `outputs/hr2_ppo/analysis/hr1_report.md` 和 `outputs/hr2_grpo/analysis/hr1_report.md` **全文**（文件名沿用 HR1，内容标题是 HR2）。
3. `outputs/hr2_ppo/compare/hr2_compare.md` **全文**。
4. 两个运行 `run_manifest.json` 里的 `phase_wall_seconds`、`rollouts`、`template_identical`、`retrieval_audit_changes`。
5. pack 生成的 feedback 包路径。

# FC1 厂内执行指导：评测保真度检查（检查点版本 × 提示词）

更新：2026-10-20　分支：`feat/credit-testbed`
协议（**先读第 1–3 节**）：[research/FIDELITY_PROTOCOL.md](../research/FIDELITY_PROTOCOL.md)

为什么做：HR1 显示 7B PPO 不比 Instruct 强，AC1 中 7B 也比 3B 差。我们发现：
- 现用的 7B 是 Search-R1 **最早的 v0.1**，而 3B 是 v0.3；
- 我们的提示词与官方评测模板不同。

写论文之前，必须先确认 harness 没有系统性压低智能体的表现。本轮**只做贪心推理**，每个检查点 × 提示词跑 1000 题，单次约 5–10 分钟。

---

## 0. 先说三件事（关于上一轮）

1. **决策已记录**：HR1 判 NO-GO；CT1 的“E 作为真值主量”早在 10-19 就已处理（CT1 v1 判 NO-GO，E 侧结论改由 AC1 的 H3 在新题上预登记确认），不需要再记录。
2. **请不要在厂内修改 `research/DECISION_LOG.md` 和 `research/STATE.json`**。这两个文件由家服维护，厂内改了，下次 pull 时会冲突。
   厂内的执行记录请写到厂内自己的文件里（例如 `tmp/messages.md`），或者直接贴在回报里。上一轮厂内本地的 `9a919a2` 也在改这两个文件，pull 前请先 `git stash` 或另存。
3. **evonly 读法更正**：`evonly` 臂保留了真实观测，去掉的是**思考文本**。HR1 里 evonly 下 R 明显差于 I（−8.9 pp），说明 R 更依赖自己的思考轨迹（也可能是去掉 `<think>` 后 R 的分布偏移更大），**不是**“R 更依赖真实观测”。

## 1. 拉代码、测试

```bash
cd /root/develop/workspace/work2_byd
git status --short           # 如有本地修改（包括 DECISION_LOG / STATE），先 git stash
git fetch origin && git switch feat/credit-testbed && git pull --ff-only
git rev-parse HEAD
export FC_PYTHON=/root/develop/persistent-miniconda3/envs/searchr1/bin/python
bash scripts/run_fidelity.sh test     # 期望 OK（家服 153 项）
```

## 2. 下载 v0.3 检查点（内部镜像），并在回报里写明实际仓库名

```bash
export HF_ENDPOINT=https://hf-cache.byd.com
for repo in SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3 SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3; do
  hf download PeterJinGo/$repo --local-dir /data/models/PeterJinGo/$repo
done
```

下载完逐个确认权重能读（同 3B 时的 `safe_open` 检查）。

**镜像上没有某个仓库时**：跳过它并在回报里写明，不要换成别的仓库名。

## 3. 准备 local 配置（不提交）

```bash
"$FC_PYTHON" - <<'PY'
import json
base = json.load(open("configs/local.json"))
for name, path in {"fc_v03_7b_ppo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3",
                   "fc_v03_7b_grpo": "/data/models/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3",
                   "fc_instruct_7b": "/root/develop/workspace/models/Qwen2.5-7B-Instruct"}.items():
    c = dict(base, agent_checkpoint=path, checkpoint_version=path.rsplit("/", 1)[-1])
    json.dump(c, open("configs/local.%s.json" % name, "w"), indent=2)
    print(name, path)
PY
bash scripts/run_fidelity.sh check     # 每个标签都应显示 OK；local.second.json 是 AC1 时的 3B 配置
```

## 4. 运行

```bash
export FC_GPUS=0,1,2,3
nohup bash scripts/run_fidelity.sh all > outputs/fc1_all.log 2>&1 &
tail -f outputs/fc1_all.log
```

- `v01_7b_ppo` 和 `v03_3b_ppo` 的 bounded 结果直接复用 AC1 已有的源轨迹（`outputs/ac1_*/sources`），不重跑。
- 其余 8 次运行，每次 1000 题贪心，约 5–10 分钟，共约 1–1.5 小时。中断后重跑 `all` 会跳过已完成的部分。
- **运行期间不要提交任何东西**：worker 会检查 commit 是否变化，变了就拒绝运行（HR1 v2 就是这么失败的）。

完成后：

```bash
bash scripts/run_fidelity.sh summarize
bash scripts/run_fidelity.sh pack
```

## 5. 停止条件

- `Missing local checkpoint config.json`：检查点路径不对或没下完。
- 检索漂移（`RetrieverDriftError`）：重跑同一条命令；同一个 query 反复失败就停下回报。
- 某个检查点的格式 / 长度失败率明显偏高（> 30%）：照常跑完，在回报里点名。

## 6. 回传清单

1. `git rev-parse HEAD`；实际下载到的仓库名与路径；哪些仓库镜像上没有。
2. `outputs/fc1/fc1_report.md` **全文**。
3. 顺带补回 AC1 的两项数据：
   - `outputs/ac1_7b_ppo/pcr/analysis/pcr_report.json` 与 `outputs/ac1_3b_ppo/pcr/analysis/pcr_report.json` 中 `recovery.semantic.selected_fragile` 的完整内容；
   - 两个检查点 `ct/analysis/ct1_report.json` 中 `signals` 的 `vs_A` 部分。
4. pack 生成的 feedback 包路径。

## 7. 判定规则（家服按此记录，厂内不下结论）

| 规则 | 看哪一行 | 含义 |
| --- | --- | --- |
| D1 | `v01_7b_ppo_official-v01_7b_ppo_bounded`、`v03_3b_ppo_official-v03_3b_ppo_bounded` | ≥ 3 pp 且下界 > 0 → 我们的提示词压低了表现，需要用官方提示词重跑 AC1 |
| D2 | `v03_7b_ppo_*-v01_7b_ppo_*` | ≥ 3 pp 且下界 > 0 → 主 7B 检查点换成 v0.3 |
| D3 | `*_official-instruct_7b_official` | RL 相对 Instruct 的增益，只报告 |
| D4 | `v03_7b_grpo` 的格式 / 长度失败率 | < 10% → 可作为 AC1 第三检查点 |

# FAM1 厂内运行指南

协议：`research/FAM1_PROTOCOL.md`（含第 11 节修订 1）。脚本：`scripts/run_fam1.sh`。分析：`work2/fam1.py`。只做推理。

## 0. 硬约束

- **检索**：沿用主表的四卡部署，不为腾显存改动；每次检索都会与源轨迹逐条比对。
- **代码冻结**：每个运行在 `run_plan.json` 里记下 `repository_commit`，worker 会核对 HEAD。所以**运行期间不要提交，也不要 pull**，否则 worker 会报 “Code changed after run freezing”。
- **Python**：用 searchr1 环境，`export U1_PYTHON=<searchr1 的 python>`。

## 1. 本地配置（9 个）

- 沿用现有 schema（与 `configs/local.u1_v01_7b_ppo.json` 相同的键）。只需按模型改：
  - `agent_checkpoint`：`/root/develop/workspace/models/<目录名>`；
  - `checkpoint_version`：该仓库在 hf-cache 上的 commit sha。
- 检索相关的键（`retrieval_url`、`corpus_file`、`index_file` 及对应 version、`actual_index_type`）照抄主表。
- 文件名为 `configs/local.u1_<标签>.json`，标签见 `configs/fam1_agents.json`：
  - `v01_l3b_{ppo,grpo,it_ppo,it_grpo}`
  - `v01_q3b_{ppo,grpo,it_ppo,it_grpo}`
  - `r1s_l8b`
- 生成后先跑：

```bash
bash scripts/run_fam1.sh test     # 全部单元测试（无 GPU）
bash scripts/run_fam1.sh check    # 9 个 local 配置都能找到 config.json
```

## 2. 题集 E（只用 CPU，跑一次）

```bash
bash scripts/run_fam1.sh freeze-e configs/local.<U1 冻结题目时用的那个文件>.json
```

- 这个 local 文件须含 `hotpotqa_file`、`two_wiki_file`、`nq_file`、`musique_file`。
- 输出到 `outputs/fam1_e/`（`samples.jsonl` 与 `selection_manifest.json`）。
- 程序会遍历 `outputs/**/samples.jsonl` 做排除。若有哪个同名文件不是题目文件，它会报错并指出路径；确认后用 `--ignore` 跳过（直接调用 `python -m work2.fam1 freeze-e ... --ignore <路径>`），并在回传中说明。
- 请回传 `selection_manifest.json` 中的 `counts` 和 `shortfall`。

## 3. 运行

建议先跑 smoke（几个状态，输出不进入分析）：

```bash
bash scripts/run_fam1.sh smoke v01_l3b_ppo
bash scripts/run_fam1.sh smoke r1s_l8b
```

正式运行，二选一：

```bash
bash scripts/run_fam1.sh all            # 协议 4.2 的顺序跑 9 个智能体，然后题集 E（已冻结时）
bash scripts/run_fam1.sh agent v01_l3b_ppo
bash scripts/run_fam1.sh e v01_l3b_ppo  # 题集 E：只跑源轨迹 + H1
```

- `agent` 依次执行：源轨迹 → H1 → 打印插入有效性 → AS1 → U1N。
- 结果在 `outputs/u1/<标签>/{sources, pcr, as1, u1n}`；题集 E 的结果在 `{sources_e, pcr_e}`。
- 中断后重跑同一命令会续跑（`--resume`）；已完成的步骤会跳过。
- 环境变量：
  - `FAM1_GPUS`：默认 `0,1,2,3`；
  - `FAM1_GPU_MEM`：默认 0.75，3B 和 8B 都够用。
- 日期门：`r1s_l8b` 和题集 E 的代码已在 11-28 前推送，日期门满足。

## 4. 汇总与回传（只用 CPU）

```bash
bash scripts/run_fam1.sh summary   # outputs/fam1_fidelity.json，outputs/paper/fam1/{fam1.json, fam1.md}
bash scripts/run_fam1.sh pack
```

- `fam1.md` 给出 F1–F8 的机械值；决定由家服写进 `DECISION_LOG.md`。
- 请回传 `fam1.md`、`fam1.json`、`fam1_fidelity.json`、题集 E 的 manifest，以及每个智能体打印的插入有效性。

## 5. 看板

看板（`docs/BOARD_GUIDE.md`）已配套更新，需要在**开始 FAM1 运行之前**拉取最新代码并重启看板（`pkill -f "work2.board --host"` 后按看板指南重新启动）：

- FAM1 的 9 个智能体，以及所有题集 E 运行（含 `zs_llama32_3b` 的），实验列显示为 “FAM1”；阶段列显示“AS1 阳性对照”“U1N 自然失败”“题集 E · H1 安慰剂分支”等。
- “当前批次”换成 FAM1 的 46 项：9 个智能体各 4 步，加题集 E 的 10 个 H1 运行。

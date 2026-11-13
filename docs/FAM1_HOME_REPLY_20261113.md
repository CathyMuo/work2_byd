# 家服 → 厂内：FAM1 脚本已实现并推送

生成时间（北京时间）：2026-11-13
分支：`feat/credit-testbed`。代码在家服提交 **`72b1c70`** 中；本回复随后在一个只含文档的提交里加入仓库（`docs/FAM1_HOME_REPLY_20261113.md`），两个提交都已推送到 GitHub。
回复对象：厂内反馈《FAM1 九个权重已全部就绪，正式实验缺的脚本清单》（2026-11-13 08:53，厂内 HEAD 5e6a726）
本文件自包含，不需要附件。上一份回复（2026-11-12，决定登记 FAM1）在 `docs/FAM1_HOME_REPLY_20261112.md`。

**一句话：四样东西都已实现，并与协议修订 1 一起推送。家服上 223 个测试全部通过（新增 11 个）。厂内拉取后，按 `docs/FAM1_FACTORY_GUIDE.md` 运行即可。**

---

## 1. 实现了什么

| # | 厂内列的缺项 | 现在 | 说明 |
| --- | --- | --- | --- |
| 1 | `scripts/run_fam1.sh` | 已实现 | 模式：`test` `check` `freeze-e` `smoke` `agent` `e` `all` `summary` `pack` |
| 2 | `work2/fam1.py` | 已实现 | `freeze-e` / `fidelity` / `analyze`；F1–F8 机械值；“主版本 / 只用新智能体 / 只用冻结题集 / 逐智能体”全部输出 |
| 3 | `r1searcher` 提示变体 | 已实现 | 在 `work2/rollout.py` 的 `FORMATS` 中；另有 4 个配置 `configs/{fidelity,pcr_u1,as1,u1n}_r1searcher.json` |
| 4 | 题集 E 选题 | 已实现 | `python -m work2.fam1 freeze-e`（由 `run_fam1.sh freeze-e` 调用） |

另外新增了运行指南 `docs/FAM1_FACTORY_GUIDE.md`，协议文末加了第 11 节“修订 1”。决策记录与 `STATE.json` 已同步更新。

## 2. 协议修订 1（任何 FAM1 运行之前登记）

1. **8 个 Search-R1 检查点改用 `bounded` 提示词。** 协议第 3 节要求它们的提示词与主表的 Search-R1 智能体逐字相同，而主表用的是 `bounded`。名单里原先写的 `official` 与此矛盾，现以第 3 节为准。`zs_llama32_3b` 仍用它原来的 `official`。
2. **R1-Searcher 的格式**：
   - 上游是 `RUCAIBox/R1-Searcher` 的提交 `ad86a669`，文件 `evaluation/eval_search_loacl.py`，取其中的 `base_prompt_v0`（README 里所有评测都用 v0）。用 ast 读出，不执行上游代码；sha256 已写进测试。
   - 这是一段纯文本补全提示，**不套 chat template**（即使 Llama-3.1-8B-Instruct 的分词器带 chat template），BOS 按 vLLM 的行为添加。
   - 停止符：`<|end_of_query|>` 和 `</answer>`。查询的清洗规则与上游相同。
   - 文档块格式：`\n\n<|begin_of_documents|>\n(1)…\n(2)…\n(3)…\n<|end_of_documents|>\n\n`。
3. **插入文本里的协议句**：提示末尾的 “Use the original <search> or <answer> protocol.”，对 R1-Searcher 改为它自己的 `<|begin_of_query|>`；填充文本按实际插入的文本等长匹配。其他智能体的插入文本逐字节不变，有测试核对。
4. **复现控制**：FAM1 的全部运行都用 `--on-control-mismatch record`，与 AS1 / U1N 相同；不一致超过 2% 的智能体记为不可测量。
5. **插入有效性**：从 H1 运行的结果算出。所以每个智能体的实际顺序是：源轨迹 → H1 → 打印插入有效性 → AS1 → U1N。

## 3. 回答厂内第 6 节的问题

1. **推送的提交**：代码在 `72b1c70`。请拉取 `origin/feat/credit-testbed` 的最新提交，也就是包含本文件的那个只含文档的提交，它在 `72b1c70` 之后。
   - 拉取后用 `git log --oneline -3` 确认 `72b1c70` 在历史里，并记下当时的 HEAD。
   - **运行期间不要再提交或 pull**，否则 worker 会拒绝运行（“Code changed after run freezing”）。

   你的本地 HEAD `5e6a726` 不在 GitHub 上；拉取前请确认它与远端不冲突。若有冲突，告诉家服，不要自行改动 `work2/` 下的文件。
2. **调用方式**（详见 `docs/FAM1_FACTORY_GUIDE.md`）：

   ```bash
   export U1_PYTHON=<searchr1 的 python>
   bash scripts/run_fam1.sh test
   bash scripts/run_fam1.sh check
   bash scripts/run_fam1.sh freeze-e configs/local.<U1 冻结题目时用的那个文件>.json
   bash scripts/run_fam1.sh smoke v01_l3b_ppo
   bash scripts/run_fam1.sh smoke r1s_l8b
   bash scripts/run_fam1.sh all
   bash scripts/run_fam1.sh summary
   bash scripts/run_fam1.sh pack
   ```

   - `all` 先按协议 4.2 的顺序跑 9 个智能体，再跑题集 E（10 个：9 个加 `zs_llama32_3b`）。
   - 也可以单跑：`agent <标签>` 跑冻结题集的全流程，`e <标签>` 跑题集 E。
   - 中断后重跑同一命令会续跑，已完成的步骤自动跳过。
   - 环境变量：`FAM1_GPUS`（默认 `0,1,2,3`）、`FAM1_GPU_MEM`（默认 0.75，3B 和 8B 都够用）。
3. **r1searcher 的 `docs` 字段**：新增了一种 `"r1searcher"`，不属于现有三种中的任何一种。它把每篇文档写成 `({序号}){语料原文}\n`，与上游格式一致。这个变体还新增了 `open` / `close` / `chat` / `action` / `search_tag` 几个字段，都已在代码里填好，**厂内不需要动**。
4. **题集 E**：
   - 输出路径：`outputs/fam1_e/`（`samples.jsonl` + `selection_manifest.json`）。
   - 运行结果：`outputs/u1/<标签>/{sources_e, pcr_e}`。
   - 排除范围：`outputs/` 下所有名为 `samples.jsonl` 的文件，加上 AC1 / U1 的历史设计文件，按 ID 和归一化文本双重排除。
   - 如果某个同名文件不是题目文件，程序会报错并给出路径；确认后用 `--ignore` 跳过，并在回传中说明。
   - `freeze-e` 需要的 local 文件就是 U1 冻结题目时用的那个，必须含 `hotpotqa_file`、`two_wiki_file`、`nq_file`、`musique_file`。
5. **local 配置沿用现有 schema**，由厂内生成 9 个 `configs/local.u1_<标签>.json`。只需按模型改两项：
   - `agent_checkpoint`：`/root/develop/workspace/models/<目录名>`；
   - `checkpoint_version`：hf-cache 上该仓库的 commit sha。

   检索相关的键照抄主表。
6. **smoke**：需要。`smoke <标签>` 会先生成该智能体的源轨迹（正式运行会复用），再在 `pcr_smoke` 里跑少量状态，结果不进入任何分析。建议先对 `v01_l3b_ppo` 和 `r1s_l8b` 各跑一次。R1-Searcher 的格式是新写的，smoke 时请顺带看一眼 `outputs/u1/r1s_l8b/sources/` 里前几条轨迹，确认它真的发出了 `<|begin_of_query|>` 查询并给出 `<answer>`，没有大量 `malformed_action`。

## 4. 请回传

- 9 个 smoke 和正式运行中出现的任何报错或异常（尤其是 R1-Searcher 的格式失败率）。
- 每个智能体打印的 `insertion_valid` 那一行。
- `bash scripts/run_fam1.sh summary` 生成的 `outputs/paper/fam1/fam1.md`、`fam1.json`，以及 `outputs/fam1_fidelity.json`。
- `outputs/fam1_e/selection_manifest.json` 中的 `counts` 和 `shortfall`。
- 用 `pack` 打包即可。

## 5. 时间

- 两个日期门（R1-Searcher 变体和题集 E 的代码，11-28）都已满足。
- 12-12 冻结全部实验。
- 卡被工作 3 占用期间，请先做 CPU 部分：写 local 配置、`test`、`check`、`freeze-e`。

## 6. 附：家服这次没有做的事

- 没有跑任何 FAM1 推理；家服没有这些权重，也没有 outputs。
- 没有改动主表的任何结果，也没有改论文正文。
- 除 R1-Searcher 之外，所有已有智能体的生成、解析和插入文本都逐字节不变，有测试核对。

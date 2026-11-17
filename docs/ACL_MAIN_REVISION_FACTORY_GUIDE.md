# ACL 主会修订的厂内 CPU 执行指南

日期 2026-11-17。目标是补齐逐题可复核结果、评分盲审材料和敏感性分析。本轮使用已有运行，只做 CPU 分析。原运行目录、历史协议和结果保持原样。家服负责修改论文、判断证据边界及制作匿名复现包。

## 1. 先盘点数据与运行版本

在厂内 `work2_byd` 根目录执行。若分支有厂内提交或脏工作区，无需合并即可查看指南。复用当前可验证的分析环境和既有文件。缺少某组数据时继续处理其余组，记录 `missing` 和已查位置。

```bash
git fetch origin feat/credit-testbed
git show FETCH_HEAD:docs/ACL_MAIN_REVISION_FACTORY_GUIDE.md
git rev-parse HEAD
git status --short --branch
```

| 数据 | 优先查找位置 | 要一起保留的证据 |
| --- | --- | --- |
| 原始 H1（14 个插入有效检查点） | `outputs/u1/<agent>/pcr/states.jsonl` | `pcr/run_manifest.json`、`pcr_plan/plan.json`、对应题目 ID |
| 自然失败与每题一次策略 | `outputs/u1/<agent>/u1n/states.jsonl` | `u1n/run_manifest.json`、`u1n_plan/plan.json`、其中 `sources_path` 指定的源轨迹 |
| FAM1 冻结题集及 E | `outputs/u1/<agent>/{pcr,pcr_e}/states.jsonl` | plan、manifest、`outputs/fam1_e/selection_manifest.json`、已有 `outputs/paper/fam1_v2/fam1.json` |
| ReSeek 原生门控 | `outputs/u2/reseek_reseek_3b/records.jsonl` | `run_manifest.json`、`outputs/u2/plan/plan.json`、已有分析报告 |
| 旧 CPU/评分导出 | `outputs/paper/review_cpu/` 和 `outputs/paper/arm_diagnose_all.*` | 原报告与分析版本，旧 `semantic_cases.jsonl` 只作线索 |

生成 `inventory.json`。逐文件记相对路径、SHA-256、行数、manifest 中的运行和代码提交、预期校验值与实际状态。保留 `run_commit`、`analysis_commit` 和 `selection_commit` 的来源。FAM1 回传叙述中的 HEAD 与结构化产物记录不同，按各自字段记录并核对，勿把三者合并成一个字段。对完成运行核验 manifest 所记哈希。缺文件、校验失败、带控制失配排除的运行分别记录。缺文件也可查旧备份目录，避免回填或覆盖原产物。

## 2. 先复跑现有 CPU 汇总

`U1_PYTHON` 指向厂内已有 Search-R1 Python。新的输出目录可改名，确保每次修订保留独立目录。

```bash
PY=${U1_PYTHON:-python}
OUT=outputs/paper/acl_main_revision_20261117
mkdir -p "$OUT"
"$PY" -c 'from work2.as1u1n import AGENTS; print(len(AGENTS), *AGENTS)'
```

确认原始 H1 与 U1N roster 齐全后，运行原有分析入口。若缺少部分检查点，先单列可核验的子集和缺失名单，子集结果不得标作原始完整面板。

```bash
"$PY" -m work2.review_cpu --u1-root outputs/u1 --resamples 2000 --output "$OUT/original"
mapfile -t LABELS < <("$PY" -c 'from work2.as1u1n import AGENTS; print("\n".join(a for a in AGENTS if a not in {"v03_3b_grpo", "v03_7b_grpo"}))')
test "${#LABELS[@]}" -eq 12
"$PY" -m work2.review_cpu --u1-root outputs/u1 --labels "${LABELS[@]}" --resamples 2000 --output "$OUT/original_without_small_grpo"
```

FAM1 的完成产物已在厂内时，可重建其上轮汇总，原有 roster 与题集保持不变。

```bash
"$PY" -m work2.fam1 analyze --root outputs/u1 --resamples 2000 --output "$OUT/fam1_rebuild"
```

以上脚本给出对照基线，未覆盖本指南全部新表。新计算放到 `$OUT` 下的本地分析脚本，复用 `work2.review_cpu` 的 `Boot`、`h1_rows`、`policy_rows` 和 `work2.fam1` 的相应数据加载与比较函数。回传脚本或完整伪代码和测试记录。可先用 10–20 道题核对每行、正负号、分母及完整臂，再对全量正式输出。

## 3. 评分口径重算与盲审材料

### 3.1 同一批数据跑四格评分敏感性

固定有效重放的公共状态和臂可用性。对每个状态先按原版代码同时取 `raw`（原始精确匹配）和 `sem`（以 clean 为参照的字符串启发式）结果。`y0` 用干净续写。D 是 `y0=1` 且首次重排错，S 是 `y0=1` 且首次重排仍对。按行指定用于分组的视图和用于计分的视图。

| D/S 何时用哪种判定 | 各分支结果用哪种判定 | 读法 |
| --- | --- | --- |
| raw | raw | 全程原始 EM |
| raw | sem | 固定 raw 状态集合，只换分支评分 |
| sem | raw | 固定 sem 状态集合，用原始 EM 评分 |
| sem | sem | 历史预注册口径 |

输出 `scoring_matrix.json`。原始 14 组、FAM1 Llama 主面板及原始 13 Qwen 参照分别报告。每格写 D、S、全状态数、唯一题数、D/S 交叉迁移计数、按臂的绝对率、提示减 P2 的额外修复、额外损害、全状态 EM 差及区间。另报与 P1 的修复差，两个提示分列。全状态以同一状态集合计分，换 D/S 规则本身不能改变全状态结果。不同题集的面板差只是描述，按第 4 节另报同题集检验。保留预注册结果，四格与人工核验标为审稿后分析。

### 3.2 导出独立盲审包

抽样单位为可重放的 `(题目 ID, 检查点, 轮次)`。对 `clean_a`、首次重排、复查、反思和各自 P2 建立答复列表，优先从这六臂齐全的状态抽样。以这组臂中是否有 `raw→sem` 改判分为变化和未变化两类。默认抽约 **120 个有改判状态与 60 个无改判对照状态**。两类都覆盖原始组与 FAM1 Llama，记录模型族、数据集、检查点、改判臂和 0→1/1→0 方向，避免集中于单个 32B。跨两个面板重复的同一状态只向标注者呈现一次，底表保存两组归属。先确定互斥抽样层、固定种子 `20261117`、各层总数与抽样数。稀疏层做全取或先合层并记录，随后用哈希抽样。在抽样前写下规则。不得读取案例答案后再更换样本。

- `blind_items.jsonl` 供两位标注者独立使用。只列随机编号、题干、公开金标与允许别名、待判答案。每个答案单独随机排列。隐藏臂名、模型名、`raw/sem` 标签、候选分组、前后答案对应关系及论文结论。
- `blind_key.private.jsonl` 由协调人保管。保存随机编号到状态/臂的映射、抽样层、层大小、纳入概率与答案重复使用关系。按 `correct`、`incorrect`、`uncertain` 判定答案相对题意及金标是否正确。把无法消解的题意或别名歧义留为 `uncertain`。
- 两份独立标签和仲裁结果分别保存。完成后报告一致率、分歧数量、误改与保留误差的方向，并用随机样本的纳入概率或层权重报告整体误判率。若重算评分后的 D/S，先在有完整独立判定的状态上做抽样加权估计和按题聚类区间，保留 `uncertain` 的上下界。人工样本之外仍报告 raw/sem 四格，避免把抽样核验误当成全量人工评分。
- **厂内 agent 可以完成导出、去标识、抽样检查及自动评分辅助。** 两位真人的标签是独立证据。没有真人时，返回 `human_status: pending` 和可直接交付的盲审包。模型评分若做，另标 `model_assisted`，公开所用模型、提示和版本。

旧 `semantic_cases.jsonl` 因对特定检查点过采样，只用于核对导出字段和说明历史差错，按新的盲审方案另外抽样。

## 4. 敏感性重算与样本流

除已有注册读法外，新增表均标为审稿后敏感性。使用 2,000 次、固定种子 42 的百分位区间。按数据集分层，聚类单位 `(dataset, sample_id)`，同题的轮次、检查点、两组和各臂在同一次重抽中一起进出。同一对比的数值统一由同一输出写入正文和附录。输出未四舍五入的点估计、上下界、单位、唯一问题数、状态数和缺失数。区分 `prompt − filler` 的准确率差与 `harm(prompt) − harm(filler)` 的额外损害正号。注明评估是固定检查点上的题目抽样推断。

1. **原始组 H1 和每题一次策略。** 对全状态及题目级策略分别给三种权重：按状态（或智能体–问题对）、智能体等权、训练配方等权（每配方内智能体等权）。全 14 和去掉 `v03_3b_grpo`、`v03_7b_grpo` 的 12 个都做。H1 双口径报告 review/reflect 对 P2；问题级主读数用 raw EM，列提示对 P2、提示对不干预、P1 对不干预，保留实际搜索次数及后续生成 token。出 `weighting.json` 与 `policy_flow.json`。
2. **对齐 10,156 的分母。** 按 `agent × dataset` 给候选可检索、计划选入、具有有效首轮状态、重放排除、按原哈希窗口补入未检索题、最终计入数。核对 `最终 = 有效首轮状态 + 补入的未检索题`，未检索题各臂结果沿用源轨迹。列表细项要能解释原论文的 10,156 与名义上限 10,150 的关系。若仍不一致，回报确切行、字段和差值。
3. **拆开输出状态故障与事实错答。** 用已有状态字段与 `work2/arm_diagnose.py` 核对分类。各臂至少单列：格式／动作错误、生成或上下文截断、其他未给有效答案、有效答案但事实错误、正确。映射列出原始状态键，分母包括所有可用配对状态。输出各错误类的 `prompt − P2` 概率差，并检查其和等于全状态 EM 差的相反数。被原流程判定为不满足精确复现的状态在样本流单列。出 `error_breakdown.json`。
4. **FAM1 留一与严格同题配对。** 主 Llama 面板逐个留一，明确列去 `r1s_l8b`，双口径报告修复、损害与全状态。再做三对三严格 Search-R1 v0.1 配方（基座 PPO、基座 GRPO、Instruct PPO），以及去基座 PPO 后的两对。每对先按同题目、同检索轮次对齐完成的重放，公布每对的交集、未匹配数和重叠后 D/S。分别报告冻结题集与 E，可保留合并列。主对齐视图保持各检查点在共享输入上的 D/S 定义，并另外写状态成员差异。若报告两模型共同属于 D 或 S 的交集，单列为不同的估计对象。修复、损害、全状态各做 Llama 减 Qwen，按题联合抽样。表中登记过的非平衡三对四作为原口径保留。出 `fam1_sensitivity.json`。

原始 `review_cpu` 里已有部分 H1 权重和首轮策略数据，便于复用。若原始包只剩汇总，点估计与区间的可重建性单独标记。无法重建的区间写 `null` 并给原因，禁止用逐智能体区间平均替代联合重抽。对 AS2 的协议句和插入位置，只报告现有 review 对照的适用范围。

## 5. ReSeek 先清点，记录齐全时做离线检查

`records.jsonl` 的首轮状态储存 `stage_a_judge`、`nat`、`force_yes`、`force_no` 的配对续写、状态、EM 与额外搜索次数。利用同一首轮和同一 Stage A 样本下的已保存分支重建三种单次决策，再用 `nat` 作运行参照：

1. 用原生判断分配 Yes/No，通过同索引的强制分支读出结果。
2. 在每个数据集内维持相同的 No 比率，独立重新分配给这些首轮样本。可按每个数据集的 `p(No)` 对两臂结果取期望，固定分层及抽样口径。记录实际后续搜索次数，触发率相同与后续总检索成本相同是两个指标。
3. 用强制 `Yes` 关闭首次重搜动作。原生判断缺失的样本按预先声明的共同规则处理，各策略共用相同缺失处理，保留缺失率和相同处理下的样本数。无首轮检索的题各策略沿用其原始答案。

按自然源轨迹答对或答错分层，同时报告全题 EM、搜索次数及格式失败。各题先对其可比采样重复取平均，再按题重抽。旧记录如果没有生成 token，标记缺失。该比较是用于设计后续实验的探索性离线检查。现有产物缺失就回报缺口。是否需要独立新题、重跑推理或增设预算对照由家服结合结果另定。

## 6. 回传清单

- 先回传 `inventory.json` 和简短状态：每个数据源的 `complete / partial / missing / checksum_mismatch`、本轮分析提交与输出目录。
- 本轮结束后回传 `summary.md`、`scoring_matrix.json`、`weighting.json`、`policy_flow.json`、`error_breakdown.json`、`fam1_sensitivity.json`。若已完成 ReSeek 离线比较，一并回传 `reseek_offline.json`。每张表附运行来源、分母、区间和本次用到的脚本或计算规则。
- 回传盲审抽样统计和 `human_status`。原始记录、逐题授权材料和盲审包经现有允许渠道交接。公开 Git 分支只放指南与分析代码，运行产物、厂内路径细节、访问凭据和未公开论文均留在各自受控位置。
- 若只能文字回传，先贴清单、主结果表、置信区间和样本流，再贴缺失原因及可交接文件列表。人工标签待补时，CPU 分析照常回报。

本轮完成条件：已核验的数据组交付可复算的 CPU 结果，缺失的数据组有具体文件和原因，盲审材料可交给两位标注者。保留历史结果及版本，不执行 GPU 或训练任务。

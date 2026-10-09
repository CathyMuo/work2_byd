# A3：E2–E4 单轮干预工程 Pilot 执行文档

本轮先跑 **10 题、turn 1/2、最多 20 状态**。已经实现准备审查表、离线校验、
E0/E0 + E1 + 获批 E2/E3/E4 续写、配对指标和检索审计。首次仍是工程验收，
不是完整 200 题研究实验。开发服务器只做 CPU 检查；H800 执行由厂内完成。

## 1. 报告对应的研究判断

A1 留下 586 个完整搜索前缀；A2 报告 20/20 精确控制、历史一致和有效排列，
支持开始单轮替换工程验证。排序 Pilot 中 t=1 的 EM 降低只是案例线索，不能
称为显著结果。详见 [A2 分析](../research/A2_REPORTED_RESULTS.md)。

研究问题是检索观测能否改变后续搜索目标，以及预算内能否恢复。E1 用于控制
排序敏感性；E2/E3/E4 分别检查干扰信息、缺失桥接证据和冗余反馈。不能只看
最终 EM，更不能把“下一条 Query 不同”直接标成 OffTarget。

这轮先确认：替换是否符合定义、是否只改变一个观测、前缀和预算是否一致、
续写和检索是否稳定。语义偏移/恢复及正式配对置信区间留到标注协议审查后。
不开展 RL，不处理被停止的其他 GRPO 训练。

## 2. 使用现有环境和资源

沿用厂内已有的 Search-R1 环境、`configs/local.json`、常驻 8000 Retriever、
官方 7B PPO、E5/Wiki-18/**Flat**、已转换的 Hotpot JSON、2Wiki、冻结 pilot。
不重新抽题、转换数据、下载模型或启动第二个检索服务。

| 保留项 | 本轮要求 |
| --- | --- |
| A1 | `outputs/a1_clean_200/{run_manifest.json,baseline_metrics.json,clean_rollouts.jsonl}` |
| 冻结题目 | A1 manifest 指向的原始 `pilot.jsonl` 路径及 SHA256 未变 |
| A2 | `outputs/a2_sanity/{run_manifest.json,replay_metrics.json,replay_sanity.jsonl,retrieval_audit.json}` |
| 模型/索引/语料 | 与 A1 一致，不搬迁、不修改；模型索引修复后的身份以 A1 为准 |
| 推理 | 自动继承 A1 的后端/解码/8192/512/4 次预算，TP=1、显存比例 **0.75** |
| 环境版本 | vLLM 0.6.3、torch 2.4.0+cu121、transformers 4.47.1；由 runtime 精确核验 |

无需重建环境；推荐清单保留在 [RECOMMENDED_ENVIRONMENT.md](RECOMMENDED_ENVIRONMENT.md)。
`configs/diagnostic.json` 的当前默认值不会覆盖历史 A1 参数。

## 3. 更新并检查修复

在已配置的 Search-R1 环境中：

```bash
cd /root/develop/workspace/work2_byd
bash scripts/update.sh
git rev-parse HEAD
python -m unittest discover -s tests -q
python - <<'PY'
from work2.run_replay import index_family
assert index_family('Flat (e5_Flat.index); NOT HNSW') == 'flat'
assert index_family('HNSW') == 'hnsw'
assert index_family('Flat / HNSW') is None
print('index declaration regression passed')
PY
```

不要编辑 A1/A2 manifest 或 existing local 描述文本。旧 A2 的 `hnsw` 标签被
保留为历史记录，准备脚本在新产物 `provenance.index_label_erratum` 写更正。
仅接受已知旧提交 `a17f1f3...` 的该 Flat→hnsw 标签 bug；其他类型矛盾会报错。
源文件不变且控制/服务未变时，无需重跑 A1/A2。

F1 校验继续使用精确相等。若再失败，完整报错会给出 recorded、recomputed
及类型；保留原始文件和报错反馈，不加容差、不改存储值。

## 4. 生成审查包：CPU 离线操作

```bash
python -m work2.prepare_interventions \
  --source-run outputs/a1_clean_200 \
  --a2-run outputs/a2_sanity \
  --output outputs/a3_review/proposals.json
cp outputs/a3_review/proposals.json outputs/a3_review/reviewed.json
```

脚本核验 A1 原始指标/冻结题目，重算 A2 指标、比较原始控制分支及历史续写，
检查 A2 绑定的 A1 文件哈希和 audit 无漂移。该步骤不调用服务、不加载模型。
有问题立即停止，反馈报错；不通过修改历史文件来过检查。

选题固定为 A1 顺序每数据集前 5 题，先选 ID 再判断 turn 1/2 可达性，含答错题。
不因干预候选不好而补选其他题。审查包包含问题、金标答案、编号金标支持事实、当前 Query、
以前的可见文档、当前 Top-3、原始候选 Top-20；不展示 clean 答案、EM 或后续
续写。以前已看过总体结果，不能声称严格双盲；要求在新干预结果前冻结审查。

## 5. 逐条完成独立审查

只编辑 `reviewed.json` 内每个 case 的 `reviews`，其他字段必须原样保留。
每个可达状态的 E2/E3/E4 都填写 `approve` 或 `skip`；不能留 `pending`。
审查人可为研究者或独立于被测 Search-R1 的审查 agent，不能让被测模型自己
成为唯一裁判。即使由其他 agent 初审，也应核查引用和事实角色；自动 lexical
命中只帮助筛选，不能证明“桥接”或“非关键”。

| 字段 | 填写方式 |
| --- | --- |
| `decision` | `approve` / `skip` |
| `reason` | 获批的资格依据或跳过原因，不能空 |
| `reviewer` | 可追溯的审查者标识；可达状态即使跳过也需填写 |
| `independent_of_studied_agent` | 获批必须 `true` |
| `replacement_slot` | 原始 Top-3 的 **零起始**位置 0/1/2 |
| `candidate_rank` | 原始候选列表 **从 1 开始计数**的 rank 4–20 |
| `bridge_fact_ids` | E3 用金标支持事实数组的零起始索引；其他通常 `[]` |
| `redundant_fact_ids` | E4 重复支持的事实索引；其他通常 `[]` |
| `semantic_checks` | 按下表填写获批所需的 true，不能直接全勾 |
| `evidence` | 引用被替换/保留/候选 passage 的 document_id、原句、事实索引，以及子目标关系解释 |

### 三类干预的资格

| 条件 | 审查要求 | `semantic_checks` 必需项 |
| --- | --- | --- |
| E2 | 替换非关键 passage；候选与当前 Query 相关但不支持答案，且不损失关键事实 | `query_relevant`, `critical_facts_preserved`, `replacement_has_no_answer`, `removed_passage_noncritical` |
| E3 | 原 passage 包含桥接事实；该事实在替换后和更早历史中均缺失；候选无答案/桥接，其余关键事实保留 | `query_relevant`, `bridge_role_verified`, `bridge_absent_from_prior_history`, `replacement_has_no_answer_or_bridge`, `other_critical_facts_preserved` |
| E4 | 替换非关键 passage；候选真实重复某个**仍保留的 passage**已支持的事实；关键事实不损失 | `query_relevant`, `critical_facts_preserved`, `redundant_evidence_verified`, `removed_passage_noncritical` |

E3 需在 evidence 说明“已知实体→需要连接的实体/关系→下一子目标”的链条，
引用承载连接的原文，说明其他两篇和此前历史为何没有等价桥接。金标 supporting
fact 不自动等于桥接；若桥接角色不清楚，写 `skip: bridge_role_uncertain`。
若 turn 1 已观察过该桥接，turn 2 即使移除同一篇也不符合当前的“新增缺口”定义，
写 `bridge_already_observed`。不得用这种状态宣称恢复了被移除的未知证据。

本轮机器筛选采用保守的严格子集：E2/E4 不得丢失原可见集合的已知事实文本代理；
E2 候选不得命中金标事实代理；E3 所声明的被移除事实必须原来命中、替换后
不命中、历史也不命中，且不能暗中一起删掉其他已知事实，候选不得命中任一
金标事实代理；E4 冗余事实必须在候选和保留 passage 中都命中文本代理。
等价改写可能过不了这一严格筛选，照实跳过，不伪造原句或修改文档。未来扩大
语义覆盖需单独修订并冻结资格协议，不能看结果后在本轮调整。

程序只能核验代理和所填的审查声明，无法自动证明语义判断正确；人工证据审计
仍是研究前提。若支持句无法定位、候选不足、无法确定非关键文档，都应跳过。

### 长度与真实文档约束

只用 A1 **该 Query 的原始 rank 4–20** 候选，替换恰好一个位置，顺序不变。
候选 ID 与全文不能和原 Top-3 重复；不合成、删改、拼接 passage，不增加检索
预算、不修改 corpus/index。后续所有检索使用现有正常服务，不缓存返回结果。

替换 passage 的字符数必须为原 passage 的 **0.5–2.0 倍**，这是看本轮新结果前
冻结的工程粗筛，不是 token 精确匹配或研究显著性阈值。运行时另存目标观测
token 差及完整输入/生成成本。若长度差较大，结果只能作为工程案例；正式研究
需审查更严的长度控制，不能声称当前已消除了长度混杂。

### 审查条目示例（仅字段示例，不是真实合格样本）

```json
{
  "decision": "approve",
  "reason": "经独立核验：rank 3 不承担关键事实，rank 7 与 Query 相关但不支持答案",
  "reviewer": "填写真实审查者标识",
  "independent_of_studied_agent": true,
  "replacement_slot": 2,
  "candidate_rank": 7,
  "bridge_fact_ids": [],
  "redundant_fact_ids": [],
  "semantic_checks": {
    "query_relevant": true,
    "critical_facts_preserved": true,
    "replacement_has_no_answer": true,
    "removed_passage_noncritical": true
  },
  "evidence": "填写真实 doc ID 和原句；说明保留哪几个 fact ID，以及候选为何只相关而非答案证据"
}
```

此示例用于 E2 的结构说明；不可直接复制为结果。`skip` 条目只需 final decision、
具体 reason 和 reviewer，其他原始占位字段可保留。不可达条目已自动标 skip。
全部完成后保留原 `proposals.json`，不要把审查文件提交到公开仓库。

## 6. 离线验收与 GPU 执行

```bash
python -m work2.prepare_interventions \
  --source-run outputs/a1_clean_200 --a2-run outputs/a2_sanity \
  --validate outputs/a3_review/reviewed.json
python -m work2.run_interventions \
  --source-run outputs/a1_clean_200 --a2-run outputs/a2_sanity \
  --reviews outputs/a3_review/reviewed.json --check-only
```

允许部分条件全部跳过，但至少需要一项获批才能执行模型。全部无资格时，直接
反馈资格/跳过原因，不补选样本或强行批准。`check-only` 不做运行时模型、tokenizer、
资源或服务检查；正式运行会逐项核验并重构原始 token 前缀。

```bash
mkdir -p outputs/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 bash scripts/run_a3.sh \
  --source-run outputs/a1_clean_200 \
  --a2-run outputs/a2_sanity \
  --reviews outputs/a3_review/reviewed.json \
  --run-dir outputs/a3_engineering \
  2>&1 | tee outputs/logs/a3_engineering.log
```

只有至少一个条件获批的状态才跑模型。每状态先两次 clean，交替控制次序；
逐 token 控制不一致或 clean 与 A1 历史不同则整批停止，尚未执行该状态的 E1/E2–E4。
控制通过后跑 E1，再按状态循环轮换 E2/E3/E4 的执行顺序，降低固定顺序干扰。
最多每状态 6 分支，共 120 个续写；资格不足时更少。

每支从 A1 同一个原始 token 前缀恢复，全局已用预算不重置。只变目标观测，
干预前轨迹保持原样，后续检索 Top-3 正常返回。第 4 次成功检索后仍可答题，
请求第 5 次检索则预算耗尽。live 相同 Query 结果漂移或基础设施错误均停止。

## 7. 产物、读法和反馈

| 文件（`outputs/a3_engineering/`） | 用途 |
| --- | --- |
| `run_manifest.json` | 新提交/依赖、原始参数、资源身份、审查 SHA256、旧标签勘误、完成/失败状态 |
| `frozen_reviews.json` | 运行前冻结的审查内容；运行后不再改 |
| `paired_rollouts.jsonl` | 控制、E1 和获批干预的完整实际分支；所有跳过状态也记录 |
| `intervention_metrics.json` | 各条件×目标轮的计划/可达/获批/完成分母、skip 原因、配对 EM/F1 与成本 |
| `retrieval_audit.json` | live Query、参考与漂移检查 |

E2/E3/E4 资格集合不同，不直接跨条件比较平均 EM。`same_subset_e1_em` 使用
该条件同一批获批状态，便于检查它们是否超过排序对照；仍是描述性结果。
partial/failed 不把未跑分支计为 0 分，也不把跳过当成恢复成功。
`later_search_availability` 单列双方是否继续搜索；下一次 Query 字符串变化率只在
双方确实继续搜索时计算并给出分母，仍不能叫 OffTarget。指标中的
`semantic_offtarget`、`semantic_recovery` 为 null，这是后续需要独立标注的内容。
本轮不产生把 20 状态当 20 独立题的置信区间或统计结论。

请直接在对话反馈：新 commit、manifest status、20 状态中的各条件资格数/skip
原因、控制一致数、检索漂移、各条件×turn 指标，以及 2–4 个真实案例。案例引用
sample_id、turn、替换 slot/rank/doc ID、bridge/redundant fact ID、evidence，
列出 clean/E1/干预的下一次 Query、后续文档、最终答案、状态和成本。先审计这些
案例能否说明语义目标变化/缺失事实再获取，而非仅答案翻转。

可选打包（不用环境报告，不要求上传完整轨迹）：

```bash
python scripts/pack_feedback.py \
  --include outputs/a3_engineering/run_manifest.json \
  --include outputs/a3_engineering/intervention_metrics.json \
  --include outputs/a3_engineering/retrieval_audit.json \
  --include outputs/a3_engineering/frozen_reviews.json \
  --include outputs/logs/a3_engineering.log
```

打包工具有单文件大小上限，超限时原文件留厂内并反馈精简审查/案例摘要即可。
运行目录已存在会拒绝覆盖；失败后保留目录，修复后用新 `--run-dir`。本轮不实现
断点续跑，也不删状态后重算“通过率”。不要在看到结果后编辑原审查再复用。

## 8. 本轮之后

开发侧审查资格覆盖、严格代理限制、长度/成本、控制和真实案例。若某条件无
合格候选，先分析是否定义过严或冻结候选不足，不能解释成模型没有偏移。
若通过工程审查，再实现/冻结 200 题诊断扩展、独立语义 OffTarget/Recovery
标注协议、以问题为单位聚类的配对区间及独立确认集。Recovery 应允许其他文档
提供等价事实，不能只看原 doc ID。B 恢复基线和 C/RL 仍需计划中的研究证据与 GO。

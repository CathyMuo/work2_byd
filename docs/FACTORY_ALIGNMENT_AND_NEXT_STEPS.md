# 厂内分支对齐与下一轮 CPU 审计

日期：2026-10-12。用户已明确以开发服务器公开主线为准。厂内 `f4a4bb5` 源码
未提供；本轮继承其**报告的实验进展**，并对齐已知 `replay_intervention()` API，
不会声称合入全部 35 文件。未获得源码的 run_b/run_scaleup/merge/看板入口仍在
厂内备份分支；当前公开版本提供独立 CPU 审计入口。环境/模型/数据不重配。

结果判断详见 [报告审查](../research/FACTORY_SCALE_REVIEW.md)。当前动作是使用
原有产物检查原始配对、资格和语义，不重新执行 1000 题 A1/A3/B，不训练 RL。

## 1. 一次性对齐分叉代码

普通 `git pull --ff-only` 不能覆盖厂内已提交的 17 个分叉提交。首次执行：

```bash
cd /root/develop/workspace/work2_byd
GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
  -c remote.origin.url=https://github.com/CathyMuo/work2_byd.git fetch origin main
git show origin/main:scripts/align_public.sh > /tmp/work2_align_public.sh
bash /tmp/work2_align_public.sh --use-fetched-ref
git rev-parse HEAD
python -m unittest discover -s tests -q
```

脚本先拒绝未提交修改/未忽略文件，再创建 `factory-backup/<UTC时间>-<旧SHA>`
分支，最后把 main 切到已获取的公开主线并使用公开 HTTPS remote。旧提交可用
`git log factory-backup/<实际名称>` 查看；原始产物中的旧 commit 不更改。
不执行 clean、不删除 ignored outputs/、不改 local.json。若有未提交文件，
先按本机已有方式保存它们，脚本不会强制丢弃。tmp/ 已纳入公开主线 ignore。

日后普通主线更新继续使用 `bash scripts/update.sh`。保留备份分支直至审计完成。
报告中曾有 GitHub 出口错误；fetch 失败时保存日志，命令不会先切走旧代码。

## 2. 先确认原始 JSONL 的结构

以下源文件保持只读：

| 内容 | 报告路径 |
| --- | --- |
| A1 1000 题 raw | `outputs/a1_clean_scale500/clean_rollouts.jsonl` |
| A3 合并 raw | `outputs/a3_scale500_merged/` 中实际的 paired JSONL |
| B | `outputs/b_scale500/`，含 `b_recovery.json` 与原始分支 |
| 旧 L3/S1 审计 | `outputs/l3_case_audit_scale.json`、`outputs/recoverability_report_scale.json` |

报告未给出 A3 raw 文件名和完整 schema，先定位：

```bash
rg --files outputs/a3_scale500_merged -g '*.jsonl'
# 下列路径按上一条输出填写；若实际名称不同，不重命名原文件：
python -m work2.audit_pairs \
  --a3-rollouts outputs/a3_scale500_merged/paired_rollouts.jsonl --schema-only
```

schema-only 只输出键名/分支名/condition，不打印正文，也不读完所有轨迹。
审计支持公开共享 rollout schema：顶层 sample_id、dataset、target_turn、status、
branches；clean_a/clean_b 两个独立真实控制；每分支保存 steps、prefixes、
condition、gold_answer、final_answer、em/f1 等原字段。若仅分支名称不同，可
用显式 `--branch E1_permutation=实际键名` 等参数；字段不同则先反馈 schema-only
输出，我们再实现格式适配。**不猜 schema、不补造 raw token、不复制 clean_a
假扮第二次控制，也不为过检查修改原文件。**

审计要求原始 A1 的全体 sample IDs×turns 1/2 与 A3 状态完全对应。expected
不能用当前 A3 的行数反推。重叠、漏跑、伪 unreachable、源标签改变、score 不符、
干预前历史改变、后续可见文档不是正常 candidate Top-3、控制不一致都会报错。
对于工程版没有获批干预的状态，all_conditions_skipped 明确保留。

## 3. 重算同资格集合的配对效应

```bash
python -m work2.audit_pairs \
  --a1-rollouts outputs/a1_clean_scale500/clean_rollouts.jsonl \
  --a3-rollouts outputs/a3_scale500_merged/paired_rollouts.jsonl \
  --target-turns 1 2 --resamples 2000 --seed 42 \
  --output-dir outputs/semantic_audit
```

该命令不加载模型、不调用 Retriever、不占 GPU；需要读入原始 A1/A3 轨迹。
若分支 condition 字段规范，会自动识别；映射参数用于明确旧条件名称，不改变
干预语义。每条件只计算自己确实运行且观测改变的状态。

| 新产物 | 作用 |
| --- | --- |
| `paired_audit.json` | 各条件/turn/数据集效应、同状态 E1 对照、整题聚类区间、输入 SHA256 |
| `paired_events.json` | 小体积、规范化的每状态×条件 EM/F1 配对事件，供 B 转移核对 |
| `sufficiency_review.json` | 全部退化条件事件的独立证据审查包，语义结论均 pending |

区间按数据集分层、按问题重采样，保留该题不同目标轮的相关性；估计对象为
该条件资格状态加权平均，而非所有题的总体效应。每个数据集至少两个问题才
给区间，否则 null。E3/E2−E1 仅在同一事件有 E1 分支时计算，并报告实际分母。
2000 次用于探索审计；独立确认前需另冻统计计划。此区间不是研究 GO。

这一步重验 raw 配对和覆盖，但不重新核验模型/索引二进制身份、service audit
或词法资格的语义正确性；不要把程序退出 0 当完整科研验收。

## 4. 审查全部退化事件，不先只挑 74 个

先保留新生成审查包的原版，再在本地副本填写 review。每个事件包含原问题、
金标答案/事实与**致错分支实际 visible 观测**，不混用 clean、其他干预或隐藏
candidate20，也不先展示模型后续推理/预测。选取本身基于退化，应明确是
outcome-selected 开发审计，不用于估计全量错误机制比例。

每条 review 填：

| 字段 | 要求 |
| --- | --- |
| `decision` | sufficient / insufficient / uncertain，依据语义而非答案实体匹配 |
| `reviewer` | 独立于被测 Search-R1 的审查者；建议两名独立审查、分歧再裁决 |
| `earliest_sufficient_turn` | 足以支持**原问题全部约束**的最早轮次；不足/未知时 null |
| `supporting_quotes` | 数组，元素含 search_turn、document_id、原句 quote 和支撑的关系/约束说明 |
| `reason` | 证据链、时间/身份/比较条件、别名或缺口解释 |

不要将 29 个非实体答案和 13 个未匹配状态删掉。数字题验证单位、计算链和
比较对象；yes/no 题验证命题；别名题记录等价依据，不凭 clean 答对就断言别名。
14 个空答案即使充分也独立归为终止/预算/格式候选，不能全归“忽略证据”。
当前工具只准备审查包，不自动认证填写正确，也不把填写结果自动变为 L3。
人工审查结果须对照原始文件哈希和事件 ID 复核。

第二遍审查才查看原始后续 steps/prefixes：在最早充分证据之后，第一次错误候选
实体/时间/关系的文本采用出现在何处，下一 query 是否仍相关，是否重复搜索，
有没有生成有效 answer 的机会，最终失败 status 是什么。不要把文本里的首次
提及说成模型内部“锁定答案”的确切时刻。记录可观察行为，保留替代解释。

同时从 stable 449 状态中预先按 dataset/turn/上下文长度/剩余预算分层选择
对照（可先选与退化数相当的 134 状态；按固定 ID 顺序/seed，不按机制特征或
新动作结果挑选），执行同样证据核验。相同问题的不同轮不能跨训练/验证拆分。
这一步为后续触发器提供真实负例，不能用 oracle clean/EM 标签作部署触发信号。

## 5. 核对 B3 全量收益/损害

厂内 B 文件 schema 未提供，尤其大指标可能内嵌全部轨迹，不能在新程序里
假装知道键名。先从原始 B 导出一个**纯格式** compact JSONL，每个实际 B
分支一行，字段如下（示例数值只是格式，不是真实结果）：

```json
{"sample_id":"dataset:original_id","dataset":"hotpotqa","target_turn":1,"condition":"E2_distractor","baseline":"B3","clean_em":1,"perturbed_em":0,"recovered_em":1,"source_locator":"原 B 文件路径、记录 ID 和分支键"}
```

不能只导出恢复成功者；导出所有已运行状态和基线。conditions 必须与实际 B0
匹配，不用“任一干预最坏分数”替代该起点；每个 baseline 的资格与跳过另列。
JSON 字段对应不确定时反馈真实 B 键名/一条去正文的 schema，我们再适配，
不要求厂内重新跑 GPU。该格式导出本轮没有自动猜测实现。

```bash
python -m work2.audit_recovery \
  --paired-events outputs/semantic_audit/paired_events.json \
  --transitions outputs/semantic_audit/b_transitions.jsonl \
  --assignment-policy unknown \
  --output outputs/semantic_audit/recovery_audit.json
```

若已核验 B 条件在观察结果前确定，填 pre_outcome；若依 gold/EM 选择条件，填
outcome_selected；未核实保持 unknown。该字段是来源声明，程序不自动证明它。
工具重算 gain/harm/net、派生 degraded/stable_clean_correct/clean_wrong 分层，
核对 B0/E0 分数与原 A3，拒绝重复或未知事件。recovered_em 由导出提供，本工具
不读取未知 B schema 重评答案，原始答案与预算/cost/gold 必须另行审计。

coverage 基于所有输入 B 事件的并集；B5 是 E3 限定 oracle，其覆盖不同不能
算成失败或冒充全量覆盖。缺少某基线的事件不会被计为 0。各基线若用不同条件
集合，只能给各自子集结果，正式方法比较须先冻结共同资格集合和缺失政策。

## 6. 代码/记录重点复核项

1. 合并按 ID×turn×condition×baseline 核查覆盖和重叠；全部 shard 的配置、
   模型/语料/索引身份、源 split/hash/commit 一致，不只继承第一个 manifest。
2. B1 总调用上限是否为 6，剩余是否为 6−target_turn；源 prompt 仍写 4 次时
   需记录方法限制，不能静默改变 A1 初始 prompt 破坏配对。
3. B3 使用的 subgoal 来自哪里，是否 gold 泄漏；额外 instruction/input tokens
   和实际生成/检索成本全部记录。
4. B2 是否真正改变观测或后续过滤；是否只看 query 字符串去重；若过滤支持
   事实，需要独立保留资格和副作用记录。
5. B0 必须是所选条件的 A3 实际分支；记录多条件退化状态的选择规则，区分
   175 条条件事件与 134 个状态，不能对后者宣称是随机干预总体。
6. S1/S2/L3 代理的函数/版本/分母一致；visible 不与 candidate/clean 混用。
   厂内基于 a17f1f3 且报告仅修改旧 replay.py，需检查 A3/B 是否仍继承了旧
   Flat/NOT HNSW 标签误判；只能在新派生产物勘误，不修改历史 manifest。
7. 空答案、infra、不可达、条件不合格、未运行分别处理；“completed”不是覆盖证明。
8. 后续正常检索指固定 corpus/index/backend，而非强制所有分支沿用 clean 的
   query/答案序列。以实际 Query/observations/audit 核查，不凭文字“冻结”推断。

## 7. 下一次 GPU 前先冻结机制实验

完成审查后选择少量**开发**状态及 matched stable 控制，按语义标签选择，不按
新动作是否修复选择。至少区分下面三个可证伪假设：

| 候选机制 | 区分性观察 | 下一轮受控动作与必要对照 |
| --- | --- | --- |
| 问题约束/证据选择错误 | 证据已充分，仍采用不满足时间/角色/关系的实体 | 原问题约束核对 vs 普通回顾；同样完整观测、匹配额外生成预算；不输入 gold |
| 前序错误假设干扰 | 最早充分证据之后仍重复此前错误候选 | 保留所有实际 evidence 的历史摘要/重建 vs 同成本摘要控制；此为 history intervention，另标 protocol，不能冒称原 raw prefix 完全相同 |
| 终止/预算/格式错误 | 充分证据后没有规范 answer，或仍申请搜索 | 受限的答案收束/提取诊断 vs 同成本一般续写；单列 14 个空答案，禁止用 gold 作答案模板 |

先把 prompt、起点、预算、真实输入和对照写入固定协议，再实现/跑 GPU；本轮
只列这些假设，不宣称已经解释内部机制。其中约束核对和 context refinement
有明确邻近工作，作为比较基线，不能直接当新方法。

机制证据成立后，独立确认题目须排除本轮全部 1000 题、旧 pilot 和其他已观察
案例，并按问题哈希核查近重复。开发标签可帮助设计触发器，确认集只跑冻结
策略。报告 Recovery、StableHarm、全量 NetEM、成本与触发覆盖；触发决策只能
读取该状态已知问题/历史/观测/政策特征，不使用未来结果或 clean/gold 正误。

## 8. 下一次反馈

请反馈新 commit、对齐备份分支名、paired_audit 的状态/资格/同子集 E1 区间、
退化事件与独立状态数、语义 review 的所有分类/分歧（含非实体/未匹配/空答案），
以及 B 的完整 gain/harm/net/coverage/condition-assignment。先选 2–4 个具体
证据充分案例与 stable 反例核对文档原句和后续步骤。若 schema 不兼容，只反馈
schema-only 键名输出与报错即可；不要通过重跑或删行让旧产物“符合预期”。

所有输出保留本机，不提交公开仓库；不要求厂内 git push，不提供环境安装步骤。

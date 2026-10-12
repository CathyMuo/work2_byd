# Work 2 尝试性试验与结论总报告

整理日期：2026-10-12。记录范围：从 A1 clean baseline 到 E v2 CPU 收尾。
截至本报告，当前“约束核对优于一般回顾”方法分支已归档 **NO-GO**，没有
未完成的 E 必跑任务；未进入独立确认 F，未启动本研究的 RL。

本报告是试验档案，不是继续执行实验的指令。现行研究状态见
[STATE.json](../research/STATE.json)，完整决策历史见
[DECISION_LOG.md](../research/DECISION_LOG.md)。

## 1. 研究问题、证据来源与单位

原研究问题：多轮搜索 Agent 在一次检索观测被改变后，是否发生后续搜索目标
偏移或终局错误？能否在有限预算内从实际扰动状态恢复，同时避免伤害原本正确
的状态？工作标题为 Search Feedback Drift and Recovery in Multi-Turn Search Agents。

结果主要来自用户在会话中提供的厂内运行报告，包括 2026-10-10 的扩大实验
附件和随后各轮反馈。家服实现公开代码、完成 CPU 合成测试及记录分析；
**厂内模型运行、原始轨迹审计及人工解释的结果，均按厂内报告登记，家服未
获取这些原始产物独立重算。** 不将厂内未公开源码或额外测试声称为已合入。

| 单位/名称 | 本报告含义 |
| --- | --- |
| 问题 | 数据集中的独立 sample_id；同题多个轮次不当独立题 |
| 状态 | 同一问题的一个目标检索轮次，即 `(sample_id, target_turn)` |
| 条件事件 | 一个状态在 E1/E2/E3/E4 中的一条干预分支 |
| E0–E4 | 观测干预条件，见 §5；不是机制审计工作项 E |
| C/D/E/F 工作项 | 证据与机制审计、开发探针、独立确认；不等于研究阶段 C 的 RL |
| 退化 | 原 clean 正确、干预分支错误或无有效答案，EM 由 1 变 0 |
| 改善 | clean 的 EM=0，干预分支 EM=1 |
| gain/harm | 恢复动作相对同状态原样分支的 EM 增益/损失；harm 包括答错和空答案 |
| Δsearch | 与原样或另一动作的调用差；不是该臂起点后的绝对调用数 |

原始 outputs、模型、数据、索引和本地配置保留在厂内，不提交本公开仓库。

## 2. 试验过程与最终判断概览

| 步骤 | 实际范围 | 主要结果 | 当前有效结论 |
| --- | --- | --- | --- |
| A1 smoke | 10 题 | EM 4/10，基础设施错误 0 | 推理/检索流程可运行 |
| A1 clean pilot | HotpotQA 100 + 2Wiki 100 | EM .335，F1 .4341，多轮率 .975，586 前缀 | 同前缀实验具备工程前提 |
| A2 replay sanity | 10 题 × 目标轮 1/2，20 状态 | 原样控制/历史一致均 20/20，检索漂移 0/150 | 在该配置和案例上可精确续写 |
| 早期 A3/B pilot | 厂内另报 31 个退化状态 | 有恢复与 stable harm；E3 小子集信号很少 | 仅探索线索；完整运行元数据未交付家服 |
| 扩大 A1/A3/B | A1 1000 题；A3 2000 计划状态 | 1972 completed；B3 有修复也有损害 | 不构成语义机制或方法 GO |
| matched-E1 | E3 178 状态、151 题 | E3−同状态 E1 = +3 EM 计数，CI [−4,+10] | 额外内容损伤效应未建立 |
| C/D 代理与校准 | 175 事件、134 状态；独立审 8 事件 | 4 sufficient / 2 insufficient / 2 uncertain，167 pending | 词法标签撤回，部分语义校准可用于工程选例 |
| E v1 准备 | 4 个充分致错案例、12 稳定候选 | 4 组全 skip，0 充分配对；未跑 GPU | 仅固定候选池无可运行充分配对 |
| E v2 smoke/engineering | 最多 4+4 案例，共 8 案例 | 工程通过；primary general 3/4、constraint 2/4 | 约束方法未展示额外优势 |
| E v2 CPU 收尾 | 8 案例机械审计、5 事件案例人工解释 | 全通过；唯一 harm 为超预算请求被拒 | 当前约束方法归档 NO-GO |

公开的 10 题 A3 工程入口曾实现并完成合成测试；厂内随后另行开展 pilot/scale
并形成分叉。不能将“已实现/合成测试通过”写成某一公开版本实际完成过 H800
试验，也不把早期厂内 pilot 强行对应为公开 10 题入口的运行结果。

## 3. 共用实际 cohort 与预算

| 项目 | 实际使用/报告设置 |
| --- | --- |
| Agent | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo` |
| 模型 revision | `713cbe32f48a45a855da9cd09a0d980c2e3166e6` |
| Retriever | `intfloat/e5-base-v2` + Wiki-18，厂内常驻服务 |
| 索引 | `e5_Flat.index`，FAISS **Flat**；不是最初计划的 HNSW |
| 可见/候选文档 | Top-3 / Top-20 |
| backend | vLLM 0.6.3，torch 2.4.0+cu121，transformers 4.47.1，BF16 |
| 解码 | temperature 0，seed 42 |
| 模型并行/显存比例 | TP=1，gpu-memory-utilization=0.75 |
| 搜索预算 | 总成功搜索最多 4 次；保留已用次数，第 4 次返回后仍可生成答案 |
| 上下文/生成 | 8192 context tokens；512 tokens **每生成段**，不是恢复分支总上限 |
| Prompt | `work2_search_r1_bounded_v1` |
| 数据 | HotpotQA dev-distractor、2Wiki dev；最初各 100，扩大各 500 |
| 硬件 | 厂内 4×H800；A1/A2/E 单进程按源 TP=1；厂内 scale 报告采用数据并行 |

资源更换需另建 cohort，不能把这些结果写为 3B/HNSW。厂内数据路径与环境
配置不进入本报告。曾修复本地 checkpoint 的损坏 index JSON、转换 HotpotQA
数据结构；这不证明上游 checkpoint 发布文件损坏，也不构成模型权重修改。

## 4. A1：clean baseline 与扩大运行

### 4.1 10 题 smoke 与 200 题 clean pilot

A1 源代码：`aed1adc4194c02010c4c7be46872fed46fe49d40`。
样本在模型结果之前冻结，HotpotQA 和 2Wiki 各 100。smoke 完成 10/10，
EM 4/10，基础设施错误 0。

| 指标 | 总体 200 | HotpotQA 100 | 2Wiki 100 |
| --- | --- | --- | --- |
| EM | .335 | .370 | .300 |
| F1（四位舍入） | .4341 | .4994 | .3687 |
| 有效答案率 | .945 | .980 | .910 |
| 至少一次搜索 | 1.000 | 1.000 | 1.000 |
| 至少两次搜索 | .975 | .960 | .990 |
| 平均搜索次数 | 2.93 | 2.77 | 3.09 |
| context truncation | 0 | 0 | 0 |

总体 F1 报告精确值为 `0.4340773393273393`。
状态：answered 189、search_budget_exhausted 10、generation_limit 1；
基础设施错误 0。全部 586 个搜索前缀报告保存了 raw token ID、query、
候选 20、可见 3、observation、上游文档 ID 与前缀 SHA。

检索文本代理：首轮可见支持事实命中 30.0%，候选 20 命中 37.0%，任一轮
可见命中 42.5%；命中组 EM .412、未命中组 .278。这些是保守文本代理与
关联，不是语义证据充分性或检索因果效用的证明。

### 4.2 1000 题扩大 clean baseline

厂内分叉 `f4a4bb5` 报告：HotpotQA 500 + 2Wiki 500。

| 指标 | 200 题 pilot | 1000 题 scale |
| --- | --- | --- |
| EM | .3350 | .2970 |
| F1 | .4341 | .3808 |
| 多轮率 | .975 | .973 |
| 至少一次搜索 | 1.000 | .999 |
| 平均搜索次数 | 2.93 | 2.936 |
| 基础设施错误 | 0 | 0 |
| HotpotQA EM | .370 | .336 |
| 2Wiki EM | .300 | .258 |

报告称原 200 题重新运行汇总 EM/F1 一致。根据 297/1000、67/200 可推得新增
800 题 EM = `(297−67)/800=.2875`，说明总体下降与样本集合变化一致。
汇总一致不能单独证明逐题 token/资源完全无漂移，也不独立证明“固有难度”。

**结论：** 工具调用与多轮轨迹可用，A1 工程前提接受；clean 分数不回答干预
导致何种语义偏移，也不是 RL GO。

## 5. A2/A3：同前缀控制与单轮观测干预

### 5.1 A2 精确重放 sanity

运行提交 `a17f1f39c0aac5b83fbf6c7f0e17d6711084802f`。
取 A1 冻结顺序每个数据集前 5 题，目标轮 1/2，共 20 状态。
同 raw prefix 分叉：clean_a、clean_b、Top-3 轮换 `[1,2,0]` 的 permuted_a。
原样失配即停止相应干预。

| 工程指标 | 报告值 |
| --- | --- |
| 实际/计划状态 | 20/20，unreachable 0 |
| E0/E0 完全一致 | 20/20 |
| 与 A1 原历史一致 | 20/20 |
| 前缀/历史/文档多集不变量 | 20/20 |
| 确实改变观测 | 20/20 |
| Live 检索审计 | 150 次，133 次有参考；漂移 0 |
| 基础设施错误 | 0 |

描述性 clean EM .40、permuted .25；turn1 .40→.10，turn2 .40→.40。
这是 10 题相关状态的工程线索，不是显著排序损害或语义搜索偏移结论。
温度 0 的精确续写在这些案例得到报告支持，不是任意后端/状态的通用保证。

### 5.2 A3 干预定义与早期 pilot

| 条件 | 单轮观测变化 | 资格判断的限制 |
| --- | --- | --- |
| E0 clean | 原 Top-3 | 对照 |
| E1 permutation | 同三篇重排，文档多集保持 | 排序本身允许改变策略 |
| E2 distractor | 替换一篇为真实相关干扰文档 | “非关键/事实保持”须独立核验 |
| E3 bridge removal | 替换被判为桥接/事实唯一载体的文档 | 词法唯一载体不自动等于语义桥接 |
| E4 redundancy | 替换一篇为冗余事实文档 | 关键事实保持与冗余角色须核验 |

只改一次目标观测，后续为正常 live 检索，继承原始预算与模型/索引。
“检索资产冻结”不等于冻结所有后续返回或给策略使用缓存答案。
厂内早期资格主要使用标题/支持句词法代理，不能与公开独立审查资格等同。

早期厂内 pilot 另报 E3 33 个实际条件事件中退化 2；B 退化层 31 个状态，
相应摘要见 §6。其完整覆盖、全部条件指标与原始记录未交付家服。
不从词法资格率外推数据集普遍桥接频率或训练可用性。

### 5.3 扩大 A3（2000 计划状态）

A1 1000 题 × 目标轮 1/2：1972 completed、28 unreachable；报告控制失配 0。
各条件有不同资格集合，下面的 clean EM 因而不同。

| 条件 | 实际条件事件 | clean EM | perturbed EM | 退化 | 改善 | 净 EM 计数 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 1972 | .2956 | .3063 | 75 | 96 | +21 |
| E2 | 1960 | .2923 | .3082 | 81 | 112 | +31 |
| E3 | 178 | .3820 | .3258 | 14 | 4 | −10 |
| E4 | 88 | .3636 | .3864 | 5 | 7 | +2 |

净值等于改善−退化。E1/E2 汇总偏向改善，E3 子集呈净退化。
跨资格集合比较这些净计数不能建立 E3 超出排序效应，也不能建立 E1/E2 等价。
178/1972 是当前检索/代理定义的资格率，不是数据集真实桥接结构频率。

### 5.4 matched-E1 修正（报告 `839431d`，后续 `0ea168a` 补充单位）

在同一 178 个 E3 状态、151 个问题上比较 E3 与 E1：

| 项目 | 报告结果 |
| --- | --- |
| E3 相对 clean 的 Σdelta_em | −10 |
| 同状态 E1 相对 clean 的 Σdelta_em | −13 |
| E3−E1 | **+3 EM 计数** |
| 按问题聚类 bootstrap 95% 计数区间 | **[−4,+10]** |
| 重抽 | 2000 次，seed 42 |

+3 是计数，不是三个百分点。点均值 `3/178≈+1.685` 个百分点；聚类重抽可
改变状态总数，不能直接把计数区间除 178 当均值区间。
同状态干预前 prefix/预算是配对基础，不按干预后的长度/调用再筛选。

**结论修正：** 早期“E3 特异效应已经成立”撤回。额外内容损伤效应未建立，
但区间跨零也不证明等价或不存在个体失败。Query 字符串不同不能证明语义 off-target。

## 6. B：恢复基线、收益与副作用

厂内扩大实验报告 B 选择 1972 状态：degraded 134、stable 449、clean_wrong 1389。
B0 导入所选 A3 实际扰动轨迹；B1 预算 4→6；B2 去重/过滤；B3 缺失子目标
回顾提示；B5 是 E3 限定的 oracle 诊断。厂内实现未完整交付家服，预算/条件
选择/gold 隔离/覆盖仍须依原始记录核查。

| 基线 | scale 报告修复 | 比例 | 状态 |
| --- | --- | --- | --- |
| B1 | 10/134 | 7.5% | 条件结果，不能推断全部状态收益 |
| B2 | 0/134 | 0% | 需确认真实执行，而非无效动作/fallback |
| B3 | 44/134 | 32.8% | 同时损害 stable 53/449（11.8%） |
| B5 | 3/14 | 21.4% | 仅 E3 oracle；不是理论恢复上界 |
| 任一基线并集，含 oracle | 52/134 | 38.8% | 后续与 nonoracle=46 冲突，尚未对账 |

早期 pilot 报告退化层 n=31：任一基线恢复 45.2%，B3 41.9%，B1 6.5%，
B2 0%；stable harm 15.2%，该 stable 分母未在摘要中给出。
这些是来源报告的摘要，未提供独立确认，不能据比例近似认定总体稳定。

若 B0 定义相同，B3 在 134+449=583 个报告状态上净计数 `44−53=−9`；
缺少 1389 clean_wrong 全部转移，不能据此计算全 cohort NetEM。
未被这些动作修复的 82/134 不等于不可恢复，也不证明更复杂动作一定有效。

后续报告 nonoracle 修复 46 状态、旧词法 72 候选与其交集 25；同一 post-evidence
起点当时未获证明。这些数不能升级成语义恢复比例。若全集/单位/资格一致，
46 nonoracle 与 3 oracle 的并集最多 49，不可能是 52；**46/52/3 仍 unresolved**。
本轮四个 E 见证不用于补齐旧数字。

## 7. L3 代理、C/D 校准与口径撤回

### 7.1 早期“见而不用”自动分类

scale 初始 134 状态的实体代理类别为 60/29/18/14/13：见答案实体仍答错、
非实体未知、检索动作候选、见实体但无答案、双侧未匹配。
由此得到 74/134=55.2%、74/92=80.4%、74/105=70.5% 等不同分母。

这些是候选比例，不是语义 L3 发生率；答案实体出现不证明关系/时间/比较条件
满足，空答案不能统一叫作“主动不用证据”。早期“134 都语料可达”“133 可见”
同样依赖有争议的代理，不作为排除证据不可达的强结论保留。

### 7.2 C/D 全事件筛选及撤回

退化事件 175 = E1 75 + E2 81 + E3 14 + E4 5，分布在 134 状态。
厂内 `113ca69` 曾将正文重叠 ≥60% 的筛选当作语义充分性，报告：

| 已撤回的代理标签 | 事件数 | 状态数 |
| --- | --- | --- |
| sufficient | 110 | 86 |
| insufficient | 21 | 16 |
| uncertain | 44 | 32 |

旧“72 真 L3”及其条件成员 E1 38/E2 39/E3 9/E4 5 均为未充分核验的候选。
旧机制表 72+29+15+12+9+4=141，不能作为 134 状态互斥分布；去掉 uncertain
29 后为 112，不能用“非 uncertain 子集”解释 141。
厂内后来采用公开实现：**词法重叠仅作诊断，语义决定必须独立审查**。

### 7.3 正文 token recall 诊断与部分独立校准

`0ea168a` 报告已合并公开 `6be41d4`：固定 `body_token_multiset_recall_v1`，
剥离 Search-R1 标题行后，144/175 事件至少一条支持句正文 recall ≥.6。
这不是旧算法的逐位复现，也不自动赋予 sufficient 标签。

分层 high/mid/low/nonentity，12 候选经同题去重落地 8 个独立事件审查：

| 独立审查状态 | 事件数 |
| --- | --- |
| sufficient | 4 |
| insufficient | 2 |
| uncertain | 2 |
| pending | 167 |

4 个 sufficient 的正文引用及轮次通过机械校验；这只证明引用存在，不替代
完整语义推导。状态聚合报告 sufficient_any_event 4、pending_without_sufficient
128、all_insufficient 2；130/134 状态仍有 pending 条件事件。不是完整 C/D 收据。

四个 sufficient 均为 HotpotQA：

| 案例短 ID | 原干预轮/条件 | 声明最早充分轮 |
| --- | --- | --- |
| `5ae1dadf` | t1/E2 | t1 |
| `5ab7d12f` | t1/E1 | t1 |
| `5a83ed9b` | t2/E2 | t1 |
| `5a8888e3` | t1/E1 | t1 |

不足与不确定案例保留缺失关系、实体消解、年份比较等原因；不能只用金标
字符串出现/缺失作充分或不足证明。该小集合不代表 2Wiki/nonentity 总体。

### 7.4 空答案的正确互斥口径

| raw status | entity | nonentity | 总事件 |
| --- | --- | --- | --- |
| answered | 110 | 33 | 143 |
| search_budget_exhausted | 16 | 5 | 21 |
| generation_limit | 7 | 1 | 8 |
| malformed_action | 3 | 0 | 3 |
| 合计 | 136 | 39 | 175 |

空答案 **32 = 21 budget + 11 termination**。旧 16+10+6 把非实体答案类型
与终止原因混成互斥类，已停止沿用。raw 状态不证明心理原因或主动拒答。

## 8. E v1：充分致错/充分稳定配对准备

公开实现 `c0a1e8b`，厂内反馈 `faf8494`。从四个 sufficient 案例中生成四组，
每组最多三个 stable 候选，要求 clean 与扰动均正确、同 dataset/condition/
干预轮/启动轮/剩余预算，起点输入长度比 .75–1.25，固定顺序全局不同题。

起点固定为实际扰动历史的 `max(原干预轮, 最早充分轮)` 观测之后。
`5a83ed9b` 即使 t1 已充分，也必须从 t2 之后启动，避免跳过尚未发生的干预。

四个致错案例重新确认 prefix-only sufficient；十二个 stable 候选都被独立判
prefix-only insufficient。四组全 skip；validate 给出 0 完整配对，runner 正确
拒绝。**无 E v1 GPU 运行，未放宽充分配对要求。**

结论限定为固定近邻候选池没有可运行配对，不是全 cohort 的结构性缺失。
稳定终局正确可能来自后续检索、推理或知识；金标缺失不证明参数知识来源。

## 9. E v2：分层动作开发探针

### 9.1 明确修订的问题与方法

公开实现 `755340e`，厂内运行报告 `82fa3d5`。
v1 全 skip 保留，v2 为独立版本，不将 v2 工程通过记成 v1 gate 通过。
固定原候选池，不扩池找成功：

- primary `failed_sufficient`：4 个充分但原扰动答错案例，测修复。
- secondary `stable_insufficient`：每组首个已判不足 stable，4 个案例，测 harm。

每案例从同一实际 post-observation 状态运行原样 A/B、general_review、
constraint_review。全部选中案例原样控制和历史后缀通过后，才执行回顾臂。
保留原历史/raw generated token 与已用预算；固定提示不输入 gold/审查标签。
同一公开续写引擎执行 v1/v2，没有第二套恢复逻辑。

这是按已观察结果选的开发探针；两层不能合并为代表性总体 NetEM。

### 9.2 smoke 与工程门槛

smoke 为 pair_1 的两案例、8 条续写；engineering 为全部 8 案例、32 条续写。
两次均报告 `all_planned_cases_completed=true` 和
`all_original_controls_and_history_exact=true`。
smoke 的 pair_1 重复出现在 engineering，不作为第五个独立 primary 问题。

smoke primary：general 修复，constraint 未修复；secondary 两臂均正确。
单进程 TP=1，使用一张 GPU；没有因此改变源配置，也不把四卡使用率当研究 gate。

### 9.3 全四组逐层指标

下表 Δ 均相对同状态原样续写；token 数为厂内报告均值/舍入值。

| 层（各 n=4） | 动作 | 原样 EM | 动作 EM | gain | harm | Δgen | Δinput | Δsearch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| failed_sufficient | general | 0 | .75 | 3 | 0 | +30 | +984 | +.50 |
| failed_sufficient | constraint | 0 | .50 | 2 | 0 | +6 | +603 | +.25 |
| stable_insufficient | general | 1 | 1.00 | 0 | 0 | −5 | +88 | 0 |
| stable_insufficient | constraint | 1 | .75 | 0 | 1 | +6 | +722 | +.25 |

同状态 constraint−general 的直接对照：

| 层 | ΔEM | ΔF1 | Δgenerated | Δinput | Δsearch |
| --- | --- | --- | --- | --- | --- |
| failed_sufficient | −.25 | −.25 | −23.5 | −381 | −.25 |
| stable_insufficient | −.25 | −.25 | +12 | +633.8 | +.25 |

两种表的 token 舍入不同，不强求整数均值之差逐位等于精确差值。
primary 是准确率—成本取舍；secondary constraint 在观察结果上更差且更贵。
不统一称为“constraint 没有任何成本优势”，也不由 general 0/4 harm 宣布安全。

### 9.4 逐案例结果与修复并集

| 组 | primary general | primary constraint | secondary general | secondary constraint |
| --- | --- | --- | --- | --- |
| pair_1 | EM=1 | EM=0 | EM=1 | EM=1 |
| pair_2 | EM=1 | EM=0 | EM=1 | EM=1 |
| pair_3 | EM=0 | EM=1 | EM=1 | EM=0，预算终止 |
| pair_4 | EM=1 | EM=1 | EM=1 | EM=1 |

primary：general_only 2、constraint_only 1、both 1、neither 0。
两种已尝试动作的修复并集 **4/4**，证明这四个起点有局部实际恢复见证。
按 gold/终局 EM 选择较好臂只是事后 oracle，不是可部署 selector 或理论上界。
单个 t2 的 constraint 独赢不提供 t2 触发规则；其 secondary 恰是唯一 harm。

## 10. E v2 CPU 收尾与人工解释

公开收尾脚本 `3664469`，厂内报告 `8ceeda1`。
只读审计全部 8 案例通过，核对 source/父计划/冻结计划哈希、身份/实际历史、
所有原样控制与历史后缀、raw 分数、token/input/search 成本、全预算、逐字段
指标复算以及检索漂移；`mechanical_checks_passed=true`。
没有新 GPU、没有加载模型/访问 Retriever、没有重跑 policy。

人工解释完成 4 个修复 primary + 1 个 harm。其余 3 个无事件解释 pending，
不是未完成的 v2 资格审查，不要求补审或重复审计。

### 10.1 已解释的修复路径

| primary | 解释的修复臂 | 起点后成功检索绝对数 | 可以保留的行为描述 |
| --- | --- | --- | --- |
| pair_1 | general | 3 | 后续检索与初始证据共同出现 |
| pair_2 | general | 2 | 修复伴随检索；生成文本有 `whichfuscated` 异常 |
| pair_3 | constraint，独赢 | 1 | 原问题作 query，追加检索后重述初始可见的植物属事实 |
| pair_4 | general | 2 | 后续检索伴随答案完整度提升 |

query 等于原问题不证明它等于某个先前实际 query，也不证明返回内容相同。
pair_3 最后复述已有事实，不能排除追加检索、重复暴露、位置/表述的贡献；
不由生成解释推断内部注意力或采纳机制。三个 general 修复都伴随检索，但也
未证明这些检索因果必要。表中是四个案例的已报告解释，不补造 pair_4 constraint
的独立修复路径结论。

### 10.2 唯一 harm：pair_3 secondary

| 项目 | constraint | general / 原样对照 |
| --- | --- | --- |
| 起点已用/剩余调用 | 2 / 2 | 同一实际起点 |
| 起点后成功新检索 | 2 | 1 |
| 最后动作 | 请求下一次搜索，被系统拒绝 | 作答 |
| final_answer | 空 | `Prince George's County` |
| status / EM | search_budget_exhausted / 0 | answered / 1 |
| 起点后 generated tokens | 151 | general 96 |
| 起点后 input tokens | 5404 | general 3027 |

被拒 query：`largest municipality in which Maryland Judget Club operates a race track`。
它发生在四次成功搜索之后；系统仍给了最后一段生成机会，模型发出第 5 次请求
被拒。**分类为 `budget_search_refusal`，不是 answered_wrong。**
被拒 query 的拼写异常没有产生检索结果，不作为 retriever miss 的原因证明。
不推断“困惑”“参数知识受干扰”等心理或内部机制。

## 11. 已撤回结论、工程勘误与未决项

| 早期说法/问题 | 当前档案口径 |
| --- | --- |
| E3 已证明超出 E1 的特异损害 | matched-E1 后撤回；+3 计数 CI 跨零，额外负效应未建立 |
| Query 变化约 60% 就是语义搜索漂移 | 字符串差异，不证明 off-target |
| 80.4% 或 72 个“真 L3” | 代理条件分母/词法候选；自动语义标签撤回 |
| 134 例都排除证据不可达 | 代理依据不足，不能保留强结论 |
| 82 例没被基线修复就是不可恢复 | 仅现有动作未修复 |
| 141 机制数是 134 状态互斥分布 | 撤销该表；事件/状态/多标签不能混合 |
| 32 空答案 = 16+10+6 | 正确互斥终止口径 21 budget + 11 termination |
| v1 全 skip 证明结构性缺乏稳定控制 | 仅 12 个固定候选；不外推全 cohort |
| v1/v2 是两份独立否定证据 | 共用源数据和四个 primary，不独立 |
| 4/4 并集就是可部署恢复率 | 本地恢复见证，事后 oracle 不提供 selector |
| constraint harm 是“答案改错” | 空答案/最终搜索被拒；EM harm 与机制类别分开 |
| 46 nonoracle、52 union、3 oracle 都一致 | 同口径下不可能；旧 ID 对账仍未完成 |
| “Flat ...; NOT HNSW” 被解析为 hnsw | 标签 bug；实际 Flat。新声明勘误，不改历史 manifest |
| A1 source F1 一次校验失败 | 后续逐位稳定，原因未定位；错误信息增加两侧值/类型 |

厂内分叉还报告过合并丢 provenance、样本覆盖不足仍 completed、硬编码选择量、
看板线程阻塞/巨大载荷、完成目录误判、tmp 导致脏树等修复。家服未获取其
全部源码，不声称审计或合入这些具体补丁。公开审计采用实际 ID/覆盖、raw
分数、原样控制和哈希检查，而不以日志 completed 或退出码 0 替代验证。

## 12. 最终结论与资源决策

1. **工程成立：** 实际 raw prefix 重放、单轮观测替换、预算保留与控制门槛具有
   可用工程资产；厂内报告了精确控制及完整收尾通过。
2. **原科学证据链未闭合：** 内容特异损害、后续语义 off-target、全 cohort
   恢复净收益和可部署选择规则尚未建立；部分词法结论已撤回。
3. **专门约束方法不继续：** 在已选开发案例上，约束动作未展示对一般回顾的
   额外优势，并出现预算终止 harm。停止其提示调参/扩大/训练投入，归档 NO-GO。
4. **NO-GO 是投入决策：** n=4/层不足以证明总体显著劣势、反思无用或所有状态
   不可恢复；保留案例差异、4/4 见证和强基线。
5. **没有未完成的 E 必跑任务：** 不重跑 smoke/审计，不补无事件解释，不为 8
   案例新增分片。独立确认 F 与本研究 RL 未执行，其他 GRPO 作业不计入本研究。
6. **未来候选未立项：** 有限预算下回答/检索/保持原策略的动作选择，只保留为
   文献与假设评审问题。当前四题不训练、不挑规则、不用于新方法确认；不能改名
   自动续跑旧路线，也不引入其他停止项目的方法或已选择产物。

## 13. 版本、产物与参考档案

### 13.1 关键报告/代码版本

| 阶段 | 公开代码/基础提交 | 厂内报告提交 |
| --- | --- | --- |
| A1 | `aed1adc` | 同源 A1 报告 |
| A2 | `a17f1f3` | 同源 A2 报告 |
| 公开 reviewed A3 工程实现 | `c4d9521` | 后续 scale 分叉并非完整导入 |
| 厂内 scale A1/A3/B | `a17f1f3` 分叉 | `f4a4bb5` |
| matched-E1 / 旧 C/D | 后续公开独立审计能力 | `839431d` / `113ca69` |
| 正文校准 | `6be41d4` | `0ea168a` |
| E v1 | `c0a1e8b` | `faf8494` |
| E v2 | `755340e` | `82fa3d5` |
| CPU 收尾 | `3664469` | `8ceeda1` |
| 公开归档状态 | `8117e4b` | 按最终反馈登记 |

厂内无法 push；上述厂内 HEAD 是报告标识，不假设可由公开远端获取。
公开家服代码为后续依据，厂内本地旧提交与 outputs 保留。

### 13.2 厂内产物索引（仅记录相对路径）

| 内容 | 报告路径 |
| --- | --- |
| A1 200 baseline | `outputs/a1_clean_200/` |
| A1 冻结样本 | `outputs/samples/pilot.jsonl`、`selection_manifest.json` |
| A2 sanity | `outputs/a2_sanity/` |
| A1 scale | `outputs/a1_clean_scale500/` |
| A3 scale | `outputs/a3_scale500_merged/` |
| B scale | `outputs/b_scale500/` |
| 旧 L3/recoverability | `outputs/l3_case_audit_scale.json`、`recoverability_report_scale.json` |
| C/D | `outputs/semantic_audit/body_diagnostics.json`、`body_reviewed.json`、`calibration_receipt.json` |
| E v1 | `outputs/e_review/` |
| E v2 计划 | `outputs/e_probe_review/` |
| E v2 smoke / engineering | `outputs/e_probe_smoke/`、`outputs/e_probe_engineering/` |
| E v2 收尾 | `outputs/e_probe_closeout/receipt.json` |

这些路径不表示文件已在家服存在；历史 hash/manifest 与 raw 记录保持不变。

### 13.3 家服核验与来源入口

公开 CPU 合成测试最后记录为 **81 项通过**，覆盖原有 A1/A2/A3/v1/v2、
预算与 token 身份、控制停止、来源保留、指标/收尾损坏检测。
厂内曾报告 77 或 113 项本地测试，不与公开测试树数量混同。
合成测试不是 H800 结果或语义机制的验证。本次仅整理 Markdown，未重跑实验。

- [A1 原始报告分析](../research/A1_REPORTED_RESULTS.md)
- [A2 原始报告分析](../research/A2_REPORTED_RESULTS.md)
- [扩大 A3/B 结果审查](../research/FACTORY_SCALE_REVIEW.md)
- [C/D 历史审查协议](../docs/CD_REVIEW_AND_MECHANISM_PROTOCOL.md)
- [E v1 历史协议](../docs/E_EXECUTION_GUIDE.md)
- [E v2 历史协议](../docs/E_V2_STRATIFIED_PROBE.md)
- [E v2 深度分析与 NO-GO 决策](../research/E_V2_RESULTS_AND_DIRECTION.md)
- [最终归档与未决项安排](../docs/POST_ARCHIVE_NEXT_STEPS.md)

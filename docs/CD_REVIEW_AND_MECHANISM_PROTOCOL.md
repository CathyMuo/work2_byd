# C/D 报告验收与 E/F 下一步

更新：厂内 `8ceeda1` 已报告 E v2 收尾完成，当前方法分支归档 NO-GO，无必跑 E
任务。最新安排见 [归档后步骤](POST_ARCHIVE_NEXT_STEPS.md)。以下保留历史协议，
不作为重跑审计、新 GPU/F/RL 或自动立项预算方法的指令。

2026-10-12。这里 C/D/E/F 指审计工作项，**不是研究阶段 C（RL）或干预条件
E1–E4**。用户报告厂内 `839431d`、`113ca69` 的结果，但源码/逐事件标注未提供，
并要求代码以公开主线为准。本轮新 `semantic_review.py` 是独立主线实现，
不冒充原脚本，也不声称重现报告的 110/86/72。

## 1. 本轮报告中可以接受的进展

- 报告覆盖全部 175 个退化条件事件、134 个状态，未只审旧 74 个代理候选。
- 事件 110+21+44=175，状态 86+32+16=134，计数相容。
- 空答案开始按实际 budget/termination 状态分离，纠正了“看到证据却主动不答”
  的笼统解释。但 raw 终止状态只解释输出模式，不自动解释为何耗尽预算。
- matched-E1 报告称 E3 “+3，CI 跨零”；据此保持内容特异效应未建立。未提供
  +3 的单位、区间上下界或匹配规模，不能补造为 3 个百分点或宣称已证明无效应。

## 2. 三个必须修正的口径

**60% overlap 仍是词法筛选。** 剥离标题能减少假阳性，但只要漏掉 NOT、日期、
角色或比较关系，60% 甚至更高的词汇覆盖仍可能不支持原问题。两个手工正/反例
不能验证全部 175 事件的语义精度。主线保留该筛选为诊断，不自动标 sufficient。
独立审查应保留可见正文原句、原问题约束及答案推导；非实体答案不自动判 unknown。

**机制表相加是 141，而不是 134。** 若为多标签/跨条件重叠，明确给唯一状态
集合和标签交集；若声称互斥主分类，须修正。主线使用每事件一个 primary_label，
secondary_labels 另记。相同状态不同条件出现不同主标签，记 mixed_across_conditions，
不按“L3 优先”吞掉其他情况。“非实体答案”是 answer_type，不是空答案原因；
报告中 6 个非实体空答案仍需实际终止状态，不能与 16 budget/10 termination
当作同一维度的互斥原因。

**证据充分不等于动作已证明可恢复。** 72 是新筛选下的证据使用失败候选；
可辩护的“被测 Agent 可恢复却未恢复”应取：独立充分性确认、确有作答机会、
答错，以及**同起点/同剩余预算的非 oracle 恢复见证**的交集。此前 52 见证包含
受限 oracle，它与 72 的 ID 交集尚未报告。不能把可恢复分母直接改成 72。

条件成员 38+39+9+5=91 可以在 72 个状态上重叠；不是互斥贡献。E1/E2 数量
相近是线索，不证明关联弱。必须按资格暴露和退化事件分母比较，并处理同题/
同状态相关性。E3/E4 资格分母远小于 E1/E2，不能只比原始个数。

## 3. 对齐公开主线，保留厂内两次新提交

厂内若有 `839431d`、`113ca69` 等未 push 的提交，普通 ff-only pull 会失败。
按此前授权再次备份切换，保留 ignored outputs 和 local.json：

```bash
cd /root/develop/workspace/work2_byd
GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
  -c remote.origin.url=https://github.com/CathyMuo/work2_byd.git fetch origin main
git show origin/main:scripts/align_public.sh > /tmp/work2_align_public.sh
bash /tmp/work2_align_public.sh --use-fetched-ref
python -m unittest discover -s tests -q
```

脚本拒绝未提交文件，先保存旧 HEAD 到 factory-backup 分支，不执行 git clean。
旧 semantic_review.py 与原始结果仍可从本地备份分支/outputs 查看。

## 4. 主线正文诊断：不重跑模型

使用 `audit_pairs` 已生成的 native `sufficiency_review.json`；格式不符则先反馈
键名，不能删行或伪造新 packet。按实际目录填写：

```bash
python -m work2.semantic_review diagnose \
  --packet outputs/semantic_audit/sufficiency_review.json \
  --threshold 0.6 \
  --output outputs/semantic_audit/body_diagnostics.json
cp outputs/semantic_audit/body_diagnostics.json outputs/semantic_audit/body_reviewed.json
```

诊断使用公开版本 `body_token_multiset_recall_v1`：移除 Search-R1 contents 第一行
标题，按大小写归一的 Unicode 单词多重集计算支持句 token recall，逐事实保存
最佳 body 匹配与轮次。不强制标题相等，不推断语义充分。score 的完整定义已经
固定；原厂内 overlap 算法未知，不声称与之逐位一致。未知支持句保持未知。
它使用金标 supporting sentence，属于离线标签/诊断，**不可用作部署触发分数**。
这个离线诊断可查看完整轨迹，它**不是可部署触发器**，不得读取其未来证据来
决策更早状态。原始源文件、review.decision 都不会被自动修改。

## 5. 校准和审查收据

可以先做小规模独立校准，不必再次无差别手工重读全部 175 条。预先按高/低/
未知 proxy、数据集、答案类型和目标轮分层冻结样本 ID；包括关键否定/时间/
身份约束、非实体/别名案例及匹配 stable 对照，不按新动作是否修复选样本。
保留之前有效审查原句，在同一事件 ID 上补齐约束说明。仅校准子集时，不把
全量 72 当“真 L3”；若论文主张精确 72 个语义正例，应有这些正例的逐例证据。

只编辑 body_reviewed 中的 review：

- decision：sufficient / insufficient / uncertain；未审保持 pending。
- reviewer、independent_of_studied_agent=true、reason：可追溯独立判断。
- sufficient 另填 question_constraints_verified=true、answer_derivation、
  earliest_sufficient_turn，以及 supporting_quotes（search_turn/doc ID/正文原句）。
- 引用必须来自实际 visible **正文**，不得从标题、candidate20、clean 分支或
  未来于所声明 sufficient turn 的观测取证。

```bash
# 校准子集验收，明确保留未审数量，不能称完整 C/D：
python -m work2.semantic_review validate \
  --packet outputs/semantic_audit/body_diagnostics.json \
  --reviewed outputs/semantic_audit/body_reviewed.json --allow-pending \
  --output outputs/semantic_audit/calibration_receipt.json
```

去掉 allow-pending 才要求全部事件 final。程序校验上下文/引用/覆盖与审查声明，
不自动认证语义真伪。状态聚合明确采用 any-sufficient-event，仍记录 pending
事件；跨条件一个 sufficient 不意味着该状态所有干预分支都充分。

## 6. 机制标签与实际原始终止状态核对

另存逐事件机制 JSONL，必须覆盖同一批全部事件，各字段如下：

```json
{"dataset":"hotpotqa","sample_id":"真实ID","target_turn":1,"condition":"E1_permutation","primary_label":"evidence_use_candidate","secondary_labels":[],"answer_type":"entity","reviewer":"真实审查者","reason":"实际原句及后续作答行为说明"}
```

primary_label 支持 evidence_use_candidate、budget_exhaustion、termination_failure、
retrieval_action_candidate、uncertain、other；answer_type 单列 entity/yes_no/numeric/
other/unknown。额外标签只放 secondary_labels，不能把多标签 sum 当人数。

```bash
python -m work2.semantic_review validate \
  --packet outputs/semantic_audit/body_diagnostics.json \
  --reviewed outputs/semantic_audit/body_reviewed.json \
  --mechanisms outputs/semantic_audit/mechanism_labels.jsonl \
  --a3-rollouts outputs/a3_scale500_merged/paired_rollouts.jsonl \
  --output outputs/semantic_audit/cd_receipt.json
```

该模式要求全部充分性审查完成、原 A3 SHA256 未变、正文与问题仍对应原分支。
budget_exhausted 必须作为 budget 主终止类，generation/malformed/context_limit
作为 termination 类；上游可能的证据/搜索缺陷另标 secondary，不当作已证实因果。
evidence_use_candidate 要求独立 sufficient、充分轮次之后实际有生成、非空 wrong
answer。空答案不能纳入该类。不同条件不同主机制的状态单列 mixed，互斥状态
表加总应等于 distinct_states。raw schema 不一致明确停止，先适配，不覆盖旧文件。

## 7. E：现有案例的小规模机制对照

E 不要求新题。当前实现先运行 1 对、最多 4 对独立核验的失败/稳定开发案例；
按 dataset/condition/干预轮/启动轮/长度/剩余预算匹配，详见 E 实施文档。
同一问题的多个状态不得当多个独立题；这是工程范围，不是 GO 效应阈值。

**起点固定为原致错分支 max(干预轮, 最早充分轮) 观察之后的真实 token 状态**。若证据在
turn 3 才到达，不能把它拿去证明 turn 1 已具备充分信息。保存起点 raw token
SHA、实际观察、已花预算和上下文长度；只读取截至该时点的信息。

冻结以下对照再实现 GPU 入口：

| 分支 | 用途 |
| --- | --- |
| 原样续写两次 | 检查同实际状态的逐 token 可复现性 |
| 一般回顾 | 活性对照：在相同新增生成上限下重新检查问题与已有证据 |
| 原问题约束核对 | 目标动作：明确实体/关系/时间/比较约束，引用已有证据；不输入 gold |

保持原模型/检索资产/seed/解码，总检索上限 4，不重置已用调用。两个恢复动作
使用原始每段生成上限 512（并非整个分支总上限）并记录实际 instruction/input/generated token
差，而非假装实际成本完全一致。无剩余 search 时仍可使用已有证据作答；若有
context/format 失败，单列，不删样本。不得按 action 是否成功反过来挑机制案例。

历史压缩/重建属于另一种干预，若加入须保留完整实际 evidence、加同成本摘要
对照并另冻协议，不能声称原 raw prefix 未变。shared A1 replay validator 仍只接收 clean source；新 `post_evidence` 单独验证实际
扰动历史与 post-observation 起点，不能将 perturbed condition 改成 E0 或绕过
原始 token 检查。当前已实现 `prepare_e` / `run_e`，H800 执行待厂内反馈。

## 8. F：独立确认与选择性触发

F 才使用新题：从已有本地数据冻结未观察 ID，排除全部已观察 1000 题、pilot、
机制案例及近重复；开发/确认按整题分离。在 E 的控制与方法冻结后跑 F，不能
先看新题结果再挑触发条件。环境/资产沿用，不需下载新模型或配置环境。

触发器仅可读当时原问题、历史、当前观测与剩余预算；禁止使用 clean 是否正确、
gold 标签、最终答案对错或之后检索到的证据。F 报告全量 NetEM、Recovery、
StableHarm、触发覆盖和增量成本，不只报告 72 个 outcome-selected 候选。
确认题数/统计计划需预先冻结，不用当前代理计数倒推显著性门槛。RL 仍未 GO。

## 9. 这次需要反馈的最小内容

先反馈正文引用校准的准确/错误/未知数量、pending 覆盖、141 计数的互斥/多标签
解释、空答案的 raw 状态交叉表、matched-E1 的单位/上下界/匹配问题数、以及
72 候选与非 oracle 恢复见证的 ID 交集及相同起点证明。已有两次手工案例保留，
加至少一个高 overlap 但约束不成立的反例和 stable 对照。输出保持本机，不 push。

# E：实际扰动历史后的回顾动作工程实验

2026-10-12。本轮沿用厂内已有模型、数据、Flat 索引和常驻 Retriever，
不配置环境，不下载资产，不重跑规模 A1/A3，不启动训练。
这里 E 是机制审计任务；E1–E4 是原检索干预条件；研究阶段 C 的 RL 仍未 GO。
公开主线提供唯一实现，厂内负责执行并反馈。

## 1. 最新报告的验收与五个问题的决定

接受厂内 `0ea168a` 报告中的部分校准进展：8/175 事件已审，4 sufficient、
2 insufficient、2 uncertain，167 pending。引用存在性通过机械检查有帮助，
语义充分性仍依赖独立审查；开发服务器没有原始产物，未重新计算这些数字。
撤回旧自动语义标签和“72 真 L3”后，不再据此估计语义失败比例。

| 问题 | 本轮决定 |
| --- | --- |
| 谁实现 E runner | 家服实现 `work2.post_evidence`、`work2.prepare_e`、`work2.run_e`；厂内沿用并运行 |
| 最低校准覆盖 | 8 个已审事件可用于准备工程案例。GPU smoke 至少需要 **1 个完整失败/稳定配对**，两者都独立核验起点信息充分；不要求先读完 175。最多 4 对工程扩展；不能据此宣布完整 C/D 或总体语义精度 |
| matched-E1 是否再匹配长度/预算 | 同状态 E3/E1 的干预前 raw prefix、已用预算天然相同，先核验这些不变量。不要按干预后的长度或调用次数再筛选。E 中不同问题的失败/稳定配对才做固定长度/预算匹配 |
| 非实体答案 | 用实体消解、关系/年份比较或数值推导核验，不能永久自动归为 uncertain。缺乏推导时保持 uncertain；不阻塞实体案例 smoke，但在 F 前必须冻结覆盖与分层报告 |
| 141 来源 | 旧表 72+29+15+12+9+4=141，包含 uncertain；去掉 29 后是 112。“非 uncertain 子集”不能解释它。撤销该分布，采用原始事件 175、状态 134，并保留事件主标签/状态 mixed 规则 |

matched-E3 最新报告：178 状态、151 问题；E3 净 EM 计数 −10，同状态 E1 −13，
差值 +3，按题聚类 bootstrap 95% 计数区间 [−4,+10]，2000 次、seed 42。
点估计折合 3/178=+1.685 个百分点，但这个**计数区间不是平均差区间**：
重抽题目可能改变状态总数，不应直接除 178 冒充均值区间。
内容损伤的额外负效应尚未建立，也未证明等价或不存在失败。

空答案以新 raw 表为准：21 budget_exhaustion、8 generation_limit、3 malformed_action，
共 32；即 **21 预算耗尽 + 11 终止失败**。其中 6 个非实体类型是交叉分类，
不能作为第三种互斥终止原因；这些状态不能证明模型“有证据但主动不答”。

仍待对账：新报告 46 个非 oracle 恢复状态，与旧 union=52、B5 成功=3 在同一
全集/单位下不相容（非 oracle 至少应为 49）。需列出状态 ID、条件选择、成功
判据及是否改变样本范围。25 个交集仅是已撤回词法候选与恢复见证的交集，
并未核验同一 post-evidence 起点。此项不阻塞 E 工程，不得作为已证实恢复率。

## 2. 冻结的研究问题、起点与成本

问题限定为：在已核验有足够信息、但原扰动分支答错的开发案例上，
原问题约束核对是否比一般回顾更能修复错误，同时是否伤害稳定正确案例？
这是按已有结果选出的工程案例，不能回答总体收益、可部署触发效果或因果机制。

启动轮次固定为：

```text
resume_after_turn = max(原干预轮次, 独立审查的最早充分轮次)
```

使用**实际扰动分支**在该轮 observation 之后、下一次生成之前的历史。
例如 `hotpotqa:5a83ed9b t2 / E2` 的充分证据在 t1，仍从 t2 之后启动：
从 t1 启动会绕开尚未发生的 t2 干预。t3 才有充分证据的案例不能从 t1 启动。

保留之前生成的 raw token ID，不把生成文本重新 tokenize，不压缩历史。
观测 token 由冻结 tokenizer 按原 rollout 方式追加，核对原下一段输入长度，
如存在下一次保存的检索前缀再核对完整 token 序列。新产物记录恢复后的 token SHA；
历史文件未必独立保存过该 post-observation SHA，不能宣称与不存在的历史 SHA 比对。

每个案例有四次续写：原样 A、原样 B、一般回顾、原问题约束核对。
**全批案例的原样 A/B 逐 token 相同，且与原历史后缀一致后，才运行任何回顾动作。**
只向模型提供原问题、实际历史、固定英文回顾指令，不提供 gold、审查标签、引用
标注、错误类型或原答案对错。两个提示固定在 `post_evidence.PROMPTS` 并记录协议 hash。

继承 A1 manifest 的模型、tokenizer、资产身份、backend、TP、显存比例、seed、
温度和全部 decoding 上限；不重新套 diagnostic 默认配置。本 cohort 为官方 7B、
Flat、TP=1、显存比例 0.75、温度 0、seed 42、context 8192、Top3/20。
总成功检索最多 4 次，已用次数不重置；第 4 次观测后仍可生成答案。
**512 是每段生成上限，不是整条恢复分支总上限。** 起点已用 k 次检索时，
未来最多 4−k 次成功检索和 5−k 段生成，两种回顾相同。
指令长度和实际调用成本不假装相等，分别记录 instruction/input/generated tokens、
新增搜索及相对原样的差值。保留 context_limit、格式失败、预算耗尽等全部结果。

## 3. 先备份厂内分叉，再对齐公开主线

厂内已合并过家服代码，HEAD 仍可能有本地提交；普通 pull 不一定能快进。
在仓库执行既有对齐脚本，它拒绝未提交改动并先保留本地备份分支：

```bash
cd /root/develop/workspace/work2_byd
git status --short
GIT_TERMINAL_PROMPT=0 git -c credential.helper= fetch \
  https://github.com/CathyMuo/work2_byd.git main:refs/remotes/origin/main
bash scripts/align_public.sh --use-fetched-ref
python -m unittest discover -s tests -q
```

若存在未提交源码，先保存到厂内本地提交；不要丢弃。脚本保留忽略的 outputs、
configs/local.json 和旧提交备份，不运行 git clean。沿用当前 Search-R1 环境。

## 4. CPU：准备有固定顺序的失败/稳定配对候选

以下默认路径来自此前厂内报告；如果实际不同，显式传入对应路径。
原文件保持不变；`body_diagnostics.json` 是原始诊断包，`body_reviewed.json`
是通过 validate 的 8 事件部分校准文件，包含全部 pending 事件。

```bash
python -m work2.prepare_e \
  --source-run outputs/a1_clean_scale500 \
  --a3-rollouts outputs/a3_scale500_merged/paired_rollouts.jsonl \
  --packet outputs/semantic_audit/body_diagnostics.json \
  --reviewed outputs/semantic_audit/body_reviewed.json \
  --max-failed 4 --stable-proposals 3 \
  --output outputs/e_review/proposals.json
cp outputs/e_review/proposals.json outputs/e_review/reviewed.json
```

源码按 A1 冻结题序、原干预轮、E1–E4 顺序取最多四个独立问题：
原 clean 正确、扰动分支非空 answered 且 EM=0、独立 sufficient。
不把预算/终止失败混入这一轮机制问题。

稳定候选要求 clean 与同条件扰动均正确，同 dataset、干预条件、原干预轮、
启动轮及剩余预算；起点输入长度为失败案例的 0.75–1.25 倍。
先按长度差再按冻结题序排列，每对最多三个候选，全局不同题。
稳定性来自历史结果，**不是按新回顾结果选样**。
若无候选，该组需明确 skip，不放宽匹配或改换条件来强凑案例。

准备时重新执行原始 A1/A3 配对审计和文件 hash 绑定。未知 factory schema
明确失败；反馈错误和 `python -m work2.audit_pairs --schema-only ...` 的结构摘要，
由家服适配。不修改原始分支、不猜字段、不在厂内另写一套 runner。

## 5. 独立核验“只看到起点之前”的充分性

只编辑 `reviewed.json` 中每个 group 的 `decision/reason` 和各候选
`event.review`，其他上下文、顺序、轮次、协议、来源 hash 不变。
每个 event 已截断到该组起点，不能引用未来检索。复制的旧充分标签会被设置
`prefix_only_verified=false`，必须重新确认后才可运行。

逐组操作：

1. 检查失败案例的显示前缀，核验原问题实体、关系、时间、比较条件，确认可从
   这些实际正文推导答案，而非仅找到答案字符串。
2. 按稳定候选固定顺序审查；先前候选不足/未知须写 insufficient/uncertain，
   再看下一个。采用第一个 sufficient，不能跳过 pending 的近邻。
3. 两者皆 sufficient 且实际启动点信息充分，group `decision=run`；否则
   `decision=skip` 并写原因。所有 group 在 GPU 前明确 run/skip。
   一个组内后续未选稳定候选可以保持 pending。

充分 review 的结构示例（示例内容必须换成真实正文与真实解释）：

```json
{
  "decision": "sufficient",
  "reviewer": "独立审查者标识",
  "reason": "逐条说明原问题约束与实际证据的对应关系",
  "independent_of_studied_agent": true,
  "prefix_only_verified": true,
  "earliest_sufficient_turn": 1,
  "question_constraints_verified": true,
  "answer_derivation": "实体消解、关系链及答案推导，解释为何不是邻近主题答案",
  "supporting_quotes": [
    {"search_turn": 1, "document_id": "真实 document_id", "quote": "该文档正文的真实连续引文"}
  ]
}
```

每条引文须出现在其轮次的**可见正文**，轮次不晚于声明最早充分轮次，且至少
一条定位到该轮。机械通过只证明引文存在，不自动证明推导正确。
insufficient/uncertain 也需 reviewer、reason、independent_of_studied_agent=true，
earliest_sufficient_turn=null。

非实体的后续审查流程：先列原问题所指对象，给出实体消解引文，再列待比较的
关系/年份/数值与单位，写明比较或计算式。yes/no 由相应命题及否定/比较推出；
数值可由证据计算，不要求结果字符串直接出现。缺关系、歧义未解或资料不足时
保持 uncertain，并写缺口；不能由年份出现、答案未出现自动下结论。
本轮四个报告充分案例均来自 HotpotQA，不外推到 2Wiki 或非实体问题。

```bash
python -m work2.prepare_e --validate outputs/e_review/reviewed.json
python -m work2.run_e --plan outputs/e_review/reviewed.json --pair-limit 1 --check-only
```

这些检查只用 CPU，不访问 Retriever 或加载模型；真实资产/tokenizer/service
身份检查在 GPU 入口执行。全 skip 可以保存审查，但 runner 会拒绝零个可执行配对。

## 6. GPU：先一对 smoke，控制通过后最多四对

先确认原常驻 Retriever 可用，使用空闲 GPU；不停止或重启任何其他训练。
原样控制、实际历史及 runtime 验证都由 runner 处理。

```bash
mkdir -p outputs/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 bash scripts/run_e.sh \
  --plan outputs/e_review/reviewed.json --local configs/local.json \
  --pair-limit 1 --run-dir outputs/e_smoke \
  2>&1 | tee outputs/logs/e_smoke.log
```

一对包含失败和稳定两案例，四条分支/案例，共 8 条续写。
运行中以原子方式持续保存计划、manifest、分支、检索审计和指标；已有输出目录
会拒绝覆盖。原样失配退出码 2；资产/检索漂移或基础设施错误停止并保存部分结果。
所有计划案例保留 planned/controls_passed/error 状态，不只报告幸存者。
预检和原样控制产生的实际检索成本另存，不计入回顾动作的增量成本。

只有 manifest=completed 且 metrics 两项 engineering_gate 都为 true，才能做工程
扩展；此条件不看回顾是否成功，所以也不能因效果不好换案例。
使用同一审查计划、其固定前 N 对，N 最多 4，不修改提示或候选顺序：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_e.sh \
  --plan outputs/e_review/reviewed.json --local configs/local.json \
  --pair-limit 4 --run-dir outputs/e_engineering \
  2>&1 | tee outputs/logs/e_engineering.log
```

不足四个 run 组时只运行实际已核验的完整配对；不扩大为全 175 事件。
扩展会重复 smoke 对作为新运行中的控制，报告中不要把两次作为独立题累计。
新 Query 用正常 live 检索，已有 Query 与冻结 normal Top3 核对；原扰动观测
不会作为 normal 检索参考。新 Query 的未知语义漂移仍需案例审查，不能由检索
工程检查替代。无论结果如何，保持未 GO；F/触发器需要另冻协议后实施。

## 7. 回传内容与后续决策

输出目录包含：`frozen_plan.json`、`run_manifest.json`、`mechanism_rollouts.jsonl`、
`mechanism_metrics.json`、`retrieval_audit.json`。文件保留厂内，不提交公开仓库。
直接在会话反馈：

- 公共 commit、已审/跳过组数与原因、每组实际 ID/条件/干预轮/启动轮；
- 所有原样控制和历史后缀是否逐 token 相同、起点 SHA/剩余预算；
- 两动作在失败案例的 gain、在稳定案例的 harm、终止状态；
- instruction tokens、新增 generated/input tokens、search calls 及相对原样成本；
- manifest 状态、所有计划/完成/错误分母、Retriever 漂移记录与典型案例；
- 若并行对账，给 46/52/3 的 ID 集合口径，不再引用已撤回的 72 真 L3。

少量开发修复只支持动作/成本工程可行性；若约束核对与一般回顾同效、稳定 harm
明显或成本更高，应如实记录。下一轮再据此决定是否值得冻结新题 F，不能据
smoke 数字宣布机制成立、总体增益或启动 RL。

# E v2：充分致错与不足稳定的分层开发探针

历史执行协议。最新 `82fa3d5` 已报告工程通过，约束动作无额外优势并出现一次
预算伤害。当前停止方法扩展，**只做 CPU 收尾，不继续运行本文 GPU 命令**。
执行与研究决策以 [结果与方向文档](../research/E_V2_RESULTS_AND_DIRECTION.md) 为准。

2026-10-12。当前厂内报告 `faf8494` 已合并公开 `c0a1e8b`，E v1 四组全 skip，
GPU 未执行。**保留这项结果，旧 v1 验证器不放宽。** 本文定义新的 v2 问题和入口；
它不宣布旧 E 工程 gate 通过，也不进入 F/训练。

## 1. 对最新结果的判断与决策

四个充分致错案例、十二个判为不足的稳定候选、零个充分配对是已报告的准备
结果；开发服务器没有这些原始文件，未逐例重算。公开 v1 拒绝执行符合协议。

需要收窄报告里的解释：

- 12 个固定近邻没有充分候选，只说明这个候选池内未找到；不是完整稳定层普查，
  不能证明 cohort 中结构性不存在。前三个限制和全局不同题预留也限制了覆盖。
- 正确终局答案可能依赖后续检索、推理或模型知识。起点前缀已包含全部更早
  观测；“金标字符串未出现”不能单独证明信息不足，更不能证明参数知识来源。
- Somersby 的方位等关系是否足以推导国家，需要判断关系链与隐含条件，不能
  仅因为没有第二个字面 `England` 就判不足。数值答案也可能通过计算推出。
- 这轮尚无回顾动作输出，更没有 B3 的同起点机制对照。因此参数知识被干扰
  只是猜测，不是 B3 stable harm 的已证实原因。
- 厂内的 113 项测试属于其本地测试树；公开 c0a1e8b 有 71 项，v2 更新后有 77 项。
  保留厂内提交作本地备份，不声称导入不可获得的额外测试/源码。

本轮采用报告选项 2 的**受限版本**，并明确改变估计目标：

| 层 | 固定历史资格 | 本轮能回答什么 |
| --- | --- | --- |
| primary：`failed_sufficient` | clean 正确、实际扰动分支 answered 且错；起点信息充分 | 在这些已观察开发案例上，一般回顾与约束核对能否修复？ |
| secondary：`stable_insufficient` | clean 与实际扰动均正确；起点信息经独立核验不足 | 回顾是否伤害这些历史稳定案例？后续搜索/成本是否改变？ |

每个案例仍有**同一实际起点**上的原样 A/B、一般回顾、约束核对四条分支。
同案例的动作对照有效，并不要求另一个问题充当它的 token 对照。
primary 与 secondary 分开报告；它们不是可合并为总体 NetEM 的代表性样本，
也不把两个层的差值解释为证据充分性的因果作用。

选择范围固定在原 v1 文件里的四个失败案例与十二个候选，**不扩池找成功案例**。
每组只提出固定顺序中首个已独立判 insufficient 的稳定候选，最多 4+4 个案例。
原 dataset/condition/轮次/预算/长度近邻条件保留。改变的是安全诊断层的问题，
不是为旧“充分失败/充分稳定配对”强凑样本。

## 2. 与旧版相同的执行约束

沿用 [E v1 实施文档](E_EXECUTION_GUIDE.md) 的真实扰动历史恢复、资产身份、
原样控制、检索审计和成本口径：

- 启动轮次仍为 `max(原干预轮, 最早充分轮)`；稳定候选沿用所属失败案例的
  冻结启动轮，不拿未来证据标签回填起点。
- 不重 tokenize 已生成 token，不压缩实际历史，不把分支改写为 E0。
- 继承原 A1 manifest，包括官方 7B、Flat、原 seed/温度/backend/TP/显存比例。
  总成功搜索上限 4，已用次数不重置；512 是每段生成上限。
- 固定提示不变，不将 gold、证据审查结果、缺失约束标注或历史对错送给模型。
- **全批被选案例的原样 A/B 和原历史后缀逐 token 一致后，才执行任何回顾动作。**
- 保留全部计划、未完成、context/格式/预算及基础设施失败与真实增量成本。

v2 的 `prepare_e_probe` 负责不同的选择/审查协议，`run_e_probe` 直接调用原
`run_e` 的同一执行引擎；没有另一套 GPU rollout 或厂内自行实现的恢复逻辑。

## 3. 厂内先备份分叉，再对齐公开主线

不重复合并记录文件，不删除旧产物。对齐脚本先保存当前 HEAD 为本地备份：

```bash
cd /root/develop/workspace/work2_byd
git status --short
# 如有未提交源码，先保存到厂内本地提交；脚本会拒绝丢弃它们。
GIT_TERMINAL_PROMPT=0 git -c credential.helper= fetch \
  https://github.com/CathyMuo/work2_byd.git main:refs/remotes/origin/main
git show origin/main:scripts/align_public.sh > /tmp/work2_align_public.sh
bash /tmp/work2_align_public.sh --use-fetched-ref
python -m unittest discover -s tests -q
```

旧 `outputs/e_review/proposals.json`、`reviewed.json`、`review_summary.json` 原样
保留。本轮不需要新模型、新题、环境配置或任何训练进程操作。

## 4. CPU：从全 skip 的 v1 审查文件生成新计划

输入必须是已经 finalized 的 v1 全 skip 文件，并且原四个失败案例已确认
`prefix_only_verified=true`。v2 再校验 source hashes、真实 A1/A3 历史和原审查。
原来 skip 组里的候选也必须有独立 reviewer/reason，不能留下 pending。

```bash
python -m work2.prepare_e_probe \
  --parent-plan outputs/e_review/reviewed.json \
  --output outputs/e_probe_review/proposals.json
cp outputs/e_probe_review/proposals.json outputs/e_probe_review/reviewed.json
```

正常报告对应最多 8 个 case，每个旧组有一个 primary、一个 secondary。
如某组没有已核验 insufficient 稳定候选，就只有 primary。
旧 v1 文件 hash 被记录；修改旧文件会使新协议停止，不能将旧全 skip 擦掉。

只编辑新 `reviewed.json` 的 `cases[*].decision/reason` 与 `event.review`。
其他顺序、role、启动轮、来源、`prior_review` 和提示/协议保持冻结。
所有 case 的 `prefix_only_verified` 初始 false，须明确复核后才可 run。

### 充分致错 primary

保留独立充分审查的真实正文引文、最早充分轮、原问题约束与答案推导。
引用不晚于声明的充分轮，充分轮不晚于固定起点；模型实际只看该起点已有历史。
满足时 `decision=run`，写原因并设 `prefix_only_verified=true`。
若重新发现原判断有问题则 skip，保留解释，不反过来以新回顾效果选案例。

### 不足稳定 secondary

这一步尤其不能用“答案词没出现”作为充分理由。检查该截断前缀所有可见证据，
问：是否能消解目标实体、建立所需关系/时间/比较链，或通过已有数值计算出答案？
仅需更明确地记录此前独立审查的依据，不要求重读全 175 事件。

确认 insufficient 时填写（示例文字必须换成真实缺口）：

```json
{
  "decision": "insufficient",
  "reviewer": "独立审查者标识",
  "reason": "说明缺失的是哪条原问题约束，以及已显示证据为何无法补齐",
  "independent_of_studied_agent": true,
  "prefix_only_verified": true,
  "earliest_sufficient_turn": null,
  "supporting_quotes": [],
  "missing_question_constraints": ["真实缺失的实体/关系/时间或数值比较条件"],
  "alternative_derivations_considered": "核验别名、关系链、年份比较或计算等路径；说明为什么仍不足"
}
```

同时把该 case 的顶层 `decision=run`、`reason` 写明。
如果只是不确定能否推导，把 review 改成 uncertain，case 明确 skip；不要强判不足。
如果发现其实 sufficient，也在这个预定义不足层里 skip，记录应另列研究的原因，
不临时换成下一个候选。机械校验检查字段和来源，不能替代真实语义审查。

primary 可在没有 runnable secondary 时运行。这样只能反馈修复/成本，**stable
harm 未测量，不能填零或宣称安全**。本协议拒绝只有 secondary 而无 primary 的批次。
每个 case 必须在 GPU 前明确 run/skip，不允许跳过 pending。

```bash
python -m work2.prepare_e_probe --validate outputs/e_probe_review/reviewed.json
python -m work2.run_e_probe --group-limit 1 --check-only
```

`--check-only` 仍是 CPU 来源/选择检查，不加载模型或访问服务。
v2 使用新入口；不要把新文件交给旧 `run_e`，旧充分配对要求原样保留。

## 5. GPU：固定首组 smoke，然后工程扩展

使用当前 Retriever 和空闲 GPU。首个冻结可运行组先执行；通常有两案例、8 条
续写。若 secondary skip，可能只有一个 primary、4 条续写，报告覆盖缺口即可。

```bash
mkdir -p outputs/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 bash scripts/run_e_probe.sh \
  --plan outputs/e_probe_review/reviewed.json --local configs/local.json \
  --group-limit 1 --run-dir outputs/e_probe_smoke \
  2>&1 | tee outputs/logs/e_probe_smoke.log
```

控制失配退出码 2，整批在任何回顾前停止；基础设施/检索漂移保留部分结果后
报错停止。不要改历史、重新设置预算或换一套参数使其“通过”。已有输出目录
拒绝覆盖。先反馈错误；正常运行需要 manifest=completed 和两个 engineering_gate=true。

如果工程条件通过，使用同一冻结计划扩展至最多四组；该决定不依据修复成功与否：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_e_probe.sh \
  --plan outputs/e_probe_review/reviewed.json --local configs/local.json \
  --group-limit 4 --run-dir outputs/e_probe_engineering \
  2>&1 | tee outputs/logs/e_probe_engineering.log
```

输出名称复用引擎：`frozen_plan.json`、`run_manifest.json`、`mechanism_rollouts.jsonl`、
`mechanism_metrics.json`、`retrieval_audit.json`，位于全新的 v2 目录。
manifest 标注 `stage=E_stratified_probe`、v2 协议 hash、全部冻结 case 的 run/skip
决定。`--group-limit` 与旧引擎 `--pair-limit` 是同一个上限参数，但 v2 组不是充分配对。

## 6. 反馈、解释与停止点

直接在会话回传：公共 commit、parent plan hash、两层 run/skip 数及理由、所选
ID/条件/启动轮/剩余预算、所有原样控制与历史一致性、Retriever 漂移与完整分母。

分别反馈指标：

- `by_role.failed_sufficient`：两个动作相对原样的 EM/gain、状态和增量成本。
- `by_role.stable_insufficient`：两个动作的 harm、状态和成本。
- 每层的 `constraint_minus_general`：配对 EM/F1、新增 generated/input tokens、
  search calls 差值；这用于判断约束核对是否优于一般回顾，不能只看是否修复。
- `stratum_coverage`：计划/完成数与 secondary 是否测量。未完成保留，不能删掉。
- 典型案例：是否改答案、是否改后续 Query、是否预算/格式/上下文失败。

不要把层间结果加成总体 NetEM，不给四个观察案例作显著性结论，不把 secondary
harm 归因为参数知识，不宣称 B3 伤害的机制已验证。对缺少充分稳定层的安全性
也没有答案。开发探针结果将用于决定是否值得做进一步机制控制或停止方向；
只有随后方法、触发器、成本与未观察题目协议另行冻结后，才讨论 F。

46/52/3 的历史恢复 ID 对账仍未完成，不阻塞本轮；仍未 GO，不启动 RL。

# SI1 四卡 Pilot：结果分析与下一步决策

后续状态：本报告提出的 CPU 诊断已于 2026-10-13 由厂内报告完成；当前 ridge
原型归档 NO-GO，无必跑下一轮。见 [诊断与归档报告](SI1_CPU_DIAGNOSTICS_AND_NO_GO_20261013.md)。
以下保留诊断前的历史分析，不作为重跑指令。

2026-10-13。厂内分支 feat/selective-intervention，公开源 43473fc，本地报告提交
285cac7。来源为用户反馈文字；命名的 27K 汇总包及原始 workers/cases 不在家服，
以下区分来源数字、算术推导和代码核对，不声称家服已重算厂内真实结果。
旧 E/constraint 分支保持归档；SI1 是另行明确授权的新自然状态协议。

**当前决策：四卡工程按报告接受；现有选择器净收益未建立，不进入确认或 RL。
继续一轮已有产物的 CPU 诊断，冻结模型、动作、阈值与这 800 验证题。
不得将大量 keep 带来的低总体损害称为已证明“只在会获益时动手”。**

## 1. 实际试验与结果

2000 题：1200 train + 800 validation，HotpotQA/2Wiki 各半；排除历史已观察
题集。本轮 smoke 的 20 题按协议属于当前 train 子集，允许工程重复，但不作为
新增独立样本；“排除 smoke 已见题”应区分之前的其他已观察 cohort。

四个 TP=1 worker 各 500 题，全部 baseline/双原样控制通过全局屏障后才执行
干预臂。报告全部机械核验通过、全局控制一致，dev 80.8 min（controls 46.4，
actions 34.4），smoke 134 s。这支持工程可行性，不证明固定四倍加速或研究效果。

### 1.1 800 题 validation（EM 按计数换算）

| 策略 | 答对 | EM | gain | harm | net EM vs keep |
| --- | --- | --- | --- | --- | --- |
| keep | 230 | .28750 | 0 | 0 | 0 |
| review | 223 | .27875 | 27 | 34 | −.00875 |
| budget | 172 | .21500 | 24 | 82 | −.07250 |
| answer | 21 | .02625 | 1 | 210 | −.26125 |
| selector | 228 | .28500 | 3 | 5 | −.00250 |
| 有限动作 oracle | 271 | .33875 | 41 | 0（包含 keep） | +.05125 |

selector 相对 keep 的报告 bootstrap 95% CI 为 [−.010,+.00375]，即
[−1.0,+0.375] 个百分点。未展示正向优势，也未统计证明所有选择器不可学。
keep 是预设参照；验证集最佳固定臂与 train 冻结最佳固定臂应分别登记，
后者的 frozen_fixed_arm 字段尚未在本次反馈中给出。

## 2. 低损害是否说明学会了避险？目前不能

源码 `arm_summary` 中条件损害率的分母是**所有 keep 正确的题**，本轮为 230。
selector 的 2.2% 是 5/230；它不是被干预且原本正确题的条件失败率。

选择分布：keep 649、review 135、budget 14、answer 2。因此：

| 算术项 | 值 |
| --- | --- |
| 不干预比例 | 649/800 = 81.125% |
| 干预比例 | 151/800 = 18.875% |
| 每次触发 gain | 3/151 = 1.99% |
| 每次触发 harm | 5/151 = 3.31% |
| 触发后 EM 不变 | 143/151 = 94.70% |
| 所有题的 harm | 5/800 = .625% |
| 所有 keep 正确题的 harm | 5/230 = 2.17% |

143 次 EM 不变不意味着 F1/费用也不变。真正的“被触发原本正确题损害率”
还需要分母 C_triggered，目前反馈未给，不能用 230 替代。

全量 keep 的伤害为零。把 18.875% 触发的 selector 与 100% 固定干预比较，
低损害包含暴露减少的作用；不能仅凭 2.2% vs 14.8%–91.3% 宣布“显著学会
安全触发”或“会在获益时动手”。

预设且已实现的核心对照正是 random_matched_rate：在可执行状态中随机排列
完全相同的动作数量。需要它及 simple_fixed_rule、risk_neutral_ablation 的
真实结果，再判断信息选择、少干预和风险惩罚各自的作用。没有置信/配对证据
时，“显著”应改为“观察到较低的汇总损害”。

### 2.1 一个说明暴露效应的例子，非真实对照结果

**假设全部 800 题可执行**，同数量 review135/budget14/answer2 随机分配，则：

```text
期望 gain = (135×27 + 14×24 + 2×1)/800 = 4.97875
期望 harm = (135×34 + 14×82 + 2×210)/800 = 7.6975
期望 net EM = (4.97875−7.6975)/800 = −.0033984375
```

selector 实际 net 为 −.0025，距这个示例仅约 .00090 EM（.09 个百分点）。
这说明大幅低于固定臂的 harm 可以主要来自少触发。实际 random control 会受
可执行数、动作分布与随机实现影响，必须读现有报告，不把该例当已跑结果或检验。

## 3. Oracle headroom 表示动作机会，不等于可识别机会

keep230、oracle271，说明有 **41 个 keep 错误题**被至少一个已测试干预修复。
selector 只捕获 3 个，遗漏 38；另损失 5 个 keep 正确题。
对有限 oracle 的 EM 差共 43/800，由“漏掉 38 gain + 引入 5 harm”构成。

review/budget/answer 的 gain 成员数 27+24+1=52，联合独立题数 41，存在交叠。
需分开：review 独有机会、budget 独有机会、共同机会、answer 独有机会；
按 train/validation 和数据集报告，而不是只看 oracle 总值。

answer 只有一个 gain，因此从有限 oracle 中去掉它最多损失 1/800=.125 个
百分点；review+budget 的 headroom 至少为 40/800=5.0 个百分点。
这是有限数据的事后诊断，不授权用验证结果改动作集合再宣称重新验证。

有限 oracle 按终局选臂，看到选择器禁止读取的信息。41 个机会可能依赖无法
由当前 prefix 识别的因素；headroom 不保证已有特征或更大模型可以兑现。
当前结果支持有限动作差异存在（H1 的局部读数），不支持前缀选择净收益（H2）。

## 4. 两个失败臂首先是动作契约问题

### 4.1 answer 的 722 次失败

公开 `continue_arm` 的 answer：追加仅作答指令，最多一次生成；若仍输出
search，则不执行检索，记录 answer_only_search_refused，没有免费重试。
722/800 = 90.25% 是该动作的搜索拒绝率。210/230 = 91.3% 则是原本正确题
的损害率，两者是不同分母。

所以结果说明：这个 checkpoint 在自然第一观测后，对该提示与禁搜契约的
遵从/兼容性很差。它**没有成功强制生成 800 个最终答案**，不能将其概括为
“所有基于已有证据的作答策略都不可行”或“证据普遍不足”。

当前 answer 不应作为已成立的恢复方法；保留为动作兼容性的负对照。
原始结果不修改，不放宽 parser、不补救拒绝、不重新给免费生成。
若未来设计新作答动作，它是新协议，需 train-only 设计与未观察评估。

### 4.2 budget 的 170 malformed_action

budget 仅追加 used/remaining/total 的提示，实际总预算仍 4，不是增加预算或
完整预算控制算法。170/800=21.25% 的 malformed 需要检查最后原始文本：
没有动作标签、多个动作、空动作、错误标签等。它们是政策输出失败，机械审计
通过不等于动作设计有效。先统计 train 与 validation 的类型，不能直接归因为
预算规划能力或环境故障。当前不改评分/格式规则来“修好”指标。

## 5. 选择器当前看得到什么、学不到什么

固定 first-observation anchor：所有 reachable 状态 spent=1、remaining=3，
当前代码不是在多预算阶段采样。spent/remaining 特征是常数，不能学何时临近
预算边界。simple_rule 中 remaining<=1 选择 budget 的分支实际不会触发。
状态只含一个已发生 query，也没有多轮语义漂移轨迹可观察。

特征为 19 个数值/词标记 + 64 维 signed hash 词袋，多输出 ridge。
它可测廉价表面信号，但词重叠不是语义充分性，hash 特征不是完整关系推理。
这次失败限定于该动作集合、anchor、特征和冻结风险设置，不宣称所有语义
选择器都不可学，也不据此直接更换大模型/开始 RL。

冻结打分：

```text
utility = clipped_gain − 2×clipped_harm − .01×predicted_delta_calls
仅 utility > .02 才干预
```

回归输出裁剪不是校准概率。节省调用时 delta_calls<0，成本项会抬高 utility，
可能在没有正准确率净分数时触发。CPU 诊断会统计此类越过 margin 的选择，
不能自动将这些触发称为“预计会修复错误”。

若确实沿用该冻结 risk_weight=2/call_penalty=.01，即使151触发每题都省去最多
3次调用，其经验打分均值也最多：

```text
(3 − 2×5 + .01×151×3)/800 = −.0030875
```

因此已报告点读数不能由这个冻结的调用费用项挽救为正经验 utility。
该算术不替代实际 token/延迟成本，也不是统计显著性结论。
真实 search_calls/generated_tokens/model_input_tokens 和风险消融仍需补充。

## 6. 下一步：已有产物 CPU 诊断，不继续当前验证集调参

当前 SI1 定位为“工程通过，现有 ridge 未建立净收益”。不立即安排新确认题
或 RL。新学习器是否值得做取决于以下现有数据证据：

1. 是否优于相同触发率/动作比例的 random control 和简单固定规则？
2. 高分是否对应实际 gain，而非低估 harm 或只奖励省调用？
3. train 的动作机会和错误类型是什么，train 冻结最佳固定臂是否也为 keep？
4. Oracle 机会由 review/budget 哪些模式贡献，是否有真实可观察区别？
5. train 内有无稳定的预算/语义信号；这里只做诊断，不将 in-sample 拟合当泛化。
6. validation 按 dataset 的方向与实际成本是否一致，是否由少数例子/失败类型主导？

已有报告含 random_matched_rate、risk_neutral_ablation、simple_fixed_rule、
frozen_fixed_arm、by_dataset。本次反馈未附这些数值；先读它们，不重跑生成。
新增只读 CPU 工具。厂内有本地提交 285cac7 时普通 pull 可能不能快进；先
保存本地 HEAD 再对齐公开分支，忽略的 outputs/configs 不清理：

```bash
cd /root/develop/workspace/work2_byd
git status --short
# 若非空，先保存未提交源码；下面只针对已提交的分叉。
GIT_TERMINAL_PROMPT=0 git -c credential.helper= fetch \
  https://github.com/CathyMuo/work2_byd.git \
  feat/selective-intervention:refs/remotes/origin/feat/selective-intervention
git branch "factory-backup/si1-report-$(date -u +%Y%m%dT%H%M%SZ)" HEAD
git switch -C feat/selective-intervention origin/feat/selective-intervention
```

仅当工作树干净才执行切换；保留备份，不 merge/reset/clean 旧产物。
随后运行诊断：

```bash
# 使用已存在的 Search-R1 环境解释器；不配置/重建环境。
export SI1_PYTHON=/实际已有环境/bin/python
"$SI1_PYTHON" -m work2.diagnose_selective \
  --run-dir outputs/si1_dp4_dev \
  --output outputs/si1_dp4_diagnostics/report.json
```

它复用原 raw/四卡屏障来源检查，核对冻结模型/状态/决定并重算预设对照；
输出真实 trigger 分母、oracle 机会交叠、train 特征变化、费用项触发数与分层
原始统计。只新建诊断文件，不覆盖 old artifacts，不拟合、不加载模型、不调用
Retriever。诊断结果是 post-report 解释，800 题不能由此变回未观察确认集。

若只是随机低覆盖水平，无可解释的 train 信号，则当前原型 NO-GO；如果存在
可观察、跨 train 内划分稳定的信号，另冻 SI2 模型/特征/动作协议和新验证题，
不围绕这 800 题寻找最佳阈值。以上是下一步决策条件，不是已经获得 GO。

## 7. 两个工程问题与本次修复

- run_si1.sh 现支持 `SI1_PYTHON`（优先）、`PY`（兼容）、当前 python（默认）。
  各模式使用同一选择解释器，workers 继承 sys.executable；不硬编码厂内路径。
- test_selective/test_parallel_selective 的协议 fixture 用仓库 ROOT 绝对定位，
  不依赖运行 cwd。修改不改变动作、解码、选择器或历史结果。

推荐复用当前 searchr1 环境的解释器绝对路径；脚本能检查路径存在，不保证默认
base python 已安装 vLLM。环境管理仍由用户负责，不在家服重建。

验证：111 项公开 CPU 测试通过；在 tests/ 目录运行 30 项 selective 相关测试
也通过（工作区包在 PYTHONPATH）。解释器覆盖使用 stub 验证，不访问服务/GPU。
CLI、shell、Python 编译和文档链接检查通过。未获取/复算厂内原始报告包。

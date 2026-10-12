# SI1 独立尝试性实验协议

2026-10-12，用户明确要求选定方向、实现、推送及提供厂内运行指南。Stage为新SI1实现/开发探针，不恢复旧E/F；旧constraint仍NO-GO，RL未获GO。

## 主问题与假设

在自然搜索状态上，当前可见信息能否预测一次固定干预相对冻结Search-R1的gain/harm，并在整题隔离的新验证题上改善净效果？H1是有限动作集合有可利用差异；H2是prefix-only选择能兑现差异；H3是简单固定动作/规则未解决全部机会。三者都仍待模型实验，不以文献阅读或CPU测试当成立。

## 冻结内容

- 模型资源：历史7B Search-R1 PPO；E5-base-v2/Wiki18/FAISS Flat；本地资源由厂内继承并登记。
- 4次全局成功搜索、8192 context、每段512 generation、greedy/seed42、Top3/20、TP1。
- 按整题预选HotpotQA/2Wiki各120题，排除提供的所有历史题ID和问题文本。160train/80validation，无结果筛选。
- anchor固定第一次真实观测后，一题一次；无检索题保留原策略与端到端分母。不是已证实语义漂移的条件子集。
- K/V/G/A动作定义在`work2.selective.PROMPTS`，manifest冻结其hash。全体双原样控制与历史吻合后才运行其他臂。
- A禁搜改变动作空间：未遵守就失败，不加额外免费生成；原总预算不增加。
- 状态允许question、实际历史/可见正文、当前query、已用/剩余预算与长度。禁止gold、支持事实、未来、原策略终局、其他臂结果和hidden候选。
- 标签是同状态实际EM/F1、gain/harm及全成本；温度0实际差值不等于准确期望Q或全局最优。
- 原型固定为19个数值/词标记+64维带符号hash词袋，多输出ridge预测三个干预的gain/harm/relative-call-cost，risk_weight=2/margin=.02/call_penalty=.01/ridge=5。无外部judge或新依赖。
- 验证前select只读冻结模型和states；evaluate后读cases并复算决定。可见状态token长度由实际anchor核验。

## 对照和判断

原样、三个固定臂、训练集冻结最佳固定臂、固定规则、相同动作比例随机选择、risk_weight=1消融。全题EM/F1、gain/harm/net、条件harm、搜索/输入/生成费用、失败类别与题级配对区间；数据集分别报告。有限四臂oracle仅诊断。

只有开发数据中动作机会、可观察选择及净改善均支持时，再安排直接论文基线和独立确认。VOI/R²-Searcher/SAAS等尚未完整适配，当前不得声称超越它们或完成新颖性证明。弱ridge失败仅否定本原型，若人工证据支持明显可观察语义信号，需另列后续模型评审；不在80验证题上改提示继续试。

## 成本和停止

收集需要每题完整baseline、两条原样控制及三个干预续跑。baseline本身在anchor后的结果是历史对照，不额外重复计为独立题。smoke仅20个train题；全开发重复其中20题，不增加独立样本数。

停止于基础设施或控制不一致；未完成覆盖不进入分析；烟雾测试不拟合。目录拒绝覆盖，source/hash/coverage/raw分数和费用审计。坏结果不可由completed日志掩盖。不得事后改动作/分母/验证集，或根据oracle选臂当部署表现。

确认与RL继续pending。旧46/52/3不阻塞SI1，旧四题不得训练/选规则/确认，GMST方法/数据/停止控制器不引入。论文叙事可衔接，实验保持独立。

厂内命令见[执行指导](../docs/SI1_FACTORY_GUIDE.md)。

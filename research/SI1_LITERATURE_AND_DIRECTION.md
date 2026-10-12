# 直接近邻的方法阅读与SI1实现取舍

2026-10-12。阅读范围：下面原文的方法、关键算法/奖励与相关附录；另检查可访问作者仓库的结构/训练入口。没有运行作者实验，没有把所有官方框架完整复现。这里只给本次立项需要的方法结论，不声称穷尽相关研究。

## 1. 锁定的实现

**冻结已有离线搜索Agent，学习一次干预相对原策略的收益与损害，用当前真实状态选择是否干预。** 第一轮仅自然状态、四个固定动作、一个anchor与一个预算，先测可选择性。

理由是实现风险低且结果容易反驳：已有原始token前缀、正常检索和同预算续跑接口可复用；标签由实际分支结果产生；轻量学习器不需要新teacher、权重或训练框架；完整gain/harm可直接检查是否只是修复失败题同时破坏正确题。首轮可以产生清楚的GO/NO-GO证据，不保证正效果或ACL新颖性。

保留原样、不追加干预是显式参照。监督目标为gain/harm及相对费用，部署候选只读当时状态。80题验证先写prefix-only决定，再读结果评价；不是gold挑臂。

## 2. VOI：最直接的预算控制近邻

[Inference-Time Budget Control for LLM Search Agents](https://arxiv.org/html/2605.05701v1)，重点§3、§4.1—4.3、附录B.1—B.3。

其搜索控制在当前状态、剩余工具/输出预算上，对Search/Decompose/Answer计算进展、结构信号、预算惩罚和成本归一化，再加保守guard。方法部分明确其执行分数由固定系数和特征规则构成，不是学习的value模型。答案阶段从既有轨迹构造窄范围候选，按支持与风险规则决定是否替换。

对我们的影响：预算可见、状态控制、保守保持和净收益风险都已出现。SI1能争取的是从真实同状态结果学习相对原策略效果及正常状态损害；仅改成learned controller并不足以自动形成贡献。VOI实验使用全程工具/输出双硬预算，而我们历史为调用上限+每段生成限额；统一比较时必须处理这个差异。

本轮没有找到并引入可确认对应的作者控制器源码，因此不编造仓库或“官方复现”。将它列为下一阶段算法适配基线，不把简单budget提示称为VOI。

## 3. Search-R2：完成轨迹后的选择性修复

[Search-R2](https://arxiv.org/html/2602.03647v1)，重点§3.2—3.5、Algorithm1、奖励定义与附录提示。

Actor先生成一条轨迹，Meta-Refiner的discriminator判断全局一致性，不通过时trimmer选择cut-point，保留前缀再生成后缀，最多迭代Nmax。Actor和Refiner共享模型，通过包含修复动作的增强轨迹联合GRPO训练；过程奖励评估检索片段效用，并由终局正确性门控。

与SI1的关系：它已覆盖有选择地保留/修复及收益分解，不能说首次提出选择性恢复。其Algorithm1在已生成轨迹上判别、切断和重生成；我们的第一轮在第一次真实观测后，不读取尚未发生的原策略终局，学习一次干预相对原策略的gain/harm。完整Search-R2还需要过程judge/训练与多次轨迹修复；首轮不移入这些成本。

局部区别是可观察信息与监督对象，不是因果方法已全新。后续必须做有费用记账的适配对照。

## 4. R²-Searcher：约束回顾已被强覆盖

[R²-Searcher](https://arxiv.org/html/2606.28566v1)，重点§3.1—3.4、公式6—11。

查询token要素引导事实抽取，形成reasoning region；对足够/不足/错配的检索分别回答、找缺失信息或修订query；先SFT获得复合格式，再R²PO树形rollout共同探索事实抽取/反思/搜索。其过程奖励含事实/格式等规则项，不只是EM。

检查[官方仓库](https://github.com/szhang-cityu/R2-Searcher)的`train.sh`，版本`8556739c6a02db61b181762d52fb6c8a0370db88`：入口`verl.trainer.main_ppo_format_ts`，tree advantage、多个树形采样参数和已SFT模型路径。这不是在当前PPO checkpoint上加一段constraint提示就能等同实现的方法。

结论：不继续旧constraint文字优化，不将“回顾原问题约束+GRPO”当新方法；官方训练栈及模型准备放到信号成立后的对照阶段。SI1保留generic review强简单臂，避免专门方法只赢弱基线。

## 5. AnySearch：预算意识已有训练实现

[AnySearch](https://arxiv.org/html/2609.00813v1)，重点§3.2—3.4、附录C。

PhaseI显式注入预算状态与证据充分/预算策略推理支架，并递减预算；PhaseII去掉支架，按近期各预算正确率自适应采样。奖励结合准确率、格式、长度以及绝对/相对搜索效率；难题上效率权重降低，避免简单压制必要搜索。

[官方实现](https://github.com/xwsun01/AnySearch)是Slime上的`examples/anysearch`，含独立curriculum/prompts/protocol/rewards和E5+Flat服务。改变当前环境到Slime/Megatron会引入不必要工程风险，因此首轮复用原环境，用V臂检查预算提示是否已消除主要差距；V不是AnySearch完整复现。SI1首轮只budget4，不能宣称任意预算能力。

## 6. SAAS：禁搜对照与模型能力边界

[SAAS](https://arxiv.org/html/2605.29796v1)，重点§3.1—3.3。

其search-disabled/enabled rollout groups比较当前策略对同一问题的成功数，分NoSearch/NeedSearch/Undetermined；以正确性门控的搜索奖励约束不必要与超过最少成功调用数的搜索，并分阶段优化。主要边界是随策略变化的模型能力，而不是人类给出的固定题目难度。

[官方代码](https://github.com/XMUDeepLIT/SAAS)，检查版本`d27fa5450f2316ff531d915bcff73e010a172635`，仓库包含Slime及训练入口。无新权重/环境导入。

这要求我们不能把A与K对照称为全新反事实设计。SI1的A基于已有真实观测而非问题起点纯参数知识；比较用于测当前动作的终局效果与损害。证据充分不保证模型立即答对这一点必须保留。

## 7. CRRL：上下文精炼与历史masking

[CRRL原文PDF](https://aclanthology.org/2026.acl-long.160.pdf)，重点§3.2、§3.3、训练设计及消融；[作者仓库](https://github.com/AmourWaltz/CRRL)。

原文分别mask先前documents、queries、thinking，检查上下文干扰；进一步针对最新检索内容由teacher抽取关键事实，保留正确轨迹的抽取对，并用entailment检查抽取正文不增加事实，训练Context Refiner，再结合搜索训练。

其方法的输入处理对象是检索正文；不等同于我们选择是否实施一次干预。为了保持真实前缀和可审计状态，SI1不压缩/替换历史，不搬一个refiner进来改变所有臂。后续若主张源自证据处理，CRRL必须进入强对照。

## 8. AutoSearch、RAIL、RICE-PO和LAPO的边界

- [AutoSearch](https://aclanthology.org/2026.findings-acl.1399.pdf)：中间自作答识别最小充分深度，配合效率、稳定和搜索质量奖励。A臂不能宣传为首次评估已有证据能否作答，提前停止也不是新贡献。
- [RAIL](https://arxiv.org/html/2608.05080v1)：§2—3将训练rollout干预做成可学习的上下文决策，recoverability目标对应奖励分布对比/方差信号。SI1只预测部署时相对最终QA结果，不优化rollout奖励方差；也不将RAIL“恢复性”直接改名成我们的目标。
- [RICE-PO](https://arxiv.org/html/2605.26352v1)：§3—4针对可执行summary与latent reasoning区分可测性，用局部分叉的检索反馈与未来残余稳定性决定信用传播。主要实验为BRIGHT/BEIR，不将检索分数直接当QA净收益。
- [LAPO](https://arxiv.org/html/2607.13501v1)：§3删除某完整search interaction但保留其他未来上下文，测gold答案似然差并做符号一致门控；是回顾归因。SI1从当前状态实际执行新动作、正常续跑到任务结果，不读取未来来触发，也不以gold似然作部署特征。

## 9. 为什么采用这条路线并保留失败出口

直接端到端重训R²PO、换Slime做AnySearch/SAAS，当前会先承担训练栈和权重成本，仍不回答是否存在我们的独立方法空间。SI1能在同一环境下先产出完整动作转移、自然题上的机会、可见特征是否可预测及真实损害。

它可作为后续生成式策略学习的数据基础，但当前ridge仅是廉价探针。若没有动作机会、简单budget/review已经占优，或者选择只靠事后gold成立，就停止；若标签明显有可观察语义结构而廉价词袋未学到，应另评估语义模型，不能直接宣布所有选择不可学。80验证题不继续作调参集。

本次实现没有复制作者代码或改变原submodule，因此没有新增许可/大框架依赖。论文链接与代码版本用于阅读记录，不视为作者方法已在本项目跑通。下一步按[厂内指导](../docs/SI1_FACTORY_GUIDE.md)执行，用真实报告决定是否值得适配强近邻并进入独立确认。

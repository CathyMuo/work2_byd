# SI1 开发验证记录

2026-10-12。开发分支 feat/selective-intervention，基础提交 a425cb8。

完整标准库CPU测试 **95项通过**（原81项+新增14项）。新增覆盖：原始token与总预算继承、双原样历史一致、全局控制失败后零干预、答复限定的实际禁搜、gold/hidden候选/未来不进入特征、无检索与context-limit无机会题保留分母、ID/问题排除与split冻结、完整合成smoke→dev→audit→fit→select→evaluate、模型资源变化拒绝旧smoke、来源哈希及原始分数/费用审计、烟雾结果拒绝训练、验证对决策/模型来源的复算。

Linux脚本 bash -n、模块编译、命令帮助、文档本地链接和git diff检查通过。合成测试仅使用FakeGenerator/FakeRetriever，没有实际加载模型/检索服务/运行GPU；它们不是实验后端，不对用户开放为推理替代。

原始厂内轨迹仍不可用，本轮没有复算历史H800实验。新SI1厂内模型结果、强论文基线、独立确认与RL均pending。当前95项证明工程协议可运行，不证明选择器有效、安全或具有新颖性。

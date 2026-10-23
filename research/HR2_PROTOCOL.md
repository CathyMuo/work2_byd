# HR2 预登记协议：在 Search-R1 v0.3 上重做交接重放

登记：2026-10-23　分支：`feat/credit-testbed`　配置：`configs/handoff_v2.json`　脚本：`scripts/run_hr2.sh`
用户决定（2026-10-23）：在 v0.3 上重新登记 HR1。本文件在看到任何 HR2 结果前冻结。
HR1 的协议与结果：`research/HANDOFF_PROTOCOL.md`、`research/HR1_RESULTS_AND_DECISION.md`。

## 1. 为什么重做

- HR1 判 NO-GO：gain_end = −0.5 pp [−4.9, 4.0]，R（v0.1 7B PPO）根本不比 I 强，没有可分解的增益。
- FC1 查明原因在检查点：v0.3 7B PPO 比 v0.1 高 13.5–15.4 pp，相对 Instruct 高 **+16.2 pp**（GRPO +14.3 pp），都显著（FC1 的 D3）。
- 所以“RL 增益来自获取、利用，还是与证据无关的部分”这个问题在 v0.3 上才有意义。

## 2. 与 HR1 相比改了什么（全部在看结果之前）

| 项 | HR1 | HR2 |
| --- | --- | --- |
| R | v0.1 7B PPO | **v0.3 7B PPO（主运行）**；**v0.3 7B GRPO（第二运行）** |
| 题目 | HotpotQA 150 + 2Wiki 150 | **HotpotQA 300 + 2Wiki 300**，按 `sha256(20261023:sample_id)` 排序；按 id 和题面排除全部历史题，**含 AC1、HR1、U1 的新题** |
| 臂 | real / placebo / evonly | 再加 **`evonly_placebo`**：去掉思考、保留查询、把第 1..t 次观测都换成冻结的他题文档 |
| G-A1(b) | real − placebo ≥ 5 pp | **evonly − evonly_placebo ≥ 5 pp**（见下） |
| 与证据无关的部分（主口径） | placebo 臂上的 R−I 差 | **evonly_placebo 臂上的 R−I 差**；HR1 口径一并报告 |
| G-A3(iii) | 不适用（只有一个 RL 检查点） | **适用**：PPO 与 GRPO 的分解份额比较 |

修改 G-A1(b) 和主口径的理由：HR1 里 placebo 臂只换文档、保留思考，而思考常常已经转述了证据，所以 placebo 只让 R 下降 1.2 pp，“换掉证据”并没有真正拿掉证据（HR1 结果 §3.2）。
`evonly` 与 `evonly_placebo` 都去掉了思考，两者只差观测是真是假，这才是干净的“证据有没有用”的对照。

不变的：I（Qwen2.5-7B-Instruct），两个策略都用 R 的 tokenizer 渲染同一个 bounded 模板（FC1 的 D1：bounded 与官方模板无显著差异），温度 1.0、top_p 1.0、k = 8，交接点 t ∈ {0,1,2,3} ∩ [0,T] 加 end，三阶段运行（r1 / i2 / r3），2×2 Shapley 分解，按题聚类、数据集内分层的 bootstrap（2000 次）。

## 3. 估计量

记号同 HR1：H^{arm}_{P→Q}(t)，end = 源轨迹的检索次数 T。
- 总增益 G_end = H_{R→R}(end) − H_{I→I}(end)，以及 G_0。
- 获取、利用：end 处 2×2 Shapley（real 臂），获取 + 利用 = G_end。
- **EF（主口径）** = H^{evonly_placebo}_{R→R}(end) − H^{evonly_placebo}_{I→I}(end)：两边都只剩查询，没有思考也没有真证据，剩下的差距来自“问什么”与无证据作答。
- 报告但不设门槛：HR1 口径的 EF（placebo 臂）、evonly 臂上的 R−I 差、各臂交接曲线、格式失败率。
- **D_ev** = H^{evonly}_{R→R}(end) − H^{evonly_placebo}_{R→R}(end)，只在 T_R ≥ 1 的题上算。

## 4. Go / No-Go（每个 R 各判一次；(iii) 用两个运行一起判）

- **G-A1 协议有效**：(a) 对 P ∈ {I, R}，H_{P→P}(0) − H_{P→P}(end) 的 95% CI 含 0；(b) D_ev ≥ 5 pp 且 CI 下界 > 0。
- **G-A2 精度**：获取、利用、EF（主口径）的 95% CI 半宽都 ≤ 3 pp。HR1 在 300 题时半宽 3.0–4.1 pp，600 题预计约 2.1–2.9 pp。
- **G-A3 信息量**，满足一条即可：
  - (i) EF 份额 ≥ 30%；
  - (ii) 获取或利用的份额 ≥ 60%，且 CI 下界 ≥ 50%；
  - (iii) 对获取或利用，PPO 份额 − GRPO 份额的绝对值 ≥ 20 pp，且同题配对 bootstrap 的 95% CI 不含 0（`handoff_analysis --compare-with`）。
- **主运行（PPO）Go = G-A1 ∧ G-A2 ∧ G-A3 ∧ G_end 的 CI 不含 0** → (b) 方向作为论文的第二部分（“RL 收益从哪来”）扩写。
- 只过 G-A1、G-A2 → 交接曲线和分解作为 (a) 的补充分析，不单独成文。
- G-A1 不过 → 交接协议在 v0.3 上也无效，(b)-A 彻底关闭。
- 机械值由 `work2.handoff_analysis` 给出；最终决定由人写进 `DECISION_LOG.md`。

## 5. 资源与时间

- 只做推理。模型：v0.3 7B PPO、v0.3 7B GRPO（FC1/U1 已下载）、Qwen2.5-7B-Instruct（HR1 已下载）。
- 估算：HR1 300 题 7.65 万次续写用时约 1 小时。HR2 的 PPO 平均检索约 4 次，交接点更多，每题约 400 次续写，600 题约 25 万次，**约 3–4 小时**；GRPO（约 1.9 次检索）约 2 小时。

## 6. 已知局限

- 与 HR1 相同：I 不是 R 的训练起点（R 由 base 训练）；evonly 与 evonly_placebo 重新编码，不再逐 token 相同；只用 HotpotQA 和 2Wiki。
- `evonly_placebo` 去掉了思考，R 在去掉思考后的分布偏移可能比 I 大（HR1 里 evonly 让 R 比 I 低 8.9 pp）。EF 主口径因此偏向低估 R 的“与证据无关的优势”，结果要连同 evonly 臂上的 R−I 差一起读。

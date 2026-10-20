# FC1 预登记协议：评测保真度检查（harness fidelity）

登记：2026-10-20　分支：`feat/credit-testbed`　配置：`configs/fidelity_bounded.json`、`configs/fidelity_official.json`
本文件在看到任何 FC1 结果之前冻结。

## 1. 为什么要做

HR1 发现，在我们的 harness 下，Search-R1 7B PPO 的 EM 不比 Qwen2.5-7B-Instruct 高（0.240 vs 0.250）；AC1 中 7B PPO（0.253）也低于 3B PPO（0.359）。
核查后发现两个可能压低表现的差异：

1. **检查点版本**：我们一直用的 7B 是最早的 `SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo`（v0.1，无版本后缀），而 3B 是 v0.3。官方后来发布了 v0.3 的 7B PPO / 7B GRPO 等。
2. **提示词**：官方评测模板（`third_party/Search-R1/scripts/data_process/qa_search_test_merge.py`）写的是 “You can search as many times as your want … you can directly provide the answer”；我们的 harness 改成了 “You may make at most 4 search calls … provide the answer”。

其他设置与官方评测一致：E5 + Wiki-18、Top-3、最多 4 轮、贪心验证、观测格式 `Doc i(Title: …) …`。
仍有一处没改：官方会把观测截断到 500 token，我们不截断。这一点如实报告。

(a) 的全部结论都建立在这个 harness 上，所以写论文之前，必须确认它没有系统性地低估智能体。

## 2. 设置

- **题目**：AC1 的 1000 道新题（`outputs/ac1_fresh`）。这批题已经用于 AC1，但 FC1 只比较 EM 水平，不检验假设，也不据此调参。
- **贪心解码**，其余同 AC1 的源轨迹引擎；每个（检查点 × 提示词）跑一次。
- **检查点**（厂内通过内部 HF 镜像获取，跑之前在决策日志登记实际仓库名）：

| 标签 | 检查点 |
| --- | --- |
| `v01_7b_ppo` | 现用的 7B PPO（v0.1） |
| `v03_3b_ppo` | AC1 第二检查点 |
| `v03_7b_ppo` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3` |
| `v03_7b_grpo` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.3` |
| `instruct_7b` | Qwen2.5-7B-Instruct，用自己的原生模板，作为“未经搜索 RL”的参照 |

- **提示词变体**：`bounded`（我们的）和 `official`（官方原文）。`v01_7b_ppo` 和 `v03_3b_ppo` 的 bounded 结果直接复用 AC1 已有的源轨迹（同一批题、同样的贪心设置）。

## 3. 预登记的判定规则（ΔEM 为同题配对差，按题聚类 bootstrap）

| 规则 | 条件 | 处置 |
| --- | --- | --- |
| **D1 提示词** | 对 (a) 已用的检查点（`v01_7b_ppo`、`v03_3b_ppo`），若 official − bounded ≥ +3 pp 且 CI 下界 > 0 | 我们的 bounded 提示词压低了该检查点 → 写论文前，在该检查点上用 official 提示词重跑 AC1 的 H1/H3（另行登记） |
| **D2 主 7B 检查点** | 若 `v03_7b_ppo` 在两种提示词中较好那一种下，比 `v01_7b_ppo` 在同一提示词下高 ≥ 3 pp 且 CI 下界 > 0 | (a) 的主 7B 检查点改为 v0.3 7B PPO，并在它上面重跑 AC1（另行登记） |
| **D3 RL 增益** | 各 RL 检查点（official）− `instruct_7b`（official） | 只报告。若 `v03_7b_ppo` ≥ 5 pp 且 CI 下界 > 0，而 v0.1 没有，则 HR1 的零增益归因于检查点或 harness；能否在 v0.3 上重新登记 HR1，由用户决定 |
| **D4 第三检查点** | `v03_7b_grpo` 能正常作答：格式 / 长度失败率 < 10% | 作为 AC1 的第三检查点候选（另行登记后运行） |

参考：第三方论文报告的 Search-R1 7B 在 HotpotQA / 2Wiki 上约 0.37–0.43 / 0.28–0.38，各论文的评测设置不同，**只作对照，不作门槛**。

## 4. 不做的事

- 不根据 FC1 结果修改 AC1 / HR1 已经出来的判定，只按 D1–D4 决定要不要另行登记重跑。
- 不改官方模板的文字，也不调解码参数。

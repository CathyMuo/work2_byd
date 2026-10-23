# U2 预登记协议：已发表恢复机制的同状态安慰剂审计（ReSeek、MR-Search）

登记：2026-10-23　分支：`feat/credit-testbed`　配置：`configs/u2_reseek.json`、`configs/u2_mrsearch.json`　脚本：`scripts/run_u2.sh`
用户决定（2026-10-23）：U2 直接完善代码和厂内指导。本文件在看到任何 U2 结果前冻结。
选型依据：家服本地文档《工作二_U2方法审计候选.md》（13 个候选里只有 ReSeek 有权重且 harness 一致；MR-Search 的反思提示可以不训练直接审计）。

## 1. 问题与表述

H1 已在两个检查点上确认：在按失败挑出的状态上，通用恢复操作（review、reflect）的修复率，大部分能被同状态安慰剂复现（均值回归）。
U2 把同一把尺子用到**作者自己发表的恢复机制**上。被审计的方法都只报告全题 EM，没有报告“按失败挑出的状态上的修复率”，所以 U2 的结论只能写成：
> 这些方法的恢复机制在按失败挑出的状态上的修复率，有 X% 能被同状态的安慰剂或重采样复现；它们的全题收益 = 救回 − 伤害，两项都没有经过安慰剂对照。

**不能**写成“它们报告的修复率是均值回归”。

## 2. 两个审计

### 2.1 ReSeek（arXiv 2510.00568）

- **智能体**：`TencentBAC/ReSeek-qwen2.5-3b-em-grpo`（标签 `reseek_3b`，作者唯一放出的权重）。可选的“只给提示”对照：Search-R1 v0.3 3B PPO 套 ReSeek 提示（标签 `v03_3b_ppo`），用来区分训练出来的机制和提示本身。
- **提示**：系统消息 `You are a helpful and harmless assistant.`；用户消息 = 仓库 `utils/preprocess_reseek_dataset.py` 的 `DEFAULT_USER_CONTENT_PREFIX.rstrip("\n") + question`。两者都用 ast 读出原文，逐字节核对过（`tests/test_u2.py` 里有 sha256）。
- **环境**：E5 + Wiki-18，Top-3，最多 4 次检索，观测格式 `Doc i(Title: t) text`，与仓库 `search/llm_agent/generation.py` 一致。
- **源轨迹**：每题一条贪心轨迹（同时就是保真度检查）。状态 s_t = 第 t 次观测之后的前缀，t = 1..T。
- **臂**（每个状态每臂 k = 4 条续写，温度 1.0）：

| 臂 | 构造 |
| --- | --- |
| `nat` | 自然续写。先采样到 `</judge>` 为止的“判断头”，原样保留再接着生成 |
| `force_no` | **同一个**判断头，在 `<judge>` 处截断，强制写 `No</judge>` 后接着生成（强制触发恢复） |
| `force_yes` | 同一个判断头，强制写 `Yes</judge>`（抑制恢复） |
| `perm` | 第 t 次观测换一种文档顺序（P1 安慰剂，非恒等排列，按题和轮次确定），自然续写 |

  判断头里没有 judge 的样本（模型直接检索或作答），`nat` 改为整段重新采样，强制臂把 `<judge>V</judge>` 放在该段开头；这种样本的比例作为保真度报告。
- **状态集合**：
  - **S_fail（主分析）**：源轨迹答错，且模型在第 t 步自己判了 Yes（接受了证据、然后失败）。
  - **S_natNo**：模型在第 t 步自己判了 No（作者设想的恢复场景）。
  - **S_ok**：源轨迹答对且判了 Yes，用来量伤害。
- **估计量**：各臂在各集合上的平均 EM（修复率 / 保持率）；S_fail 上 `force_no − nat`、`force_no − perm`，以及 explained = nat / force_no、perm / force_no；S_natNo 上 `nat − force_yes`（自然触发的恢复到底有没有作用）；S_ok 上 `force_no − nat`（强制恢复造成的伤害）；描述量：证据里含答案与否时判 No 的比例、额外检索次数。

### 2.2 MR-Search（arXiv 2603.11327）

- **智能体**：Search-R1 v0.3 7B PPO（主，`v03_7b_ppo`）与 v0.3 3B PPO（第二，`v03_3b_ppo`），沿用 U1 的本地配置。MR-Search 没有放出权重，所以审计的是论文自己的测试时设置 Search-R1-S（同一反思提示套在 Search-R1 上）。
- **提示**：第一回合用 Search-R1 官方模板原文（MR-Search 自己的初始提示也是它）。第二回合按 MR-Search `generation.py` 的做法：`'<|im_end|>\n' + 对用户消息套 chat template 后去掉默认系统段`，用户消息为仓库 `retry_prompt` 原文（逐字节核对过，含换行前的空格）。
- **源回合**：每题一条贪心的第一回合，最多 4 次检索。
- **臂**（第一回合之后，每臂 k = 4 次采样，温度 1.0；第二回合重新给 4 次检索预算，与 MR-Search 每回合独立的 max_turns 一致；最终答案 = 最后一个有效答案）：

| 臂 | 构造 |
| --- | --- |
| `reflect` | 原文反思提示作为新的用户回合 |
| `reask` | 同一段文字，只把反思句换成 “Answer the question again.”（去掉“反思并改进”的内容） |
| `neutral` | 与反思提示 token 数相同（±2）的无意义文字作为用户回合（P2 安慰剂） |
| `fresh` | 不带历史，从初始提示重新开一个回合（重试运气，pass@2 式） |

- **状态集合**：**S_wrong（主分析）** = 第一回合作答且答错；S_right = 第一回合答对（量伤害）；S_unanswered 单列。
- **估计量**：各臂在各集合上的平均最终 EM；S_wrong 上 `reflect − neutral / reask / fresh` 及对应 explained；S_right 上各臂相对第一回合的净变化（伤害）；全题净效应 = 救回 − 伤害；改答率、第二回合检索次数。

## 3. 题目与公共设置

- 题目来自 U1 冻结的新题（`outputs/u1_fresh`，HotpotQA / 2Wiki 来自 AC1 新题，另有 NQ、MuSiQue、Bamboogle），每个数据集按 `sha256(20261023:sample_id)` 取前 300 题（Bamboogle 全用），两个审计共用 `outputs/u2/plan`。U1 的题对这些智能体来说都没用于调参。
- 检索沿用 CT1 的批量检索、规范化排序和逐 worker 缓存。bootstrap 以题为单位、按数据集分层，2000 次。

## 4. 判定（每个智能体分别判，机械值由 `work2.u2_analysis` 给出）

- **F 保真度**（不过则该智能体的 H1 判定作废，只报告）：
  - ReSeek：源轨迹中 ≥ 80% 的观测之后出现 judge，且格式 / 长度失败 ≤ 10%。平均 EM 与论文 0.312 相差超过 5 pp 时标记（论文平均覆盖 7 个数据集，与我们的 5 个不同，所以只标记、不设门槛）。
  - MR-Search：第一回合格式 / 长度失败 ≤ 10%，且 `reflect` 臂第二回合作答率 ≥ 80%。
- **H1-U2（沿用 AC1 的写法）**：在主集合上，若 安慰剂修复率 / 方法修复率 ≥ 1/3，或安慰剂校正后的净值 ≤ 0，则判 **确认**。
  - ReSeek：主比较 = `nat`（同状态重采样）对 `force_no`；次比较 = `perm`。
  - MR-Search：主比较 = `neutral` 对 `reflect`；次比较 = `fresh`、`reask`。
- 另报（不设门槛）：差值的 95% CI；ReSeek 在 S_natNo 上 `nat − force_yes` 的 CI（如果下界 > 0，说明训练出来的 judge 有真实作用，要如实写）；两种方法在答对状态上的伤害。
- **论文用法**：两个方法都过 F 时，U2 进主文的“审计已发表方法”一节；H1-U2 不确认（即机制有明显超出安慰剂的修复）同样如实报告，作为“哪些恢复是真的”的正面例子。

## 5. 资源与时间（4×H800，只推理）

- 新下载：仅 ReSeek 3B（约 6 GB）。其余模型 U1 已有。
- 估算：ReSeek 约 1300 题 × 约 2.5 个状态 × 16 条 ≈ 5 万条 3B 续写，约 1–2 小时；MR-Search 每个智能体约 1300 × 16 ≈ 2 万个第二回合，7B 约 1.5–2 小时，3B 约 1 小时。合计约半天。

## 6. 已知局限

- ReSeek 只有 3B 权重；7B 没放出。
- MR-Search 审计的是测试时版本（论文的 Search-R1-S），不是 meta-RL 训练后的模型；训练版放到 U3（训练流程 pilot 通过之后）。
- ReSeek 的“判 No 必须再检索”只在训练中约束，推理时模型可能不遵守；`force_no` 臂的额外检索次数会报告。
- `force_*` 臂在 judge 处重新编码一小段文本，可能与原 token 边界不同；用真实 Qwen2.5 分词器测过，截断后文本与预期一致。

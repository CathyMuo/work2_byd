# SI1：选择性干预的厂内尝试性实验指导

日期：2026-10-12。当前分支：`feat/selective-intervention`。用户已授权选定方向、开发和推送，厂内运行仍待执行。旧constraint/E方法继续归档，本实验是独立SI1协议。

## 1. 这次锁定什么

**冻结当前Search-R1，只看第一次真实检索后的状态，预测一次干预相对原策略的gain/harm，从而选择继续、预算提醒、一般回顾或立即作答。** 首轮只检验新自然状态上的机会与可选择性，不训练7B，不下载新模型，不构造旧E扰动，不使用GMST方法/选例。

已有论文覆盖预算控制、反思、选择性修复和禁搜比较。此次将动作相对原策略的收益/损害作为可执行标签，先判断是否可预测；不是已经证实的新颖算法。文献详细方法阅读见[论文阅读与实现取舍](../research/SI1_LITERATURE_AND_DIRECTION.md)，独立协议见[研究计划](../research/SI1_PROTOCOL.md)。

## 2. 更新代码，保留厂内旧分叉

在原work2_byd仓库中操作。原厂内分叉不能直接拉main做ff-only；新分支独立，不删除旧提交或outputs。

```bash
cd /root/develop/workspace/work2_byd
GIT_TERMINAL_PROMPT=0 git -c credential.helper= fetch \
  https://github.com/CathyMuo/work2_byd.git \
  feat/selective-intervention:refs/remotes/origin/feat/selective-intervention
git show origin/feat/selective-intervention:scripts/switch_si1.sh > /tmp/work2_switch_si1.sh
bash /tmp/work2_switch_si1.sh
git pull --ff-only
python -m unittest discover -s tests -q
```

切换脚本先检查未提交改动，再备份当前HEAD到`factory-backup/si1-...`，然后切换新分支。若有未提交文件，它会停止；保存这些改动后再运行，不使用`reset --hard`或`git clean`。outputs、数据、local.json均保留。已有依赖提交不变，无需拉新的训练框架或更新submodule。

沿用已经工作的Search-R1环境和检索服务。配置仍是`configs/local.json`，需要已有`agent_checkpoint/retriever_model/corpus_file/index_file/hotpotqa_file/two_wiki_file/retrieval_url`。不要求环境报告。模型使用历史7B PPO checkpoint，检索用E5-base-v2/Wiki18/Flat。厂内保持这些资源版本一致。

## 3. 冻结新240题，明确排除历史题

必须使用厂内真实旧样本文件：包含`sample_id/dataset/question`的canonical JSONL。可以直接用已有clean_rollouts.jsonl，它含有这些字段。**不要传只有ID的汇总或selection_manifest。** 程序按完整ID和归一化问题文本同时排除历史题。

```bash
bash scripts/run_si1.sh prepare \
  outputs/a1_clean_scale500/clean_rollouts.jsonl \
  outputs/samples/pilot.jsonl
```

上面是档案中的常见路径。若厂内保存名称不同，用真实路径替换；若还有其他看过结果的题集，把对应JSONL一起追加。程序只能证明提供的历史文件被排除，不能自动猜出遗漏的实验集合。没有历史文件时先找回/导出现有记录，不重跑旧实验制造排除表。

默认每数据集80题训练选择器、40题验证，共160+80。先固定ID再产生模型结果；第一轮启动点固定第一次观测后，一题一次。没有检索、或第一次观测后已到上下文上限而无生成机会的题，保留在全题分母，所有臂保持原样。`smoke.jsonl`为训练部分每数据集前10题，**不使用验证题做smoke**。

产物：`outputs/si1_samples/samples.jsonl`、`smoke.jsonl`、`selection_manifest.json`。目录已存在时拒绝覆盖。若资源/协议变更，为新版本设置新的SI1目录，不能删旧结果重新选成功题。

## 4. 服务检查与20题工程smoke

下面按单张空闲H800运行，TP=1；若0号正在做别的任务，替换成你安排的空闲卡。

```bash
bash scripts/run_si1.sh check
CUDA_VISIBLE_DEVICES=0 bash scripts/run_si1.sh smoke
```

check只验证输入和一次检索，不加载模型。smoke真实执行20题：完整baseline→全部题两次原样续跑与历史一致→全局控制通过→三个干预臂。任一基础设施/控制问题立即停止，不能把退出码0或目录存在当通过。成功后脚本自动做CPU审计，生成`outputs/si1_smoke/audit_receipt.json`。

必须看到run_manifest的`status=completed`、`all_original_controls_exact=true`、`all_planned_questions_completed=true`及receipt的`mechanical_checks_passed=true`。按工程通过决定是否跑全开发集，不按smoke修复多少题决定扩样，不根据结果改提示。smoke结果不训练选择器、不当独立确认数据。

## 5. 全240题开发实验

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_si1.sh dev
```

默认在`outputs/si1_dev`运行。dev要求已完成的smoke收据。smoke和dev的样本/协议/资源一致性也由Python执行器核验；改变了样本或资源不能沿用旧smoke放行。

全局四次成功搜索预算继承已用次数，第4次观测后仍可生成答案；再请求搜索记预算拒绝和任务失败。8192是上下文上限，512是每生成段上限，没有偷偷新增全程512 token预算。

| 臂 | 操作 | 预算与成本 |
| --- | --- | --- |
| keep | 原策略续跑 | 不加提示；原始控制必须吻合 |
| budget | 只告知已用/剩余/总预算 | 不改原总调用数，追加提示费用计入 |
| review | 固定一般回顾 | 允许原剩余预算内正常检索 |
| answer | 基于当前证据立即作答 | 不执行后续检索；若仍请求search，记`answer_only_search_refused`，不免费补答 |

答案错误、空答案、预算/格式失败都计入EM/F1。检索服务same-query可见文档漂移会停止实验。baseline、控制及每臂原始tokens、正文、评分和全成本都在厂内JSONL保留。原始数据不git add，不发到公开仓库。

首次不做四卡分片。20题实测后用吞吐估计240题费用；单张卡可满足7B开发探针。环境已经配置，不为本轮安装Slime/Megatron或重新建立veRL。没有完整RL或新确认集运行。

## 6. CPU学习与验证，一条命令

```bash
bash scripts/run_si1.sh analyze
```

该命令依次：

1. audit并加载已完成原始结果；只用train题拟合多输出ridge模型，分别预测gain、harm和相对调用成本。
2. `select`只读`states.jsonl`和冻结模型，不读cases/gold/验证分支结果，先写下选择。
3. `evaluate`再加载验证结果，核对模型、状态和决定的哈希及可重现性，计算实际所选臂表现。

无额外Python依赖、无GPU、无LLM-as-a-judge。64维固定词袋hash加可见状态数值只是便宜信号探针，不能把失败理解为所有语义选择器不可学。分数没有校准成安全概率。正式结果需要在新确认题上真正只执行被选臂；本轮是整题隔离的离线动作评价。

固定settings在`configs/selective_v1.json`：ridge=5、risk_weight=2、margin=.02、call_penalty=.01。对照risk_weight=1、相同动作比例随机打乱、固定简单规则，以及各固定动作。单一预算4、单一anchor1的首轮不能宣称跨预算/跨轮泛化。

主要文件：

- `metrics.json`：240题四臂完整转移与有限动作oracle诊断。
- `selector.json`：只由160训练题得到的模型及训练ID/配置证明。
- `decisions.json`：验证题prefix-only的预先决定。
- `validation_report.json`：80验证题实际gain/harm/net、EM/F1、费用、分数据集、随机/简单规则、风险权重消融。
- `audit_receipt.json`：原始分数/成本/控制/coverage及来源审计。

## 7. 如何解读，避免再绕四个成功案例调参

优先看验证selector相对**训练集冻结的最佳固定臂**的`net_em_vs_train_frozen_fixed`，以及每个数据集的方向、完整harm、实际输入token/搜索费用。验证上表现最佳的固定臂另列为保守诊断竞争者，不用它重拟合选择器。

`finite_action_oracle_em`是事后取四臂最好，绝不是部署恢复率。只有oracle提升而selector不提升时，说明当前可观察特征/学习器没有兑现机会，不能直接进入RL。若budget已解决主要差距，就按预算信息收益记录。若answer收益主要来自格式/终止，也应按该原因解释。

80题仅用于方向筛选，置信区间会宽。首轮希望看到数个百分点量级的可利用机会、验证选择相对强简单方法的净改善及可接受harm；无明确证据则按未确认记录，不挑子集、不重写提示后继续用这80题验证。进一步强基线适配和独立确认由真实报告决定。

## 8. 回传哪些结果

```bash
bash scripts/run_si1.sh pack
```

生成feedback tar.gz，仅包含汇总与run/audit provenance，不打包local.json、权重、数据集、原始JSONL或凭据。将该包反馈到聊天即可，厂内无需push。

如某阶段失败，保留目录，回传run_manifest中的error、最后几十行日志及失败阶段；不要反复跑同一路径覆盖结果。若失败发生在smoke，可只用现有打包工具指定smoke/run_manifest.json与已有receipt。后续修复依据工程错误，不依据哪题没修复。

目录覆盖可以通过环境变量配置：

```bash
export SI1_LOCAL=configs/local.json
export SI1_SAMPLES=outputs/si1_samples
export SI1_SMOKE=outputs/si1_smoke
export SI1_DEV=outputs/si1_dev
```

这里只是路径设置，不改变冻结数据与研究协议。

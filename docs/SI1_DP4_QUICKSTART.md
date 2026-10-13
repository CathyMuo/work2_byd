# 四卡与2000题版本：直接运行

最新结果已报告，当前按 [SI1 分析与 CPU 诊断](../report/SI1_DP4_PILOT_ANALYSIS_20261013.md)
继续；下面 GPU 命令为历史执行协议，不要求重跑。本次新增 `SI1_PYTHON` 支持：
可设为现有 Search-R1 环境解释器绝对路径，四卡 worker 会继承该解释器。

2026-10-12。用户要求扩大题量、必须四卡加速。默认1200题拟合+800题验证，HotpotQA/2Wiki各600训练+400验证。20题smoke也是四卡。推断保持每卡独立7B、TP=1，四个GPU worker同时启动；CPU汇总/选择器仍使用全部题的统一数据。

## 题量为何扩大

旧80题验证时，一题使EM变化1.25个百分点；新800题是一题0.125个百分点。若配对方法不一致率约10%，800题净EM差标准误约1.12个百分点，95%区间半宽约2.2个百分点，旧80题约6.9个百分点。新规模能支持更稳定的方向判断，但仍不保证检出1—2个百分点提升。

稳定损害取决于基线正确题的数量。例如正确率约30%时，800题约有240个基线正确题；零伤害的单侧95%近似上界仍约3/240=1.25%。正式小效应/低损害主张需独立确认，不能把四张卡或同题多臂当额外独立样本。

## 已有分支直接更新

```bash
cd /root/develop/workspace/work2_byd
git switch feat/selective-intervention
git pull --ff-only
python -m unittest discover -s tests -q
```

若尚未切换此分支，先按[完整指导](SI1_FACTORY_GUIDE.md)使用switch_si1.sh保留厂内旧提交。

## 运行顺序

```bash
export SI1_GPUS=0,1,2,3

bash scripts/run_si1.sh prepare \
  outputs/a1_clean_scale500/clean_rollouts.jsonl \
  outputs/samples/pilot.jsonl

bash scripts/run_si1.sh check
bash scripts/run_si1.sh smoke
bash scripts/run_si1.sh dev
bash scripts/run_si1.sh analyze
bash scripts/run_si1.sh pack
```

历史文件名不同就换成实际路径；更多已观察的题集一并追加。脚本自动追加已有`outputs/si1_samples/samples.jsonl`作为排除，但自定义旧题集路径须手动给出。已有旧240题目录保留，新默认路径为`outputs/si1_dp4_samples`、`si1_dp4_smoke`、`si1_dp4_dev`。不覆盖旧结果；旧单卡smoke不能放行新版。

## 四卡实际怎样运行

1. 按预先冻结的题目顺序轮转分片，开发阶段每卡500题；同题全部动作固定在同一卡，防止跨GPU续跑改变对照。
2. 四卡先并行生成baseline和两次原样续跑。全部控制精确一致、全题覆盖、跨卡相同query可见返回一致，才发布全局屏障。
3. 四卡再并行执行三个干预臂。任一卡失败终止本次协调器启动的其他worker，保留失败记录，不停止已有检索服务或其他训练作业。
4. 四卡全部完成后按原始题序合并。重复、缺失、额外记录或跨卡漂移拒绝汇总；原有raw score/cost/token/预算审计继续执行。

每阶段四张卡各加载一个模型；分阶段重新加载是全局门槛的固定开销。四卡共享原Retriever，实际加速受检索服务、磁盘和模型启动限制，不承诺固定4倍。`coordinator_status.json`记录实际墙钟时间；每30秒打印各卡进度。

## 查看状态

```bash
cat outputs/si1_dp4_dev/coordinator_status.json
tail -n 30 outputs/si1_dp4_dev/logs/controls_worker_0.log
tail -n 30 outputs/si1_dp4_dev/logs/actions_worker_3.log
```

必须保留`workers/worker_0`到`worker_3`、`parallel_plan.json`和`global_control_barrier.json`；后续审计依赖这些原始分片。`pack`只打包汇总及状态，不公开原始轨迹、权重、local.json或凭据。

本版由CPU合成测试验证调度、屏障、来源、故障停止与合并；开发服务器没有运行真实四卡H800实验。实际吞吐和GPU使用由厂内smoke确认。

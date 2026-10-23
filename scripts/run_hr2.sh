#!/usr/bin/env bash
# HR2: handoff replay re-registered on Search-R1 v0.3 (R = 7B PPO main, 7B GRPO second) vs Qwen2.5-7B-Instruct.
# Inference only; no downloads or training. See docs/HR2_FACTORY_GUIDE.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${HR_PYTHON:-${U1_PYTHON:-${AC_PYTHON:-python}}}
gpus=${HR_GPUS:-0,1,2,3}
protocol=configs/handoff_v2.json
fresh=${HR2_FRESH:-outputs/hr2_fresh}
if ! command -v "$py" >/dev/null 2>&1; then
  printf '%s\n' 'Python 不存在；请设置 HR_PYTHON 为已有 Search-R1 环境解释器。' >&2
  exit 1
fi
need_r() {
  if [[ "${1:-}" != "ppo" && "${1:-}" != "grpo" ]]; then
    printf '%s\n' '第二个参数必须是 ppo 或 grpo（选择 R 检查点）。' >&2; exit 1
  fi
  local_cfg=configs/local.hr2_$1.json
  [[ -f "$local_cfg" ]] || { printf '缺少 %s（local.json + instruct_checkpoint，agent_checkpoint 指向 v0.3 7B %s）\n' "$local_cfg" "$1" >&2; exit 1; }
}
case "$mode" in
  test)    "$py" -m unittest discover -s tests -q ;;
  freeze)  "$py" -m work2.handoff freeze --protocol "$protocol" --local "${HR2_FREEZE_LOCAL:-configs/local.hr2_ppo.json}" --per-dataset "${HR_PER_DATASET:-300}" --output "$fresh" "$@" ;;
  check)   need_r "${1:-}"; "$py" -m work2.handoff run --protocol "$protocol" --local "$local_cfg" --samples-dir "$fresh" --run-dir "outputs/hr2_smoke_$1" --gpus "$gpus" --smoke --check-only ;;
  smoke)
    need_r "${1:-}"; r=$1; shift
    "$py" -m work2.handoff run --protocol "$protocol" --local "$local_cfg" --samples-dir "$fresh" --run-dir "outputs/hr2_smoke_$r" --gpus "$gpus" --smoke "$@"
    [[ -d "outputs/hr2_smoke_$r/analysis" ]] || "$py" -m work2.handoff_analysis --run-dir "outputs/hr2_smoke_$r" --resamples 100
    ;;
  run)
    need_r "${1:-}"; r=$1; shift
    if [[ ! -f "outputs/hr2_smoke_$r/analysis/hr1_report.json" ]]; then printf '%s\n' "先完成 smoke $r。" >&2; exit 1; fi
    "$py" -m work2.handoff run --protocol "$protocol" --local "$local_cfg" --samples-dir "$fresh" --run-dir "outputs/hr2_$r" --gpus "$gpus" "$@"
    ;;
  analyze) need_r "${1:-}"; "$py" -m work2.handoff_analysis --run-dir "outputs/hr2_$1" ;;
  compare) "$py" -m work2.handoff_analysis --run-dir outputs/hr2_ppo --compare-with outputs/hr2_grpo ;;
  pack)
    args=(--include "$fresh/selection_manifest.json")
    for r in ppo grpo; do
      for f in "outputs/hr2_$r/run_manifest.json" "outputs/hr2_$r/analysis/hr1_report.json" "outputs/hr2_$r/analysis/hr1_report.md"; do
        [[ -f "$f" ]] && args+=(--include "$f")
      done
    done
    for f in outputs/hr2_ppo/compare/hr2_compare.json outputs/hr2_ppo/compare/hr2_compare.md; do [[ -f "$f" ]] && args+=(--include "$f"); done
    "$py" scripts/pack_feedback.py "${args[@]}"
    ;;
  *) printf '%s\n' 'HR2 modes: test | freeze | check ppo|grpo | smoke ppo|grpo | run ppo|grpo | analyze ppo|grpo | compare | pack'
     printf '%s\n' '环境变量：HR_PYTHON HR_GPUS HR2_FRESH HR_PER_DATASET HR2_FREEZE_LOCAL。详见 docs/HR2_FACTORY_GUIDE.md' ;;
esac

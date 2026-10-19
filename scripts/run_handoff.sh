#!/usr/bin/env bash
# HR1 handoff replay pilot (Qwen2.5-7B-Instruct <-> Search-R1 7B PPO). Inference only; no downloads or training.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${HR_PYTHON:-${AC_PYTHON:-${CT_PYTHON:-python}}}
local_config=${HR_LOCAL:-configs/local.handoff.json}   # local.json + "instruct_checkpoint"
gpus=${HR_GPUS:-0,1,2,3}
fresh=${HR_FRESH:-outputs/hr1_fresh}
smoke=${HR_SMOKE:-outputs/hr1_smoke}
run=${HR_RUN:-outputs/hr1_pilot}
if ! command -v "$py" >/dev/null 2>&1; then
  printf '%s\n' 'Python 不存在；请设置 HR_PYTHON 为已有 Search-R1 环境解释器。' >&2
  exit 1
fi
case "$mode" in
  test)    "$py" -m unittest discover -s tests -q ;;
  freeze)  "$py" -m work2.handoff freeze --local "$local_config" --per-dataset "${HR_PER_DATASET:-150}" --output "$fresh" "$@" ;;
  check)   "$py" -m work2.handoff run --local "$local_config" --samples-dir "$fresh" --run-dir "$smoke" --gpus "$gpus" --smoke --check-only ;;
  smoke)
    "$py" -m work2.handoff run --local "$local_config" --samples-dir "$fresh" --run-dir "$smoke" --gpus "$gpus" --smoke "$@"
    [[ -d "$smoke/analysis" ]] || "$py" -m work2.handoff_analysis --run-dir "$smoke" --resamples 100
    ;;
  run)
    if [[ ! -f "$smoke/analysis/hr1_report.json" ]]; then printf '%s\n' '先完成 smoke。' >&2; exit 1; fi
    "$py" -m work2.handoff run --local "$local_config" --samples-dir "$fresh" --run-dir "$run" --gpus "$gpus" "$@"
    ;;
  analyze) "$py" -m work2.handoff_analysis --run-dir "$run" "$@" ;;
  pack)
    "$py" scripts/pack_feedback.py --include "$fresh/selection_manifest.json" --include "$run/run_manifest.json" \
      --include "$run/analysis/hr1_report.json" --include "$run/analysis/hr1_report.md"
    ;;
  *) printf '%s\n' 'HR1 modes: test | freeze | check | smoke | run | analyze | pack'
     printf '%s\n' '环境变量：HR_PYTHON HR_LOCAL HR_GPUS HR_FRESH HR_SMOKE HR_RUN HR_PER_DATASET。详见 docs/HR1_FACTORY_GUIDE.md' ;;
esac
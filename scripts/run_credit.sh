#!/usr/bin/env bash
# CT1 interventional credit pilot. User-managed offline Search-R1 environment; no downloads or training.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${CT_PYTHON:-${PCR_PYTHON:-${SI1_PYTHON:-python}}}
local_config=${CT_LOCAL:-configs/local.json}
protocol=${CT_PROTOCOL:-configs/credit_v1.json}
sources=${CT_SOURCES:-outputs/si1_dp4_dev/sources.jsonl}
plan=${CT_PLAN:-outputs/ct1_plan}
smoke=${CT_SMOKE_DIR:-outputs/ct1_smoke}
run=${CT_RUN:-outputs/ct1_pilot}
gpus=${CT_GPUS:-0,1,2,3}
attn=${CT_ATTN:-eager}
if ! command -v "$py" >/dev/null 2>&1; then
  printf '%s\n' 'Python 不存在；请设置 CT_PYTHON 为已有 Search-R1 环境解释器。' >&2
  exit 1
fi
case "$mode" in
  test)
    "$py" -m unittest discover -s tests -q
    ;;
  prepare)
    "$py" -m work2.credit_testbed prepare --sources "$sources" --protocol "$protocol" --per-dataset "${CT_PER_DATASET:-150}" --output "$plan"
    ;;
  check)
    "$py" -m work2.credit_testbed run --plan-dir "$plan" --run-dir "$smoke" --protocol "$protocol" --local "$local_config" --gpus "$gpus" --smoke --check-only "$@"
    ;;
  smoke)
    "$py" -m work2.credit_testbed run --plan-dir "$plan" --run-dir "$smoke" --protocol "$protocol" --local "$local_config" --gpus "$gpus" --smoke "$@"
    "$py" -m work2.credit_signals run --run-dir "$smoke" --role policy --attn "$attn"
    "$py" -m work2.credit_analysis --run-dir "$smoke" --resamples 100
    ;;
  run)
    if [[ ! -f "$smoke/analysis/ct1_report.json" ]]; then
      printf '%s\n' '先完成 smoke（MC + policy 信号 + 分析）。' >&2
      exit 1
    fi
    "$py" -m work2.credit_testbed run --plan-dir "$plan" --run-dir "$run" --protocol "$protocol" --local "$local_config" --gpus "$gpus" "$@"
    ;;
  signals)
    "$py" -m work2.credit_signals run --run-dir "$run" --role policy --attn "$attn"
    ;;
  signals-ref)
    "$py" -m work2.credit_signals run --run-dir "$run" --role reference --attn "$attn"
    ;;
  analyze)
    "$py" -m work2.credit_analysis --run-dir "$run" "$@"
    ;;
  pack)
    "$py" scripts/pack_feedback.py \
      --include "$plan/plan.json" \
      --include "$run/run_manifest.json" \
      --include "$run/run_plan.json" \
      --include "$run/analysis/ct1_report.json" \
      --include "$run/analysis/ct1_report.md" \
      --include "$run/analysis/ct1_turns.json"
    ;;
  *) printf '%s\n' 'CT1 modes: test | prepare | check | smoke | run | signals | signals-ref | analyze | pack'
     printf '%s\n' '环境变量：CT_PYTHON CT_SOURCES CT_PER_DATASET CT_GPUS CT_PLAN CT_SMOKE_DIR CT_RUN CT_ATTN。详见 docs/CT1_FACTORY_GUIDE.md' ;;
esac

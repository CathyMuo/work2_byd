#!/usr/bin/env bash
# PCR v1 (path luck). User-managed offline Search-R1 environment; no downloads, installs or training.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${PCR_PYTHON:-${SI1_PYTHON:-python}}
local_config=${PCR_LOCAL:-configs/local.json}
protocol=${PCR_PROTOCOL:-configs/pcr_v1.json}
sources=${PCR_SOURCES:-outputs/si1_dp4_dev/sources.jsonl}
plan=${PCR_PLAN:-outputs/pcr_plan}
smoke=${PCR_SMOKE_DIR:-outputs/pcr_smoke}
run=${PCR_RUN:-outputs/pcr_exp1}
gpus=${PCR_GPUS:-0,1,2,3}
per_dataset=${PCR_PER_DATASET:-300}
if ! command -v "$py" >/dev/null 2>&1; then
  printf '%s\n' 'Python 不存在；请设置 PCR_PYTHON 为已有 Search-R1 环境解释器。' >&2
  exit 1
fi
case "$mode" in
  test)
    "$py" -m unittest discover -s tests -q
    ;;
  phase0)
    a1=${PCR_A1:-outputs/a1_clean_scale500/clean_rollouts.jsonl}
    a3=${PCR_A3:-outputs/a3_scale500_merged/paired_rollouts.jsonl}
    "$py" -m work2.audit_pairs --a3-rollouts "$a3" --schema-only
    "$py" -m work2.phase0_luck --a1-rollouts "$a1" --a3-rollouts "$a3" --output-dir "${PCR_PHASE0_OUT:-outputs/pcr_phase0}" "$@"
    ;;
  prepare)
    "$py" -m work2.pcr prepare --sources "$sources" --protocol "$protocol" --per-dataset "$per_dataset" --output "$plan"
    ;;
  check)
    "$py" -m work2.pcr run --plan-dir "$plan" --run-dir "$smoke" --protocol "$protocol" --local "$local_config" --gpus "$gpus" --smoke --check-only "$@"
    ;;
  smoke)
    "$py" -m work2.pcr run --plan-dir "$plan" --run-dir "$smoke" --protocol "$protocol" --local "$local_config" --gpus "$gpus" --smoke "$@"
    "$py" -m work2.pcr_analysis --run-dir "$smoke" --resamples 200
    ;;
  run)
    if [[ ! -f "$smoke/run_manifest.json" ]] || ! grep -q '"all_controls_exact": true' "$smoke/run_manifest.json"; then
      printf '%s\n' '先完成同一 plan 的 smoke，且 E0/E0 全部精确。' >&2
      exit 1
    fi
    "$py" -m work2.pcr run --plan-dir "$plan" --run-dir "$run" --protocol "$protocol" --local "$local_config" --gpus "$gpus" "$@"
    ;;
  resume)
    "$py" -m work2.pcr run --plan-dir "$plan" --run-dir "$run" --protocol "$protocol" --local "$local_config" --gpus "$gpus" --resume "$@"
    ;;
  analyze)
    si1_dir=$(dirname "$sources")
    extra=()
    if [[ -f "$si1_dir/cases.jsonl" ]]; then extra=(--si1-run-dir "$si1_dir"); fi
    "$py" -m work2.pcr_analysis --run-dir "$run" ${extra[@]+"${extra[@]}"} "$@"
    ;;
  pack)
    "$py" scripts/pack_feedback.py \
      --include "$plan/plan.json" \
      --include "$run/run_manifest.json" \
      --include "$run/run_plan.json" \
      --include "$run/analysis/pcr_report.json" \
      --include "$run/analysis/pcr_report.md" \
      --include "$run/analysis/flip_cases.jsonl"
    ;;
  *) printf '%s\n' 'PCR modes: test | phase0 | prepare | check | smoke | run | resume | analyze | pack'
     printf '%s\n' '环境变量：PCR_PYTHON PCR_SOURCES PCR_PER_DATASET PCR_GPUS PCR_PLAN PCR_SMOKE_DIR PCR_RUN PCR_A1 PCR_A3。详见 docs/PCR_FACTORY_GUIDE.md' ;;
esac

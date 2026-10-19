#!/usr/bin/env bash
# AC1 confirmation of the interventional-audit claims on fresh questions, per checkpoint.
# User-managed offline Search-R1 environment; no downloads, installs or training.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${AC_PYTHON:-${CT_PYTHON:-${PCR_PYTHON:-python}}}
label=${AC_LABEL:-7b_ppo}                 # checkpoint label, e.g. 7b_ppo, 3b_ppo, 7b_grpo
local_config=${AC_LOCAL:-configs/local.json}   # its agent_checkpoint is the checkpoint under test
gpus=${AC_GPUS:-0,1,2,3}
fresh=${AC_FRESH:-outputs/ac1_fresh}
root=outputs/ac1_${label}
if ! command -v "$py" >/dev/null 2>&1; then
  printf '%s\n' 'Python 不存在；请设置 AC_PYTHON 为已有 Search-R1 环境解释器。' >&2
  exit 1
fi
step() { printf '\n==== [%s] %s ====\n' "$label" "$1"; }
case "$mode" in
  test)
    "$py" -m unittest discover -s tests -q
    ;;
  freeze)          # once for all checkpoints
    "$py" -m work2.audit_confirm freeze --local "$local_config" --per-dataset "${AC_PER_DATASET:-500}" --output "$fresh" "$@"
    ;;
  check)
    "$py" -m work2.audit_confirm sources --local "$local_config" --samples-dir "$fresh" --run-dir "$root/sources" --gpus "$gpus" --check-only
    ;;
  sources)
    step "E0 sources"
    if [[ -f "$root/sources/sources.jsonl" ]]; then echo "sources already merged; skip"; exit 0; fi
    "$py" -m work2.audit_confirm sources --local "$local_config" --samples-dir "$fresh" --run-dir "$root/sources" --gpus "$gpus" "$@"
    ;;
  pcr)
    step "PCR plan"
    [[ -f "$root/pcr_plan/plan.json" ]] || "$py" -m work2.pcr prepare --sources "$root/sources/sources.jsonl" --protocol configs/pcr_audit_v1.json --per-dataset 0 --output "$root/pcr_plan"
    step "PCR smoke"
    [[ -f "$root/pcr_smoke/states.jsonl" ]] || "$py" -m work2.pcr run --plan-dir "$root/pcr_plan" --run-dir "$root/pcr_smoke" --protocol configs/pcr_audit_v1.json --local "$local_config" --gpus "$gpus" --smoke
    step "PCR full"
    if [[ -d "$root/pcr" ]]; then extra=(--resume); else extra=(); fi
    "$py" -m work2.pcr run --plan-dir "$root/pcr_plan" --run-dir "$root/pcr" --protocol configs/pcr_audit_v1.json --local "$local_config" --gpus "$gpus" ${extra[@]+"${extra[@]}"} "$@"
    step "PCR analysis"
    [[ -d "$root/pcr/analysis" ]] || "$py" -m work2.pcr_analysis --run-dir "$root/pcr"
    ;;
  ct)
    step "CT plan"
    [[ -f "$root/ct_plan/plan.json" ]] || "$py" -m work2.credit_testbed prepare --sources "$fresh/samples.jsonl" --protocol configs/credit_v1_k16.json --per-dataset "${AC_CT_PER_DATASET:-250}" --output "$root/ct_plan"
    step "CT MC (k=16)"
    "$py" -m work2.credit_testbed run --plan-dir "$root/ct_plan" --run-dir "$root/ct" --protocol configs/credit_v1_k16.json --local "$local_config" --gpus "$gpus" "$@"
    step "CT policy signals"
    [[ -f "$root/ct/signals_policy.jsonl" ]] || "$py" -m work2.credit_signals run --run-dir "$root/ct" --role policy --attn eager
    if [[ "${AC_WITH_REF:-0}" == 1 && ! -f "$root/ct/signals_reference.jsonl" ]]; then
      step "CT reference signals"; "$py" -m work2.credit_signals run --run-dir "$root/ct" --role reference --attn eager
    fi
    step "CT analysis"
    [[ -d "$root/ct/analysis" ]] || "$py" -m work2.credit_analysis --run-dir "$root/ct"
    ;;
  report)
    step "AC1 verdicts"
    "$py" -m work2.audit_confirm report --pcr-report "$root/pcr/analysis/pcr_report.json" --ct1-report "$root/ct/analysis/ct1_report.json" --label "$label" --output "$root/ac1_verdicts.json"
    ;;
  all)
    bash "$0" sources; bash "$0" pcr; bash "$0" ct; bash "$0" report
    ;;
  pack)
    "$py" scripts/pack_feedback.py \
      --include "$fresh/selection_manifest.json" \
      --include "$root/sources/run_manifest.json" \
      --include "$root/pcr/run_manifest.json" \
      --include "$root/pcr/analysis/pcr_report.json" \
      --include "$root/pcr/analysis/pcr_report.md" \
      --include "$root/ct/run_manifest.json" \
      --include "$root/ct/analysis/ct1_report.json" \
      --include "$root/ct/analysis/ct1_report.md" \
      --include "$root/ac1_verdicts.json" \
      --include "$root/ac1_verdicts.md"
    ;;
  *) printf '%s\n' 'AC1 modes: test | freeze | check | sources | pcr | ct | report | all | pack'
     printf '%s\n' '环境变量：AC_PYTHON AC_LABEL AC_LOCAL AC_GPUS AC_PER_DATASET AC_CT_PER_DATASET AC_WITH_REF。详见 docs/AC1_FACTORY_GUIDE.md' ;;
esac

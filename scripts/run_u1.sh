#!/usr/bin/env bash
# U1: H1/H3 audit across the agent roster and five datasets. Inference only; no training.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${U1_PYTHON:-${AC_PYTHON:-python}}
gpus=${U1_GPUS:-0,1,2,3}
fresh=${U1_FRESH:-outputs/u1_fresh}
root=outputs/u1
need_prompt() {
  if [[ "${U1_PROMPT:-}" != "bounded" && "${U1_PROMPT:-}" != "official" ]]; then
    printf '%s\n' '请先按 FC1 的决定设置 U1_PROMPT=bounded 或 official（见 docs/U1_FACTORY_GUIDE.md）。' >&2; exit 1
  fi
}
if ! command -v "$py" >/dev/null 2>&1; then printf '%s\n' 'Python 不存在；请设置 U1_PYTHON。' >&2; exit 1; fi
step() { printf '\n==== [%s] %s ====\n' "$1" "$2"; }
case "$mode" in
  test)   "$py" -m unittest discover -s tests -q ;;
  freeze) "$py" -m work2.u1 freeze --output "$fresh" "$@" ;;
  check)  "$py" -m work2.u1 check ;;
  agent)  # agent LABEL
    need_prompt
    label=$1; d=$root/$label; local_cfg=configs/local.u1_${label}.json; v=$U1_PROMPT
    [[ -f "$local_cfg" ]] || { echo "missing $local_cfg"; exit 1; }
    step "$label" "E0 sources ($v prompt)"
    [[ -f "$d/sources/sources.jsonl" ]] || "$py" -m work2.audit_confirm sources --protocol "configs/fidelity_${v}.json" --local "$local_cfg" --samples-dir "$fresh" --run-dir "$d/sources" --gpus "$gpus"
    step "$label" "PCR (H1 arms)"
    [[ -f "$d/pcr_plan/plan.json" ]] || "$py" -m work2.pcr prepare --sources "$d/sources/sources.jsonl" --protocol "configs/pcr_u1_${v}.json" --per-dataset "${U1_PCR_PER_DATASET:-400}" --output "$d/pcr_plan"
    if [[ ! -f "$d/pcr/states.jsonl" ]]; then
      if [[ -d "$d/pcr" ]]; then extra=(--resume); else extra=(); fi
      "$py" -m work2.pcr run --plan-dir "$d/pcr_plan" --run-dir "$d/pcr" --protocol "configs/pcr_u1_${v}.json" --local "$local_cfg" --gpus "$gpus" ${extra[@]+"${extra[@]}"}
    fi
    [[ -d "$d/pcr/analysis" ]] || "$py" -m work2.pcr_analysis --run-dir "$d/pcr" --resamples 1000
    step "$label" "CT (H3, k=16)"
    [[ -f "$d/ct_plan/plan.json" ]] || "$py" -m work2.credit_testbed prepare --sources "$fresh/samples.jsonl" --protocol "configs/credit_u1_${v}.json" --per-dataset "${U1_CT_PER_DATASET:-150}" --output "$d/ct_plan"
    [[ -f "$d/ct/records.jsonl" ]] || "$py" -m work2.credit_testbed run --plan-dir "$d/ct_plan" --run-dir "$d/ct" --protocol "configs/credit_u1_${v}.json" --local "$local_cfg" --gpus "$gpus"
    [[ -f "$d/ct/signals_policy.jsonl" ]] || "$py" -m work2.credit_signals run --run-dir "$d/ct" --role policy --attn eager
    [[ -d "$d/ct/analysis" ]] || "$py" -m work2.credit_analysis --run-dir "$d/ct" --resamples 1000
    step "$label" "verdicts"
    [[ -f "$d/verdicts.json" ]] || "$py" -m work2.audit_confirm report --pcr-report "$d/pcr/analysis/pcr_report.json" --ct1-report "$d/ct/analysis/ct1_report.json" --label "$label" --output "$d/verdicts.json"
    ;;
  all)
    need_prompt
    labels=$("$py" -c "import json;print(' '.join(a['label'] for a in json.load(open('configs/u1_agents.json'))['agents']))")
    for label in $labels; do
      if [[ -f "configs/local.u1_${label}.json" ]]; then bash "$0" agent "$label"; else echo "skip $label (no configs/local.u1_${label}.json)"; fi
    done ;;
  summary) "$py" -m work2.u1 summary --output "$root/u1_summary.json" "$@" ;;
  pack)
    args=(--include "$fresh/selection_manifest.json" --include "$root/u1_summary.json" --include "$root/u1_summary.md")
    for f in $root/*/verdicts.json $root/*/verdicts.md; do [[ -f "$f" ]] && args+=(--include "$f"); done
    "$py" scripts/pack_feedback.py "${args[@]}" ;;
  *) printf '%s\n' 'U1 modes: test | freeze | check | agent LABEL | all | summary | pack'
     printf '%s\n' '环境变量：U1_PROMPT（必填，由 FC1 决定） U1_PYTHON U1_GPUS U1_PCR_PER_DATASET U1_CT_PER_DATASET。详见 docs/U1_FACTORY_GUIDE.md' ;;
esac
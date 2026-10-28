#!/usr/bin/env bash
# U1L: H1 only (sources + PCR h1 arms) for larger agents (ReSearch 7B/32B, prompted Qwen2.5-Instruct 7B/32B). Inference only.
# See docs/U1L_FACTORY_GUIDE.md. 32B weights need U1_GPU_MEM ~0.92 on 80 GB cards (default below).
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${U1_PYTHON:-${AC_PYTHON:-python}}
gpus=${U1_GPUS:-0,1,2,3}
fresh=${U1_FRESH:-outputs/u1_fresh}
roster=configs/u1l_agents.json
mem=${U1_GPU_MEM:-0.92}
root=outputs/u1
if ! command -v "$py" >/dev/null 2>&1; then printf '%s\n' 'Python 不存在；请设置 U1_PYTHON。' >&2; exit 1; fi
step() { printf '\n==== [%s] %s ====\n' "$1" "$2"; }
variant_of() { "$py" -c "import json,sys;print({a['label']:a['prompt_variant'] for a in json.load(open('$roster'))['agents']}[sys.argv[1]])" "$1"; }
case "$mode" in
  test)   "$py" -m unittest discover -s tests -q ;;
  check)  "$py" -m work2.u1 check --roster "$roster" ;;
  agent)  # agent LABEL
    label=${1:?缺少智能体标签}; v=$(variant_of "$label"); d=$root/$label; local_cfg=configs/local.u1_${label}.json
    [[ -f "$local_cfg" ]] || { echo "missing $local_cfg"; exit 1; }
    step "$label" "E0 sources ($v prompt)"
    [[ -f "$d/sources/sources.jsonl" ]] || "$py" -m work2.audit_confirm sources --protocol "configs/fidelity_${v}.json" --local "$local_cfg" --samples-dir "$fresh" --run-dir "$d/sources" --gpus "$gpus" --gpu-memory-utilization "$mem"
    step "$label" "PCR (H1 arms)"
    [[ -f "$d/pcr_plan/plan.json" ]] || "$py" -m work2.pcr prepare --sources "$d/sources/sources.jsonl" --protocol "configs/pcr_u1_${v}.json" --per-dataset "${U1_PCR_PER_DATASET:-400}" --output "$d/pcr_plan"
    if [[ ! -f "$d/pcr/states.jsonl" ]]; then
      if [[ -d "$d/pcr" ]]; then extra=(--resume); else extra=(); fi
      "$py" -m work2.pcr run --plan-dir "$d/pcr_plan" --run-dir "$d/pcr" --protocol "configs/pcr_u1_${v}.json" --local "$local_cfg" --gpus "$gpus" --gpu-memory-utilization "$mem" ${extra[@]+"${extra[@]}"}
    fi
    [[ -d "$d/pcr/analysis" ]] || "$py" -m work2.pcr_analysis --run-dir "$d/pcr" --resamples 1000
    ;;
  all)
    for label in $("$py" -c "import json;print(' '.join(a['label'] for a in json.load(open('$roster'))['agents']))"); do
      if [[ -f "configs/local.u1_${label}.json" ]]; then bash "$0" agent "$label"; else echo "skip $label (no configs/local.u1_${label}.json)"; fi
    done ;;
  summary)
    "$py" -m work2.u1x --root "$root" --roster "$roster" --output outputs/u1l_fidelity.json
    "$py" -m work2.u1 summary --roster "$roster" --root "$root" --output outputs/u1l_summary.json
    "$py" -m work2.arm_diagnose --agents configs/u1_agents.json configs/u1x_agents.json "$roster" --output outputs/paper/arm_diagnose_u1l.json
    "$py" -m work2.paper_h1 --u1-root "$root" --agents configs/u1_agents.json configs/u1x_agents.json "$roster" --validity outputs/paper/arm_diagnose_u1l.json --exclude qwen25_32b_inst qwen25_7b_inst --output outputs/paper/h1_pooled_u1l
    "$py" -m work2.paper_h1 --u1-root "$root" --agents "$roster" --validity outputs/paper/arm_diagnose_u1l.json --output outputs/paper/h1_u1l_only ;;
  pack)
    args=(--include outputs/u1l_fidelity.md --include outputs/u1l_summary.md --include outputs/paper/arm_diagnose_u1l.md --include outputs/paper/h1_pooled_u1l/h1_pooled.md --include outputs/paper/h1_u1l_only/h1_pooled.md)
    "$py" scripts/pack_feedback.py "${args[@]}" ;;
  *) printf '%s\n' 'U1L modes: test | check | agent LABEL | all | summary | pack'
     printf '%s\n' '环境变量：U1_PYTHON U1_GPUS U1_FRESH U1_PCR_PER_DATASET U1_GPU_MEM。详见 docs/U1L_FACTORY_GUIDE.md' ;;
esac

#!/usr/bin/env bash
# AS2: filler variants and graded positive controls (as2, 14 agents), sampled decoding (as2s, 4 Search-R1 agents). Inference only.
# Protocol: research/AS2_PROTOCOL.md. Guide: docs/AS2_FACTORY_GUIDE.md. States are the AS1 states (same selection and seed),
# reusing each agent's H1 sources; retrieval must be the same 4-card deployment those sources used.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${U1_PYTHON:-${AC_PYTHON:-python}}
root=outputs/u1
agents="v03_3b_ppo v03_3b_grpo v01_7b_ppo v03_7b_ppo v03_7b_grpo v03_7b_it_grpo zs_llama32_3b zs_qwen25_7b stepsearch_7b v03_14b_ppo v03_14b_it_grpo research_7b v03_32b_grpo research_32b"
sampled="v03_3b_ppo v03_7b_grpo v03_14b_ppo v03_32b_grpo"
if ! command -v "$py" >/dev/null 2>&1; then printf '%s\n' 'Python 不存在；请设置 U1_PYTHON。' >&2; exit 1; fi
step() { printf '\n==== [%s] %s ====\n' "$1" "$2"; }
variant_of() {  # U1 agents use the bounded prompt (FC1 decision, 2026-10-23); the others carry their own
  "$py" -c "import json,sys
v={a['label']:a['prompt_variant'] for f in ('configs/u1x_agents.json','configs/u1l_agents.json') for a in json.load(open(f))['agents']}
print(v.get(sys.argv[1],'bounded'))" "$1"
}
resources() {  # same placement as AS1: 32B on GPUs 1-3 at 0.85 beside the 4-card retrieval; others on all four at 0.75
  case "$1" in
    *32b*) gpus=${AS_GPUS_32B:-1,2,3}; mem=${AS_GPU_MEM_32B:-0.85} ;;
    *)     gpus=${AS_GPUS:-0,1,2,3};   mem=${AS_GPU_MEM:-0.75} ;;
  esac
}
run_one() {  # run_one LABEL KIND(as2|as2s) [--smoke]
  label=$1; kind=$2; smoke=${3:-}; v=$(variant_of "$label"); d=$root/$label; local_cfg=configs/local.u1_${label}.json
  [[ -f "$local_cfg" ]] || { echo "missing $local_cfg"; exit 1; }
  [[ -f "$d/sources/sources.jsonl" && -f "$d/pcr/states.jsonl" ]] || { echo "missing H1 sources or states for $label"; exit 1; }
  [[ -f "configs/${kind}_${v}.json" ]] || { echo "no configs/${kind}_${v}.json ($label is not in this experiment)"; exit 1; }
  resources "$label"
  step "$label" "$kind ($v prompt, GPUs $gpus, memory $mem)"
  if [[ ! -f "$d/${kind}_plan/plan.json" ]]; then
    # Same selection and seed as AS1, so the states coincide with outputs/u1/<label>/as1.
    "$py" -m work2.pcr prepare --sources "$d/sources/sources.jsonl" --protocol "configs/${kind}_${v}.json" --select-from "$d/pcr" \
      --stable-sample "${AS1_STABLE:-100}" --wrong-sample "${AS1_WRONG:-100}" --output "$d/${kind}_plan"
  fi
  out=$d/$kind; extra=()
  if [[ "$smoke" == --smoke ]]; then out=$d/${kind}_smoke; extra=(--smoke); fi
  [[ -f "$out/states.jsonl" ]] && return 0
  [[ -d "$out" ]] && extra+=(--resume)
  "$py" -m work2.pcr run --plan-dir "$d/${kind}_plan" --run-dir "$out" --protocol "configs/${kind}_${v}.json" --local "$local_cfg" \
    --gpus "$gpus" --gpu-memory-utilization "$mem" --on-control-mismatch record ${extra[@]+"${extra[@]}"}
}
case "$mode" in
  test)  "$py" -m unittest discover -s tests -q ;;
  smoke) label=${1:?缺少智能体标签}; run_one "$label" as2 --smoke
         case " $sampled " in *" $label "*) run_one "$label" as2s --smoke ;; esac ;;
  as2)   run_one "${1:?缺少智能体标签}" as2 ;;
  as2s)  run_one "${1:?缺少智能体标签}" as2s ;;
  all)   for a in $agents; do run_one "$a" as2; done
         for a in $sampled; do run_one "$a" as2s; done ;;
  summary)
    "$py" -m work2.as2 --u1-root "$root" --output "outputs/paper/as2" ;;
  *) printf '%s\n' 'AS2 modes: test | smoke LABEL | as2 LABEL | as2s LABEL | all | summary'
     printf '%s\n' '环境变量：U1_PYTHON AS_GPUS AS_GPU_MEM AS_GPUS_32B AS_GPU_MEM_32B。详见 docs/AS2_FACTORY_GUIDE.md' ;;
esac

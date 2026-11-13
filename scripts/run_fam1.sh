#!/usr/bin/env bash
# FAM1: do the main results carry over to non-Qwen (Llama) agents? Inference only. Protocol: research/FAM1_PROTOCOL.md.
# Guide: docs/FAM1_FACTORY_GUIDE.md. Retrieval must be the same 4-card deployment the main agents used (never moved to free
# memory). Every run freezes the repository commit: do not commit or pull while a run is going.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${U1_PYTHON:-${AC_PYTHON:-python}}
gpus=${FAM1_GPUS:-0,1,2,3}
mem=${FAM1_GPU_MEM:-0.75}
fresh=${U1_FRESH:-outputs/u1_fresh}
pool_e=${FAM1_E:-outputs/fam1_e}
roster=configs/fam1_agents.json
root=outputs/u1
# Section 4.2 order: the four Search-R1 Llama agents, the four Qwen controls, then R1-Searcher.
order="v01_l3b_ppo v01_l3b_grpo v01_l3b_it_ppo v01_l3b_it_grpo v01_q3b_ppo v01_q3b_grpo v01_q3b_it_ppo v01_q3b_it_grpo r1s_l8b"
# Pool E (section 4.4): every non-Qwen agent (zs_llama32_3b included) and the four Qwen controls; sources + H1 only.
order_e="v01_l3b_ppo v01_l3b_grpo v01_l3b_it_ppo v01_l3b_it_grpo r1s_l8b zs_llama32_3b v01_q3b_ppo v01_q3b_grpo v01_q3b_it_ppo v01_q3b_it_grpo"
if ! command -v "$py" >/dev/null 2>&1; then printf '%s\n' 'Python 不存在；请设置 U1_PYTHON。' >&2; exit 1; fi
step() { printf '\n==== [%s] %s ====\n' "$1" "$2"; }
variant_of() {  # FAM1 roster first; zs_llama32_3b keeps its U1x prompt
  "$py" -c "import json,sys
v={a['label']:a['prompt_variant'] for f in ('$roster','configs/u1x_agents.json') for a in json.load(open(f))['agents']}
print(v[sys.argv[1]])" "$1"
}
need_local() { [[ -f "configs/local.u1_$1.json" ]] || { echo "missing configs/local.u1_$1.json"; exit 1; }; }
pcr_run() {  # pcr_run LABEL PLAN RUN PROTOCOL [extra...]
  local label=$1 plan=$2 run=$3 protocol=$4; shift 4
  [[ -f "$run/states.jsonl" ]] && return 0
  local extra=("$@"); [[ -d "$run" ]] && extra+=(--resume)
  "$py" -m work2.pcr run --plan-dir "$plan" --run-dir "$run" --protocol "$protocol" --local "configs/local.u1_${label}.json" \
    --gpus "$gpus" --gpu-memory-utilization "$mem" ${extra[@]+"${extra[@]}"}
}
sources() {  # sources LABEL SAMPLES_DIR RUN_DIR
  [[ -f "$3/sources.jsonl" ]] && return 0
  "$py" -m work2.audit_confirm sources --protocol "configs/fidelity_$(variant_of "$1").json" --local "configs/local.u1_$1.json" \
    --samples-dir "$2" --run-dir "$3" --gpus "$gpus" --gpu-memory-utilization "$mem"
}
agent() {  # section 4.1, steps 1-5, on the frozen questions
  local label=$1 v d; v=$(variant_of "$label"); d=$root/$label; need_local "$label"
  step "$label" "1 sources ($v prompt)"; sources "$label" "$fresh" "$d/sources"
  step "$label" "3 H1"
  [[ -f "$d/pcr_plan/plan.json" ]] || "$py" -m work2.pcr prepare --sources "$d/sources/sources.jsonl" --protocol "configs/pcr_u1_${v}.json" \
    --per-dataset 400 --output "$d/pcr_plan"
  pcr_run "$label" "$d/pcr_plan" "$d/pcr" "configs/pcr_u1_${v}.json" --on-control-mismatch record
  step "$label" "2 insertion validity"
  "$py" -c "import json,sys;from work2.arm_diagnose import diagnose;d=diagnose(sys.argv[1]);print('insertion_valid =',d['insertion_valid'],{a:round(v['answered'],3) for a,v in d['arms'].items() if 'neutral' in a})" "$d/pcr"
  step "$label" "4 AS1"
  [[ -f "$d/as1_plan/plan.json" ]] || "$py" -m work2.pcr prepare --sources "$d/sources/sources.jsonl" --protocol "configs/as1_${v}.json" \
    --select-from "$d/pcr" --stable-sample 100 --wrong-sample 100 --output "$d/as1_plan"
  pcr_run "$label" "$d/as1_plan" "$d/as1" "configs/as1_${v}.json" --on-control-mismatch record
  step "$label" "5 U1N"
  [[ -f "$d/u1n_plan/plan.json" ]] || "$py" -m work2.pcr prepare --sources "$d/sources/sources.jsonl" --protocol "configs/u1n_${v}.json" \
    --per-dataset 150 --output "$d/u1n_plan"
  pcr_run "$label" "$d/u1n_plan" "$d/u1n" "configs/u1n_${v}.json" --on-control-mismatch record
}
agent_e() {  # pool E: steps 1 and 3 only
  local label=$1 v d; v=$(variant_of "$label"); d=$root/$label; need_local "$label"
  [[ -f "$pool_e/samples.jsonl" ]] || { echo "missing $pool_e; run: bash scripts/run_fam1.sh freeze-e configs/local.<file used for the U1 freeze>.json"; exit 1; }
  step "$label" "E 1 sources ($v prompt)"; sources "$label" "$pool_e" "$d/sources_e"
  step "$label" "E 3 H1"
  [[ -f "$d/pcr_e_plan/plan.json" ]] || "$py" -m work2.pcr prepare --sources "$d/sources_e/sources.jsonl" --protocol "configs/pcr_u1_${v}.json" \
    --per-dataset 400 --output "$d/pcr_e_plan"
  pcr_run "$label" "$d/pcr_e_plan" "$d/pcr_e" "configs/pcr_u1_${v}.json" --on-control-mismatch record
}
case "$mode" in
  test)     "$py" -m unittest discover -s tests -q ;;
  check)    "$py" -m work2.u1 check --roster "$roster" ;;
  freeze-e) "$py" -m work2.fam1 freeze-e --local "${1:?缺少 local 文件（U1 冻结题目时用的那个，含 hotpotqa_file / two_wiki_file / nq_file / musique_file）}" --output "$pool_e" ;;
  smoke)    # smoke LABEL: sources on the frozen questions must exist; one H1 smoke run (a few states, no outputs kept)
    label=${1:?缺少智能体标签}; v=$(variant_of "$label"); d=$root/$label; need_local "$label"
    sources "$label" "$fresh" "$d/sources"
    [[ -f "$d/pcr_plan/plan.json" ]] || "$py" -m work2.pcr prepare --sources "$d/sources/sources.jsonl" --protocol "configs/pcr_u1_${v}.json" \
      --per-dataset 400 --output "$d/pcr_plan"
    "$py" -m work2.pcr run --plan-dir "$d/pcr_plan" --run-dir "$d/pcr_smoke" --protocol "configs/pcr_u1_${v}.json" \
      --local "configs/local.u1_${label}.json" --gpus "$gpus" --gpu-memory-utilization "$mem" --smoke ;;
  agent)    agent "${1:?缺少智能体标签}" ;;
  e)        agent_e "${1:?缺少智能体标签}" ;;
  all)      for a in $order; do if [[ -f "configs/local.u1_${a}.json" ]]; then agent "$a"; else echo "skip $a (no configs/local.u1_${a}.json)"; fi; done
            if [[ -f "$pool_e/samples.jsonl" ]]; then
              for a in $order_e; do if [[ -f "configs/local.u1_${a}.json" ]]; then agent_e "$a"; else echo "skip E $a"; fi; done
            else echo "pool E not frozen ($pool_e); skipped"; fi ;;
  summary)  "$py" -m work2.fam1 fidelity --output outputs/fam1_fidelity.json
            "$py" -m work2.fam1 analyze --output "${1:-outputs/paper/fam1}" ;;
  pack)     "$py" scripts/pack_feedback.py --include outputs/fam1_fidelity.json --include "${1:-outputs/paper/fam1}/fam1.md" \
              --include "${1:-outputs/paper/fam1}/fam1.json" --include "$pool_e/selection_manifest.json" ;;
  *) printf '%s\n' 'FAM1 modes: test | check | freeze-e LOCAL | smoke LABEL | agent LABEL | e LABEL | all | summary [OUT] | pack [OUT]'
     printf '%s\n' '环境变量：U1_PYTHON FAM1_GPUS FAM1_GPU_MEM U1_FRESH FAM1_E。详见 docs/FAM1_FACTORY_GUIDE.md' ;;
esac

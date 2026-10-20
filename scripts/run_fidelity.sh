#!/usr/bin/env bash
# FC1 harness fidelity: greedy EM per checkpoint x prompt variant on the AC1 fresh questions. Inference only.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${FC_PYTHON:-${AC_PYTHON:-python}}
gpus=${FC_GPUS:-0,1,2,3}
fresh=${FC_FRESH:-outputs/ac1_fresh}
root=outputs/fc1
# label -> local config (each config's agent_checkpoint is that model). Labels must not contain "-".
declare -A LOCAL=(
  [v01_7b_ppo]=${FC_LOCAL_V01_7B_PPO:-configs/local.json}
  [v03_3b_ppo]=${FC_LOCAL_V03_3B_PPO:-configs/local.second.json}
  [v03_7b_ppo]=${FC_LOCAL_V03_7B_PPO:-configs/local.fc_v03_7b_ppo.json}
  [v03_7b_grpo]=${FC_LOCAL_V03_7B_GRPO:-configs/local.fc_v03_7b_grpo.json}
  [instruct_7b]=${FC_LOCAL_INSTRUCT_7B:-configs/local.fc_instruct_7b.json}
)
# Bounded-prompt runs that already exist from AC1 (same questions, same greedy settings) are reused.
declare -A REUSE=([v01_7b_ppo_bounded]=outputs/ac1_7b_ppo/sources [v03_3b_ppo_bounded]=outputs/ac1_3b_ppo/sources)
dir_for() { local key="$1_$2"; if [[ -n "${REUSE[$key]:-}" ]]; then echo "${REUSE[$key]}"; else echo "$root/$key"; fi; }
if ! command -v "$py" >/dev/null 2>&1; then printf '%s\n' 'Python 不存在；请设置 FC_PYTHON。' >&2; exit 1; fi
case "$mode" in
  test) "$py" -m unittest discover -s tests -q ;;
  check)
    for label in "${!LOCAL[@]}"; do
      cfg=${LOCAL[$label]}
      if [[ -f "$cfg" ]]; then "$py" -c "import json,sys;c=json.load(open(sys.argv[1]));import os;p=c['agent_checkpoint'];print(sys.argv[2],p,'OK' if os.path.isfile(os.path.join(p,'config.json')) else 'MISSING')" "$cfg" "$label"
      else echo "$label: local config $cfg not found"; fi
    done ;;
  run)   # run LABEL VARIANT
    label=$1; variant=$2; dir=$(dir_for "$label" "$variant")
    if [[ -f "$dir/sources.jsonl" ]]; then echo "$label/$variant done ($dir)"; exit 0; fi
    "$py" -m work2.audit_confirm sources --protocol "configs/fidelity_${variant}.json" --local "${LOCAL[$label]}" \
      --samples-dir "$fresh" --run-dir "$dir" --gpus "$gpus" ;;
  all)
    for label in v01_7b_ppo v03_3b_ppo v03_7b_ppo v03_7b_grpo instruct_7b; do
      for variant in bounded official; do
        if [[ -f "${LOCAL[$label]}" ]]; then bash "$0" run "$label" "$variant"; else echo "skip $label (no ${LOCAL[$label]})"; fi
      done
    done ;;
  summarize)
    args=()
    for label in v01_7b_ppo v03_3b_ppo v03_7b_ppo v03_7b_grpo instruct_7b; do
      for variant in bounded official; do
        d=$(dir_for "$label" "$variant"); [[ -f "$d/sources.jsonl" ]] && args+=(--run "${label}_${variant}=$d")
      done
    done
    pairs=()
    for label in v01_7b_ppo v03_3b_ppo v03_7b_ppo v03_7b_grpo instruct_7b; do pairs+=(--pair "${label}_official-${label}_bounded"); done
    pairs+=(--pair v03_7b_ppo_official-v01_7b_ppo_official --pair v03_7b_ppo_bounded-v01_7b_ppo_bounded)
    for label in v01_7b_ppo v03_7b_ppo v03_7b_grpo v03_3b_ppo; do pairs+=(--pair "${label}_official-instruct_7b_official"); done
    keep=()
    for ((i=0; i<${#pairs[@]}; i+=2)); do
      a=${pairs[i+1]%%-*}; b=${pairs[i+1]#*-}
      [[ " ${args[*]} " == *" --run ${a}="* && " ${args[*]} " == *" --run ${b}="* ]] && keep+=(--pair "${pairs[i+1]}")
    done
    "$py" -m work2.fidelity "${args[@]}" ${keep[@]+"${keep[@]}"} --output "$root/fc1_report.json" ;;
  pack)
    "$py" scripts/pack_feedback.py --include "$root/fc1_report.json" --include "$root/fc1_report.md" ;;
  *) printf '%s\n' 'FC1 modes: test | check | run LABEL VARIANT | all | summarize | pack'
     printf '%s\n' '详见 docs/FC1_FACTORY_GUIDE.md' ;;
esac
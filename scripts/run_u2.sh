#!/usr/bin/env bash
# U2: placebo audits of ReSeek (judge forcing) and MR-Search (reflection turn). Inference only; no training.
# See docs/U2_FACTORY_GUIDE.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
py=${U2_PYTHON:-${U1_PYTHON:-${AC_PYTHON:-python}}}
gpus=${U2_GPUS:-0,1,2,3}
fresh=${U1_FRESH:-outputs/u1_fresh}
plan=outputs/u2/plan
if ! command -v "$py" >/dev/null 2>&1; then printf '%s\n' 'Python 不存在；请设置 U2_PYTHON。' >&2; exit 1; fi
target() {  # target METHOD LABEL -> protocol, local_cfg, d
  case "${1:-}" in
    reseek) protocol=configs/u2_reseek.json ;;
    mr)     protocol=configs/u2_mrsearch.json ;;
    *) printf '%s\n' '方法必须是 reseek 或 mr。' >&2; exit 1 ;;
  esac
  label=${2:-}
  [[ -n "$label" ]] || { printf '%s\n' '缺少智能体标签（例如 reseek_3b、v03_7b_ppo）。' >&2; exit 1; }
  if [[ -f "configs/local.u2_${label}.json" ]]; then local_cfg=configs/local.u2_${label}.json
  elif [[ -f "configs/local.u1_${label}.json" ]]; then local_cfg=configs/local.u1_${label}.json
  else printf '缺少 configs/local.u2_%s.json 或 configs/local.u1_%s.json\n' "$label" "$label" >&2; exit 1; fi
  d=outputs/u2/${1}_${label}
}
case "$mode" in
  test)    "$py" -m unittest discover -s tests -q ;;
  prepare) [[ -f "$plan/plan.json" ]] || "$py" -m work2.u2 prepare --samples-dir "$fresh" --per-dataset "${U2_PER_DATASET:-300}" --output "$plan" ;;
  check)   target "$@"; "$py" -m work2.u2 run --protocol "$protocol" --plan-dir "$plan" --run-dir "outputs/u2/smoke_${1}_${label}" --local "$local_cfg" --label "$label" --gpus "$gpus" --smoke --check-only ;;
  smoke)
    target "$@"; s=outputs/u2/smoke_${1}_${label}
    [[ -f "$s/records.jsonl" ]] || "$py" -m work2.u2 run --protocol "$protocol" --plan-dir "$plan" --run-dir "$s" --local "$local_cfg" --label "$label" --gpus "$gpus" --smoke
    [[ -d "$s/analysis" ]] || "$py" -m work2.u2_analysis --run-dir "$s" --resamples 100
    ;;
  run)
    target "$@"
    [[ -f "outputs/u2/smoke_${1}_${label}/analysis/u2_report.json" ]] || { printf '%s\n' "先完成 smoke $1 $label。" >&2; exit 1; }
    [[ -f "$d/records.jsonl" ]] || "$py" -m work2.u2 run --protocol "$protocol" --plan-dir "$plan" --run-dir "$d" --local "$local_cfg" --label "$label" --gpus "$gpus"
    [[ -d "$d/analysis" ]] || "$py" -m work2.u2_analysis --run-dir "$d"
    ;;
  pack)
    args=(--include "$plan/plan.json")
    for f in outputs/u2/*/run_manifest.json outputs/u2/*/analysis/u2_report.json outputs/u2/*/analysis/u2_report.md; do
      [[ -f "$f" ]] && args+=(--include "$f")
    done
    "$py" scripts/pack_feedback.py "${args[@]}" ;;
  *) printf '%s\n' 'U2 modes: test | prepare | check METHOD LABEL | smoke METHOD LABEL | run METHOD LABEL | pack'
     printf '%s\n' 'METHOD = reseek | mr；LABEL 例：reseek_3b、v03_3b_ppo、v03_7b_ppo。环境变量：U2_PYTHON U2_GPUS U1_FRESH U2_PER_DATASET。详见 docs/U2_FACTORY_GUIDE.md' ;;
esac

#!/usr/bin/env bash
# User-managed offline Search-R1 environment. No downloads, installs or training.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-help}
shift || true
samples=${SI1_SAMPLES:-outputs/si1_dp4_samples}
smoke=${SI1_SMOKE:-outputs/si1_dp4_smoke}
development=${SI1_DEV:-outputs/si1_dp4_dev}
local_config=${SI1_LOCAL:-configs/local.json}
gpus=${SI1_GPUS:-0,1,2,3}
case "$mode" in
  prepare)
    if [[ $# -lt 1 ]]; then
      printf '%s\n' '用法：bash scripts/run_si1.sh prepare 历史1000题JSONL [其他已观察题JSONL ...]' >&2
      exit 1
    fi
    # Conservatively exclude a previously frozen/observed SI1 cohort if it exists.
    if [[ -f outputs/si1_samples/samples.jsonl ]]; then
      set -- "$@" outputs/si1_samples/samples.jsonl
    fi
    python -m work2.run_selective prepare --local "$local_config" --output "$samples" \
      --train-per-dataset "${SI1_TRAIN_PER_DATASET:-600}" \
      --validation-per-dataset "${SI1_VALIDATION_PER_DATASET:-400}" --exclude "$@"
    ;;
  check)
    python -m work2.parallel_selective --local "$local_config" --samples-dir "$samples" --run-dir "$smoke" --gpus "$gpus" --smoke --check-only "$@"
    ;;
  smoke)
    python -m work2.parallel_selective --local "$local_config" --samples-dir "$samples" --run-dir "$smoke" --gpus "$gpus" --smoke "$@"
    ;;
  dev)
    python - "$smoke/audit_receipt.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():raise SystemExit('先完成同一冻结样本的 smoke 和 CPU audit')
r=json.loads(p.read_text())
if not r.get('mechanical_checks_passed') or not r.get('smoke'):raise SystemExit('smoke 收据未通过')
PY
    python -m work2.parallel_selective --local "$local_config" --samples-dir "$samples" --run-dir "$development" --gpus "$gpus" --smoke-run-dir "$smoke" "$@"
    ;;
  analyze)
    python -m work2.run_selective fit --run-dir "$development" --model "$development/selector.json"
    python -m work2.run_selective select --model "$development/selector.json" --states "$development/states.jsonl" --decisions "$development/decisions.json"
    python -m work2.run_selective evaluate --run-dir "$development" --model "$development/selector.json" --decisions "$development/decisions.json" --output "$development/validation_report.json"
    ;;
  pack)
    python scripts/pack_feedback.py \
      --include "$development/run_manifest.json" \
      --include "$development/metrics.json" \
      --include "$development/audit_receipt.json" \
      --include "$development/coordinator_status.json" \
      --include "$development/validation_report.json"
    ;;
  *) printf '%s\n' 'SI1 modes: prepare 历史样本JSONL ... | check | smoke | dev | analyze | pack'
     printf '%s\n' '默认1200训练+800验证，四个TP=1 worker；SI1_GPUS=0,1,2,3。可设置 SI1_LOCAL/SI1_SAMPLES/SI1_SMOKE/SI1_DEV。' ;;
esac

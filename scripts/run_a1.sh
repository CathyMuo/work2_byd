#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  smoke)
    python -m work2.run_baseline --per-dataset-limit 5 --run-dir outputs/a1_smoke "$@"
    ;;
  pilot)
    python -m work2.run_baseline --run-dir outputs/a1_clean_200 "$@"
    ;;
  *)
    printf '%s\n' 'Usage: bash scripts/run_a1.sh smoke|pilot [run_baseline options]' >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m work2.run_replay \
  --source-run outputs/a1_clean_200 \
  --run-dir outputs/a2_sanity \
  --per-dataset-limit 5 \
  --target-turns 1 2 \
  "$@"

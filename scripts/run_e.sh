#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m work2.run_e "$@"

#!/usr/bin/env bash
# Progress board: read-only local web page (research briefing + live run progress). See docs/BOARD_GUIDE.md.
set -euo pipefail
cd "$(dirname "$0")/.."
py=${BOARD_PYTHON:-${U1_PYTHON:-${AC_PYTHON:-python}}}
if ! command -v "$py" >/dev/null 2>&1; then printf '%s\n' 'Python 不存在；请设置 BOARD_PYTHON。' >&2; exit 1; fi
exec "$py" -m work2.board --host "${BOARD_HOST:-127.0.0.1}" --port "${BOARD_PORT:-8765}" "$@"

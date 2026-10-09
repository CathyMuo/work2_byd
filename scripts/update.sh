#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain --ignore-submodules=none)" ]]; then
  printf '%s\n' '存在本地代码修改，请先保留修改；更新已停止。实验 outputs/ 不受影响。' >&2
  exit 1
fi
git -c credential.helper= pull --ff-only
git -c credential.helper= submodule sync --recursive
git -c credential.helper= submodule update --init --recursive
git rev-parse HEAD

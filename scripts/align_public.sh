#!/usr/bin/env bash
# Use only for the user-authorized switch from factory's divergent commits.
# Save the old HEAD locally; no reset --hard, clean, or ignored-file deletion.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ ! -f work2/replay.py || ! -f research/STATE.json ]]; then
  printf '%s\n' '当前目录不是预期的 Work 2 仓库；对齐已停止。' >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --ignore-submodules=none)" ]]; then
  printf '%s\n' '存在未提交修改或未忽略文件，先保留这些文件；对齐已停止。' >&2
  exit 1
fi
if [[ "${1:-}" != "--use-fetched-ref" ]]; then
  GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
    -c remote.origin.url=https://github.com/CathyMuo/work2_byd.git fetch origin main
fi
git rev-parse --verify origin/main >/dev/null
prior=$(git rev-parse HEAD)
target=$(git rev-parse origin/main)
backup="factory-backup/$(date -u +%Y%m%dT%H%M%SZ)-${prior:0:12}"
git branch "$backup" "$prior"
printf '厂内代码备份分支：%s (%s)\n' "$backup" "$prior"
git switch -C main origin/main
git remote set-url origin https://github.com/CathyMuo/work2_byd.git
git -c credential.helper= submodule sync --recursive
GIT_TERMINAL_PROMPT=0 git -c credential.helper= submodule update --init --recursive
printf '公开主线：%s\n本机 outputs/、configs/local.json 未被清理。\n' "$target"

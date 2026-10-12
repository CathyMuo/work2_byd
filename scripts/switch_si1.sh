#!/usr/bin/env bash
# Preserve divergent factory commits; never reset/clean or delete local assets.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ ! -f work2/replay.py || ! -f research/STATE.json ]]; then
  printf '%s\n' '不是 work2_byd；停止。' >&2; exit 1
fi
if [[ -n "$(git status --porcelain --ignore-submodules=none)" ]]; then
  printf '%s\n' '有未提交修改或未忽略文件；先保存这些改动，切换已停止。' >&2; exit 1
fi
branch=feat/selective-intervention
GIT_TERMINAL_PROMPT=0 git -c credential.helper= fetch \
  https://github.com/CathyMuo/work2_byd.git "$branch:refs/remotes/origin/$branch"
prior=$(git rev-parse HEAD)
backup="factory-backup/si1-$(date -u +%Y%m%dT%H%M%SZ)-${prior:0:12}"
git branch "$backup" "$prior"
if git show-ref --verify --quiet "refs/heads/$branch"; then
  git switch "$branch"
  git merge --ff-only "origin/$branch"
else
  git switch -c "$branch" "origin/$branch"
fi
git branch --set-upstream-to="origin/$branch" "$branch"
git remote set-url origin https://github.com/CathyMuo/work2_byd.git
printf '已切换 %s，原提交保留在 %s。outputs/ 和 configs/local.json 未清理。\n' "$branch" "$backup"

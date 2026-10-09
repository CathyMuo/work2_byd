#!/usr/bin/env python3
"""Bundle local environment and explicitly chosen aggregate outputs for feedback."""
import argparse
import datetime
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {".json", ".log", ".txt", ".csv", ".md"}
MAX_FILE_BYTES = 10 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include", type=Path, action="append", default=[],
                        help="specific aggregate/log file under outputs/ (repeatable)")
    args = parser.parse_args()
    output_root = (ROOT / "outputs").resolve()
    files = [output_root / "environment/environment.json"] + args.include
    selected = []
    for item in files:
        path = (ROOT / item).resolve() if not item.is_absolute() else item.resolve()
        try:
            relative = path.relative_to(output_root)
        except ValueError:
            parser.error("Only files inside outputs/ can be included")
        if not path.is_file():
            parser.error("Missing {}. Run check_environment.py first.".format(relative))
        if path.suffix.lower() not in ALLOWED_SUFFIXES or path.name.startswith("."):
            parser.error("Unsupported feedback file: {}".format(relative))
        if path.stat().st_size > MAX_FILE_BYTES:
            parser.error("Feedback file exceeds 10 MiB: {}".format(relative))
        if path not in selected:
            selected.append(path)
    # Do not archive directories, local configuration, datasets, or raw JSONL trajectories.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    if git.returncode:
        parser.error("No repository commit is available")
    commit = git.stdout.strip()
    folder = ROOT / "feedback"
    folder.mkdir(exist_ok=True)
    manifest_path = folder / (stamp + "_manifest.json")
    manifest = {"timestamp_utc": stamp, "repository_commit_at_packaging": commit,
                "note": "Use each run's own commit metadata if results predate packaging.",
                "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in selected]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    archive = folder / ("work2_feedback_{}_{}.tar.gz".format(stamp, commit[:12]))
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(manifest_path, arcname="manifest.json")
        for path in selected:
            bundle.add(path, arcname=str(path.relative_to(ROOT)))
    print(archive)
    print("请将该包反馈到对话中；无需 git push。额外选择的日志请先确认可分享。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only environment inventory; no downloads, credentials, or GPU workloads."""
import argparse
import datetime
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "agent_checkpoint": "directory", "base_model": "directory",
    "retriever_model": "directory", "corpus_file": "file",
    "index_file": "file", "hotpotqa_file": "file", "two_wiki_file": "file",
}


def command(args, timeout=20):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.rstrip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="machine-local JSON asset paths")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/environment/environment.json")
    parser.add_argument("--allow-no-gpu", action="store_true",
                        help="allow CPU-only development checks")
    args = parser.parse_args()
    checks = []
    packages = {}
    for name in ("torch", "vllm", "transformers", "datasets", "faiss-cpu",
                 "faiss-gpu", "fastapi", "uvicorn", "ray", "flash-attn", "verl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = command(["git", "rev-parse", "HEAD"])
    modules = command(["git", "submodule", "status", "--recursive"])
    # argv is fixed; no environment variables or credential configs are captured.
    gpu = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                   "--format=csv,noheader"])
    gpu_lines = [s.strip() for s in gpu["stdout"].splitlines() if s.strip()]
    checks.append({"name": "git_commit", "ok": git["returncode"] == 0})
    module_lines = modules["stdout"].splitlines()
    checks.append({"name": "pinned_submodules",
                   "ok": modules["returncode"] == 0 and len(module_lines) >= 4
                   and all(line.startswith(" ") for line in module_lines)})
    checks.append({"name": "four_h800_gpus",
                   "ok": gpu["returncode"] == 0 and len(gpu_lines) == 4
                   and all("H800" in line for line in gpu_lines),
                   "required": not args.allow_no_gpu})
    assets = {}
    if args.config:
        try:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("config must be a JSON object")
        except (OSError, ValueError) as exc:
            parser.error("Cannot read local config: " + str(exc))
        for name, kind in ASSETS.items():
            value = config.get(name)
            path = Path(value).expanduser() if isinstance(value, str) and value else None
            if path is not None and not path.is_absolute():
                path = ROOT / path
            ok = path is not None and (path.is_dir() if kind == "directory" else path.is_file())
            assets[name] = {"path": str(path) if path else None, "exists": ok, "kind": kind}
            checks.append({"name": "asset:" + name, "ok": ok})
    disk = shutil.disk_usage(ROOT)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repository_commit": git["stdout"] or None,
        "submodules": modules,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "release": platform.release()},
        "packages": packages,
        "gpu": gpu,
        "cuda_compiler": command(["nvcc", "--version"]),
        "disk_free_bytes": disk.free,
        "assets": assets,
        "checks": checks,
        "notes": ["Package absence is informational; environments may be separate.",
                  "Asset checks test existence, not checkpoint/corpus/index validity.",
                  "No inference, training, downloads, or service readiness checks ran."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    for check in checks:
        label = "OK" if check["ok"] else ("INFO" if not check.get("required", True) else "MISSING")
        print("{}: {}".format(label, check["name"]))
    print("Report: {}".format(args.output))
    print("请反馈此报告；这不是模型实验结果。")
    return 0 if all(c["ok"] or not c.get("required", True) for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Start the pinned upstream retriever with user-supplied local resources."""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .common import ROOT, load_json, local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=Path("configs/local.json"))
    parser.add_argument("--faiss-gpu", action="store_true", help="default: CPU FAISS; E5 encoder still uses CUDA")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_json(args.local)
    script = ROOT / "third_party/Search-R1/search_r1/search/retrieval_server.py"
    cmd = [sys.executable, str(script)]
    for key, flag, directory in (("index_file", "--index_path", False),
                                 ("corpus_file", "--corpus_path", False),
                                 ("retriever_model", "--retriever_model", True)):
        path = local_path(config[key])
        if not (path.is_dir() if directory else path.is_file()):
            parser.error("Local asset missing: " + key + " = " + str(path))
        cmd.extend([flag, str(path)])
    cmd.extend(["--retriever_name", "e5", "--topk", "20"])
    if args.faiss_gpu:
        cmd.append("--faiss_gpu")
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()

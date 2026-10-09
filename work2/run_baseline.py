"""Execute A1 clean baseline on frozen local samples with local model weights."""
import argparse
import collections
import datetime
import importlib.metadata
import json
import os
import platform
from pathlib import Path

from .common import ROOT, file_hash, git_output, load_json, local_path, write_json
from .datasets import read_rows
from .metrics import summarize
from .retrieval import SearchR1Retriever
from .rollout import PROMPT_VERSION, run_sample


def validate_design(design):
    agent, retrieval = design["agent"], design["retrieval"]
    for key in ("max_context_tokens", "max_generation_tokens_per_turn", "max_search_turns"):
        if not isinstance(agent[key], int) or agent[key] < 1:
            raise ValueError("Invalid positive agent setting: " + key)
    if agent["temperature"] != 0:
        raise ValueError("A1 initial diagnostic runner supports temperature=0 only")
    if not isinstance(agent["seed"], int):
        raise ValueError("seed must be an integer")
    if (not isinstance(retrieval["visible_top_k"], int) or not isinstance(retrieval["candidate_top_k"], int)
            or not 1 <= retrieval["visible_top_k"] <= retrieval["candidate_top_k"]):
        raise ValueError("Invalid retrieval top-k settings")


def selected_samples(path, limit):
    rows = read_rows(path)
    if not rows:
        raise ValueError("Frozen pilot is empty")
    counts = collections.Counter()
    ids, selected = set(), []
    for row in rows:
        for key in ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts"):
            if key not in row:
                raise ValueError("Noncanonical pilot row; run work2.prepare_pilot first")
        if row["sample_id"] in ids:
            raise ValueError("Duplicate pilot sample ID")
        ids.add(row["sample_id"])
        dataset = row["dataset"]
        if limit is None or counts[dataset] < limit:
            selected.append(row)
            counts[dataset] += 1
    return selected


def asset_identity(path, directory=False):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    if not directory:
        stat = path.stat()
        # Corpus/index may be huge. Store stat identity; never call it a content hash.
        return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "content_hash": None, "identity_strength": "stat_only"}
    weights = sorted(list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin")))
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "model.safetensors.index.json", "pytorch_model.bin.index.json")
    return {"path": str(path), "metadata_sha256": {n: file_hash(path / n) for n in names if (path / n).is_file()},
            "weight_files": [{"name": p.name, "bytes": p.stat().st_size,
                              "mtime_ns": p.stat().st_mtime_ns} for p in weights],
            "identity_strength": "metadata_hashes_and_weight_stats_not_weight_content_hash"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=Path("configs/local.json"))
    parser.add_argument("--design", type=Path, default=Path("configs/diagnostic.json"))
    parser.add_argument("--samples", type=Path, default=Path("outputs/samples/pilot.jsonl"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--per-dataset-limit", type=int)
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--checkpoint-key", choices=("agent_checkpoint", "base_model"), default="agent_checkpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--check-only", action="store_true", help="validate inputs and one retrieval call; no model load")
    args = parser.parse_args()
    try:
        local, design = load_json(args.local), load_json(args.design)
        validate_design(design)
        if args.per_dataset_limit is not None and args.per_dataset_limit < 1:
            raise ValueError("per-dataset-limit must be positive")
        if args.tensor_parallel_size < 1 or not 0 < args.gpu_memory_utilization < 1 or args.request_timeout <= 0:
            raise ValueError("Invalid runtime parallelism/memory/timeout option")
        if args.backend == "hf" and args.tensor_parallel_size != 1:
            raise ValueError("HF backend does not implement tensor parallelism")
        samples = selected_samples(args.samples, args.per_dataset_limit)
        checkpoint = local_path(local[args.checkpoint_key])
        if not (checkpoint / "config.json").is_file():
            raise ValueError("Expected a local Hugging Face checkpoint directory with config.json: " + str(checkpoint))
        retriever = SearchR1Retriever(local["retrieval_url"], args.request_timeout)
        documents = retriever.search(samples[0]["question"], design["retrieval"]["candidate_top_k"])
        if len(documents) < design["retrieval"]["visible_top_k"]:
            raise ValueError("Retriever returned fewer than visible_top_k documents")
    except Exception as exc:
        parser.error("Input/service validation failed: " + str(exc))
    print("Selected samples: {} {}".format(len(samples), dict(collections.Counter(r["dataset"] for r in samples))), flush=True)
    print("Retrieval probe: {} candidates; this probe is not an experimental rollout".format(len(documents)), flush=True)
    if args.check_only:
        print("Input/service check passed. No model inference or experiment ran.")
        return 0
    if args.run_dir.exists():
        parser.error("Run directory already exists; choose a new directory to preserve prior results")
    args.run_dir.mkdir(parents=True)
    package_versions = {}
    for package in ("torch", "transformers", "vllm", "datasets", "faiss-cpu", "faiss-gpu"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    manifest = {"schema_version": 1, "stage": "A1", "condition": "E0_clean",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "repository_commit": git_output("rev-parse", "HEAD"),
                "working_tree_dirty": bool(git_output("status", "--porcelain")),
                "submodules": git_output("submodule", "status", "--recursive"),
                "python": platform.python_version(), "packages": package_versions,
                "prompt_version": PROMPT_VERSION, "design": design,
                "samples_path": str(args.samples.resolve()), "samples_sha256": file_hash(args.samples),
                "selected_ids": [s["sample_id"] for s in samples],
                "backend": args.backend, "checkpoint_key": args.checkpoint_key,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "checkpoint": asset_identity(checkpoint, True),
                "retriever_model": asset_identity(local_path(local["retriever_model"]), True),
                "corpus": asset_identity(local_path(local["corpus_file"])),
                "index": asset_identity(local_path(local["index_file"])),
                "declared_asset_versions": {k: local.get(k) for k in ("checkpoint_version", "corpus_version", "index_version", "actual_index_type")},
                "retrieval_url": local["retrieval_url"], "request_timeout": args.request_timeout,
                "service_identity_note": "Local configured assets do not prove an external retrieval service loaded them. User must ensure service identity.",
                "status": "initializing"}
    write_json(args.run_dir / "run_manifest.json", manifest)
    rows = []
    try:
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        from .backends import HFGenerator, VLLMGenerator
        generator = (VLLMGenerator(checkpoint, design["agent"], args.tensor_parallel_size, args.gpu_memory_utilization)
                     if args.backend == "vllm" else HFGenerator(checkpoint))
        with (args.run_dir / "clean_rollouts.jsonl").open("x", encoding="utf-8") as output:
            for sample in samples:
                row = run_sample(sample, generator, retriever, design["agent"], design["retrieval"])
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                write_json(args.run_dir / "baseline_metrics.json", summarize(rows, samples))
                print("[{}/{}] {} status={} EM={} calls={}".format(
                    len(rows), len(samples), row["sample_id"], row["status"], row["em"], row["search_calls"]), flush=True)
                if row["status"] == "infrastructure_error":
                    raise RuntimeError("Infrastructure error recorded; stopped to avoid invalid batch results")
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(args.run_dir / "baseline_metrics.json", summarize(rows, samples))
        write_json(args.run_dir / "run_manifest.json", manifest)
    print("Completed A1 baseline: {}".format(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run A2 from the immutable A1 run: E0/E0 control, then E0/E1 permutation."""
import argparse
import copy
import datetime
import importlib.metadata
import json
import os
import re
from pathlib import Path

from .common import file_hash, git_output, load_json, local_path, write_json
from .datasets import read_rows
from .metrics import summarize
from .replay import (AuditedRetriever, compare_branches, continuation_signature, observation,
                     permute_documents, replay_branch, validate_prefix)
from .replay_metrics import summarize_replay
from .retrieval import SearchR1Retriever
from .run_baseline import asset_identity, validate_design


def load_source(source_run):
    manifest = load_json(source_run / "run_manifest.json")
    if manifest.get("stage") != "A1" or manifest.get("condition") != "E0_clean" or manifest.get("status") != "completed":
        raise ValueError("A2 needs a completed A1 E0 clean run; do not edit the source manifest")
    if manifest.get("working_tree_dirty", False):
        raise ValueError("A1 recorded uncommitted core changes; audit the source before replay")
    rows = read_rows(source_run / "clean_rollouts.jsonl")
    ids = [r["sample_id"] for r in rows]
    if not rows or ids != manifest["selected_ids"] or len(set(ids)) != len(ids):
        raise ValueError("Source rows do not match the frozen A1 selected-ID order")
    if any(r["status"] == "infrastructure_error" for r in rows):
        raise ValueError("Source run has infrastructure errors")
    sample_path = Path(manifest["samples_path"])
    if not sample_path.is_file() or file_hash(sample_path) != manifest["samples_sha256"]:
        raise ValueError("The original frozen pilot file is missing or changed")
    frozen = {r["sample_id"]: r for r in read_rows(sample_path)}
    for row in rows:
        expected = frozen.get(row["sample_id"])
        if expected is None or any(row.get(k) != expected.get(k) for k in
                                   ("dataset", "question", "gold_answer", "gold_supporting_facts")):
            raise ValueError("Source question/labels differ from the frozen pilot")
    recorded = load_json(source_run / "baseline_metrics.json")["overall"]
    recomputed = summarize(rows, rows)["overall"]
    for key in ("recorded_samples", "evaluable_samples", "infrastructure_errors", "em", "f1", "statuses"):
        if recorded.get(key) != recomputed[key]:
            raise ValueError("Source baseline metrics disagree with raw rows: {}: recorded={!r} ({}), recomputed={!r} ({})".format(
                key, recorded.get(key), type(recorded.get(key)).__name__, recomputed[key], type(recomputed[key]).__name__))
    validate_design(manifest["design"])
    return manifest, rows


def select_rows(rows, limit):
    counts, selected = {}, []
    for row in rows:
        dataset = row["dataset"]
        if counts.get(dataset, 0) < limit:
            selected.append(row)
            counts[dataset] = counts.get(dataset, 0) + 1
    return selected


def index_family(value):
    """Canonical enum or legacy declaration; exclude negations, reject ambiguity.

    This normalizes a declaration, not an inspection of the running FAISS index.
    Historical manifests and local descriptions must remain unchanged.
    """
    text = str(value or "").casefold()
    text = re.sub(r"\b(?:not|no|non)[\s-]+(?:hnsw|flat)(?![a-z])", "", text)
    found = set(re.findall(r"(?<![a-z])(hnsw|flat)(?![a-z])", text))
    return next(iter(found)) if len(found) == 1 else None


def check_runtime(source, local):
    checkpoint = local_path(local[source["checkpoint_key"]])
    now = asset_identity(checkpoint, True)
    prior = source["checkpoint"]
    for key in ("metadata_sha256", "weight_files"):
        if not prior.get(key) or now.get(key) != prior[key]:
            raise ValueError("Checkpoint/tokenizer identity changed or incomplete: " + key)
    if "config.json" not in now["metadata_sha256"] or not any(k.startswith("tokenizer") for k in now["metadata_sha256"]):
        raise ValueError("Source checkpoint has no concrete config/tokenizer hash")
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (checkpoint / name).exists():
            try:
                index = load_json(checkpoint / name)
                mapping = index["weight_map"]
                if not isinstance(mapping, dict) or not mapping or any(not (checkpoint / part).is_file() for part in mapping.values()):
                    raise ValueError("missing weight shards")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("Invalid checkpoint index {}: {}. No automatic repair/download is performed.".format(name, exc))
    for source_key, local_key in (("corpus", "corpus_file"), ("index", "index_file")):
        current = asset_identity(local_path(local[local_key]))
        for key in ("bytes", "mtime_ns"):
            if key not in source[source_key] or current.get(key) != source[source_key][key]:
                raise ValueError("Corpus/index stat identity changed or unavailable: " + source_key)
    if source.get("retrieval_url") and local.get("retrieval_url") != source["retrieval_url"]:
        raise ValueError("Retriever endpoint differs from A1; keep the existing service for this A2 run")
    retriever_model = asset_identity(local_path(local["retriever_model"]), True)
    prior_retriever = source["retriever_model"]
    for key in ("metadata_sha256", "weight_files"):
        if prior_retriever.get(key) != retriever_model.get(key) or not prior_retriever.get(key):
            raise ValueError("Retriever model identity changed or unavailable: " + key)
    actual_index = index_family(source.get("declared_asset_versions", {}).get("actual_index_type"))
    if not actual_index or index_family(local.get("actual_index_type")) != actual_index:
        raise ValueError("Record the real index type consistently with A1 in local.json; keep source manifest immutable")
    versions = {}
    for package in ("torch", "transformers", "vllm"):
        if package == "vllm" and source["backend"] != "vllm":
            continue
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError("Missing inference package: " + package)
        if versions[package] != source["packages"].get(package):
            raise ValueError("Inference package version differs from A1: " + package)
    return checkpoint, {"checkpoint": now, "retriever_model": retriever_model,
                        "packages": versions, "actual_index_type": actual_index,
                        "identity_note": "Metadata hashes and weight/corpus/index stats; not full weight/corpus checksums."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=Path("outputs/a1_clean_200"))
    parser.add_argument("--local", type=Path, default=Path("configs/local.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/a2_sanity"))
    parser.add_argument("--per-dataset-limit", type=int, default=5)
    parser.add_argument("--target-turns", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--check-only", action="store_true", help="validate source artifacts and frozen prefix records; no inference/service call")
    args = parser.parse_args()
    try:
        if args.per_dataset_limit < 1 or args.request_timeout <= 0 or not args.target_turns or len(set(args.target_turns)) != len(args.target_turns):
            raise ValueError("Invalid limit, timeout or duplicate target turns")
        source, rows = load_source(args.source_run)
        if any(t not in (1, 2) for t in args.target_turns):
            raise ValueError("A2 initially supports target turns 1 and 2 only")
        selected = select_rows(rows, args.per_dataset_limit)
        design = copy.deepcopy(source["design"])
        agent, retrieval = design["agent"], design["retrieval"]
        if retrieval["visible_top_k"] != 3 or retrieval["candidate_top_k"] != 20:
            raise ValueError("A2 requires the original Top-3 visible / Top-20 candidate setup")
        for row in selected:
            for turn in args.target_turns:
                if len(row["prefixes"]) >= turn:
                    validate_prefix(row, turn, agent, retrieval)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        parser.error("Source validation failed: " + str(exc))
    selection = [{"sample_id": r["sample_id"], "dataset": r["dataset"],
                  "eligible_target_turns": [t for t in args.target_turns if len(r["prefixes"]) >= t]} for r in selected]
    print(json.dumps({"selected_questions": len(selected), "selection": selection}, ensure_ascii=False), flush=True)
    if args.check_only:
        print("A2 source-record check passed. Runtime identity/tokenizer/service checks occur during execution.")
        return 0
    if args.run_dir.exists():
        parser.error("Run directory exists; use a new --run-dir to preserve prior results")
    args.run_dir.mkdir(parents=True)
    manifest = {"schema_version": 1, "stage": "A2", "repository_commit": git_output("rev-parse", "HEAD"),
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "working_tree_dirty": bool(git_output("status", "--porcelain")),
                "source_run": str(args.source_run.resolve()), "source_commit": source["repository_commit"],
                "source_manifest_sha256": file_hash(args.source_run / "run_manifest.json"),
                "source_rollouts_sha256": file_hash(args.source_run / "clean_rollouts.jsonl"),
                "source_samples_sha256": source["samples_sha256"], "source_design": design,
                "source_asset_versions": source.get("declared_asset_versions"),
                "source_checkpoint": source["checkpoint"], "backend": source["backend"],
                "tensor_parallel_size": source["tensor_parallel_size"],
                "gpu_memory_utilization": source["gpu_memory_utilization"],
                "dtype": "bfloat16", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "selected": selection, "target_turns": args.target_turns,
                "selection_policy": "first N per dataset in frozen A1 ID order, before turn eligibility, including incorrect answers",
                "branch_order_policy": "alternate clean_a/clean_b order by selected index; permutation only after an exact clean control",
                "request_timeout": args.request_timeout, "retrieval_url": None, "status": "initializing"}
    records, audits = [], None
    write_json(args.run_dir / "run_manifest.json", manifest)
    def persist():
        write_json(args.run_dir / "replay_metrics.json", summarize_replay(records, selected, args.target_turns))
        write_json(args.run_dir / "retrieval_audit.json", {"policy": "live normal retrieval, no caching, exact visible-result drift checks",
                                                        "calls": audits.calls if audits else []})
        write_json(args.run_dir / "run_manifest.json", manifest)
    try:
        local = load_json(args.local)
        checkpoint, identity = check_runtime(source, local)
        manifest.update(runtime_identity=identity, retrieval_url=local["retrieval_url"], status="running")
        audits = AuditedRetriever(SearchR1Retriever(local["retrieval_url"], args.request_timeout), retrieval["visible_top_k"])
        # Seed all historical queries in selected rows, including later historical calls.
        for row in selected:
            for prefix in row["prefixes"]:
                audits.seed(prefix["search_query"], prefix["visible_documents"])
        for query in list(audits.expected):
            audits.search(query, retrieval["candidate_top_k"])
        manifest["preflight_retrieval_calls"] = len(audits.calls)
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        from .backends import HFGenerator, VLLMGenerator
        generator = (VLLMGenerator(checkpoint, agent, source["tensor_parallel_size"], source["gpu_memory_utilization"])
                     if source["backend"] == "vllm" else HFGenerator(checkpoint))
        # Full tokenizer reconstruction for all planned reachable prefixes before any branch.
        for row in selected:
            for turn in args.target_turns:
                if len(row["prefixes"]) >= turn:
                    validate_prefix(row, turn, agent, retrieval, generator)
        with (args.run_dir / "replay_sanity.jsonl").open("x", encoding="utf-8") as output:
            for index, row in enumerate(selected):
                for turn in args.target_turns:
                    case = {"sample_id": row["sample_id"], "dataset": row["dataset"], "target_turn": turn,
                            "status": "unreachable" if len(row["prefixes"]) < turn else "running",
                            "checks": {}, "branches": {}}
                    # Preserve incomplete cases in the JSONL even on a failed branch/control.
                    try:
                        if case["status"] != "unreachable":
                            for name in (("clean_a", "clean_b") if index % 2 == 0 else ("clean_b", "clean_a")):
                                result = replay_branch(row, turn, name, generator, audits, agent, retrieval)
                                case["branches"][name] = result
                                if result["status"] == "infrastructure_error":
                                    raise RuntimeError(result.get("error", {}))
                            a, b = (case["branches"][name] for name in ("clean_a", "clean_b"))
                            historical = {**row, "target_turn": turn}
                            case["checks"].update(clean_control_exact=continuation_signature(a) == continuation_signature(b),
                                                  source_history_exact=continuation_signature(a) == continuation_signature(historical))
                            if not case["checks"]["clean_control_exact"]:
                                case["status"] = "control_failed"
                                manifest["status"] = "control_failed"
                            else:
                                permuted, _ = permute_documents(row["prefixes"][turn - 1]["visible_documents"])
                                if observation(permuted) == row["prefixes"][turn - 1]["observation_text"]:
                                    case["status"] = "permutation_unchanged"
                                else:
                                    result = replay_branch(row, turn, "permuted_a", generator, audits, agent, retrieval)
                                    case["branches"]["permuted_a"] = result
                                    if result["status"] == "infrastructure_error":
                                        raise RuntimeError(result.get("error", {}))
                                    case["checks"].update(compare_branches(case["branches"], turn))
                                    case["status"] = "completed"
                    except BaseException as exc:
                        case["status"] = "infrastructure_error"
                        case["error"] = {"type": type(exc).__name__, "message": str(exc)}
                        raise
                    finally:
                        records.append(case)
                        output.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
                        output.flush()
                        persist()
                    print("{} turn={} status={}".format(row["sample_id"], turn, case["status"]), flush=True)
                    if case["status"] == "control_failed":
                        print("E0/E0 differs: stopped before permutation. Inspect the controls; no A3/RL GO.")
                        return 2
        manifest["status"] = "completed"
        ready = summarize_replay(records, selected, args.target_turns)["engineering_gate"]["ready_for_a3_engineering_review"]
        print("A2 completed; ready_for_a3_engineering_review={}".format(ready))
        return 0 if ready else 2
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        persist()


if __name__ == "__main__":
    raise SystemExit(main())

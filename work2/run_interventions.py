"""Run the reviewed 10-question A3 engineering pilot; preserve A1/A2 history."""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path

from .common import file_hash, git_output, load_json, write_json
from .interventions import CONDITIONS, make_packet, substitution, validate_packet
from .replay import (AuditedRetriever, compare_branches, continuation_signature, observation, replay_branch,
                     replay_observation, validate_prefix)
from .retrieval import SearchR1Retriever
from .run_replay import check_runtime, load_source


def summarize_interventions(records, packet):
    def next_query(row):
        later = row["prefixes"][row["target_turn"]:]
        return later[0]["search_query"] if later else None

    def group(condition, turn):
        planned = [c for c in packet["cases"] if c["target_turn"] == turn]
        approved = [c for c in planned if c["reviews"][condition]["decision"] == "approve"]
        pairs = [r for r in records if r["target_turn"] == turn and r["status"] == "completed" and condition in r["branches"]]
        both_search = [r for r in pairs if next_query(r["branches"]["clean_a"]) is not None and next_query(r["branches"][condition]) is not None]
        mean = lambda f: sum(f(r) for r in pairs)/len(pairs) if pairs else None
        return {"planned_states": len(planned), "reachable_states": sum(c["reachable"] for c in planned),
                "approved_states": len(approved), "recorded_pairs": len(pairs),
                "skip_reasons": dict(collections.Counter(c["reviews"][condition]["reason"] for c in planned if c["reviews"][condition]["decision"] == "skip")),
                "clean_em": mean(lambda r: r["branches"]["clean_a"]["em"]),
                "same_subset_e1_em": mean(lambda r: r["branches"]["permuted_a"]["em"]),
                "perturbed_em": mean(lambda r: r["branches"][condition]["em"]),
                "delta_f1": mean(lambda r: r["branches"][condition]["f1"]-r["branches"]["clean_a"]["f1"]),
                "degradation": mean(lambda r: int(r["branches"]["clean_a"]["em"] == 1 and r["branches"][condition]["em"] == 0)),
                "improvement": mean(lambda r: int(r["branches"]["clean_a"]["em"] == 0 and r["branches"][condition]["em"] == 1)),
                "delta_search_calls": mean(lambda r: r["branches"][condition]["search_calls"]-r["branches"]["clean_a"]["search_calls"]),
                "delta_generated_tokens": mean(lambda r: r["branches"][condition]["generated_tokens"]-r["branches"]["clean_a"]["generated_tokens"]),
                "delta_model_input_tokens": mean(lambda r: r["branches"][condition]["model_input_tokens"]-r["branches"]["clean_a"]["model_input_tokens"]),
                "next_query_available_in_both": len(both_search),
                "next_query_string_change_rate_when_both_search": sum(next_query(r["branches"]["clean_a"]) != next_query(r["branches"][condition]) for r in both_search)/len(both_search) if both_search else None,
                "later_search_availability": dict(collections.Counter(("clean_search" if next_query(r["branches"]["clean_a"]) is not None else "clean_no_search") + "/" + ("perturbed_search" if next_query(r["branches"][condition]) is not None else "perturbed_no_search") for r in pairs)),
                "perturbed_statuses": dict(collections.Counter(r["branches"][condition]["status"] for r in pairs))}
    expected = {(c["sample_id"], c["target_turn"]) for c in packet["cases"]}
    complete = len(records) == len(expected) and {(r["sample_id"], r["target_turn"]) for r in records} == expected and all(r["status"] in ("completed", "all_conditions_skipped") for r in records)
    metrics = {c: {str(t): group(c, t) for t in (1, 2)} for c in CONDITIONS}
    controls = [r for r in records if r.get("checks", {}).get("clean_control_exact") is not None]
    return {"schema_version": 1, "by_condition_and_turn": metrics,
            "planned_target_states": len(expected), "recorded_target_states": len(records),
            "statuses": dict(collections.Counter(r["status"] for r in records)),
            "clean_control_pairs": len(controls),
            "clean_control_mismatches": sum(r["checks"]["clean_control_exact"] is not True for r in controls),
            "engineering_gate": {"all_planned_states_recorded": complete,
                                 "all_clean_controls_exact": bool(controls) and all(r["checks"]["clean_control_exact"] is True for r in controls),
                                 "condition_coverage": {c: sum(metrics[c][str(t)]["recorded_pairs"] for t in (1,2)) for c in CONDITIONS}},
            "semantic_offtarget": None, "semantic_recovery": None,
            "interpretation": "Engineering pilot only; 10 questions, correlated target states, condition-specific eligibility. E1 is matched to each subset. No semantic OffTarget/Recovery, confidence intervals, significance or RL GO claimed."}


def run_case(source, case, generator, retriever, agent, retrieval, index):
    approved = [c for c in CONDITIONS if case["reviews"][c]["decision"] == "approve"]
    record = {"sample_id": case["sample_id"], "dataset": case["dataset"], "target_turn": case["target_turn"],
              "reviews": case["reviews"], "status": "running", "checks": {}, "branches": {}}
    if not approved:
        record["status"] = "all_conditions_skipped"
        return record
    turn = case["target_turn"]
    try:
        def save(name, row):
            record["branches"][name] = row
            if row["status"] == "infrastructure_error":
                raise RuntimeError(row.get("error"))
        for name in (("clean_a", "clean_b") if index % 2 == 0 else ("clean_b", "clean_a")):
            save(name, replay_branch(source, turn, name, generator, retriever, agent, retrieval))
        a, b = record["branches"]["clean_a"], record["branches"]["clean_b"]
        exact = continuation_signature(a) == continuation_signature(b)
        history = continuation_signature(a) == continuation_signature({**source, "target_turn": turn})
        record["checks"].update(clean_control_exact=exact, source_history_exact=history)
        if not exact or not history:
            record["status"] = "control_failed"
            return record
        save("permuted_a", replay_branch(source, turn, "permuted_a", generator, retriever, agent, retrieval))
        record["checks"].update(compare_branches(record["branches"], turn))
        order = approved[index % len(approved):] + approved[:index % len(approved)]
        record["intervention_branch_order"] = order
        for condition in order:
            documents, metadata = substitution(case, condition)
            original = source["prefixes"][turn-1]
            metadata["target_observation_token_delta"] = len(generator.encode(observation(documents))) - len(generator.encode(original["observation_text"]))
            row = replay_observation(source, turn, condition, documents, condition, metadata,
                                     generator, retriever, agent, retrieval)
            save(condition, row)
            checks = {"same_raw_prefix": row["prefixes"][turn-1]["prefix_token_ids"] == original["prefix_token_ids"],
                      "pre_target_history_unchanged": row["prefixes"][:turn-1] == source["prefixes"][:turn-1] and row["steps"][:turn] == source["steps"][:turn],
                      "single_slot_changed": sum(x != y for x,y in zip(original["visible_documents"], row["prefixes"][turn-1]["visible_documents"])) == 1,
                      "target_observation_changed": row["prefixes"][turn-1]["observation_text"] != original["observation_text"],
                      "later_retrieval_normal": all(p["visible_documents"] == p["candidate_documents"][:retrieval["visible_top_k"]] for p in row["prefixes"][turn:])}
            record["checks"][condition] = checks
            if not all(checks.values()):
                raise ValueError("Single-observation intervention invariant failed")
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "infrastructure_error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=Path("outputs/a1_clean_200"))
    parser.add_argument("--a2-run", type=Path, default=Path("outputs/a2_sanity"))
    parser.add_argument("--reviews", type=Path, default=Path("outputs/a3_review/reviewed.json"))
    parser.add_argument("--local", type=Path, default=Path("configs/local.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/a3_engineering"))
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--check-only", action="store_true", help="offline provenance/review validation; no model/service calls")
    args = parser.parse_args()
    try:
        if args.request_timeout <= 0:
            raise ValueError("Timeout must be positive")
        source, rows = load_source(args.source_run)
        expected = make_packet(args.source_run, args.a2_run, source, rows)
        packet = load_json(args.reviews)
        approved = validate_packet(packet, expected)
        if approved == 0:
            raise ValueError("No approved interventions; report eligibility limitations before GPU execution")
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        parser.error(str(exc))
    print("A3 packet verified: {} approved interventions; {} planned target states".format(approved, len(packet["cases"])), flush=True)
    if args.check_only:
        return 0
    if args.run_dir.exists():
        parser.error("Run directory exists; use a new path and preserve previous results")
    args.run_dir.mkdir(parents=True)
    manifest = {"schema_version": 1, "stage": "A3_engineering", "status": "initializing",
                "repository_commit": git_output("rev-parse", "HEAD"), "working_tree_dirty": bool(git_output("status", "--porcelain")),
                "submodules": git_output("submodule", "status", "--recursive"),
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "review_packet_sha256": file_hash(args.reviews), "source_manifest_sha256": packet["source_manifest_sha256"],
                "source_rollouts_sha256": packet["source_rollouts_sha256"], "provenance": packet["provenance"],
                "source_design": source["design"], "source_checkpoint": source["checkpoint"],
                "source_asset_versions": source.get("declared_asset_versions"), "backend": source["backend"],
                "tensor_parallel_size": source["tensor_parallel_size"], "gpu_memory_utilization": source["gpu_memory_utilization"],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "request_timeout": args.request_timeout,
                "policy": packet["policy"], "planned_cases": [{"sample_id": c["sample_id"], "dataset": c["dataset"], "target_turn": c["target_turn"]} for c in packet["cases"]]}
    records, audited = [], None
    def persist():
        write_json(args.run_dir / "run_manifest.json", manifest)
        write_json(args.run_dir / "intervention_metrics.json", summarize_interventions(records, packet))
        write_json(args.run_dir / "retrieval_audit.json", {"policy": "live normal retrieval, no cache", "calls": audited.calls if audited else []})
    persist()
    write_json(args.run_dir / "frozen_reviews.json", packet)
    manifest["frozen_reviews_sha256"] = file_hash(args.run_dir / "frozen_reviews.json")
    try:
        local = load_json(args.local)
        checkpoint, identity = check_runtime(source, local)
        manifest.update(runtime_identity=identity, retrieval_url=local["retrieval_url"], status="running")
        audited = AuditedRetriever(SearchR1Retriever(local["retrieval_url"], args.request_timeout), 3)
        selected_ids = {c["sample_id"] for c in packet["cases"]}
        selected = {r["sample_id"]: r for r in rows if r["sample_id"] in selected_ids}
        for row in selected.values():
            for prefix in row["prefixes"]:
                audited.seed(prefix["search_query"], prefix["visible_documents"])
        for query in list(audited.expected):
            audited.search(query, 20)
        manifest["preflight_retrieval_calls"] = len(audited.calls)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        from .backends import HFGenerator, VLLMGenerator
        agent, retrieval = source["design"]["agent"], source["design"]["retrieval"]
        generator = VLLMGenerator(checkpoint, agent, source["tensor_parallel_size"], source["gpu_memory_utilization"]) if source["backend"] == "vllm" else HFGenerator(checkpoint)
        for case in packet["cases"]:
            if case["reachable"]:
                validate_prefix(selected[case["sample_id"]], case["target_turn"], agent, retrieval, generator)
        with (args.run_dir / "paired_rollouts.jsonl").open("x", encoding="utf-8") as output:
            for index, case in enumerate(packet["cases"]):
                result = run_case(selected[case["sample_id"]], case, generator, audited, agent, retrieval, index)
                records.append(result)
                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                persist()
                print("{} turn={} {}".format(case["sample_id"], case["target_turn"], result["status"]), flush=True)
                if result["status"] in ("control_failed", "infrastructure_error"):
                    manifest["status"] = result["status"]
                    manifest["failed_case"] = {"sample_id": result["sample_id"], "target_turn": result["target_turn"], "checks": result["checks"], "error": result.get("error")}
                    print(json.dumps(manifest["failed_case"], ensure_ascii=False), flush=True)
                    return 2
        manifest["status"] = "completed"
        print("A3 engineering pilot completed. Review eligibility and cases before expansion; no RL GO.")
        return 0
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        persist()


if __name__ == "__main__":
    raise SystemExit(main())

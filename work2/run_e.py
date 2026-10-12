"""Control-first E engineering replay from frozen actual perturbed post-observation states."""
import argparse
import collections
import datetime
import json
import math
import os
from pathlib import Path

from .common import file_hash, git_output, load_json, write_json
from .post_evidence import PROMPTS, continue_from_observation, restore_after_observation, value_hash
from .prepare_e import reload_plan
from .replay import AuditedRetriever, continuation_signature
from .retrieval import SearchR1Retriever
from .run_replay import check_runtime
from .semantic_review import event_key


def metrics(records, planned):
    complete = [r for r in records if r["status"] == "completed"]
    by_role = {}
    for role in ("failed", "stable"):
        items = [r for r in complete if r["role"] == role]
        arms = {}
        for name in ("general_review", "constraint_review"):
            pairs = [(r["branches"]["original_a"], r["branches"][name]) for r in items]
            mean = lambda fn: sum(fn(a,b) for a,b in pairs)/len(pairs) if pairs else None
            arms[name] = {"completed_pairs": len(pairs), "original_em": mean(lambda a,b:a["em"]),
                "arm_em": mean(lambda a,b:b["em"]), "gains": sum(a["em"]==0 and b["em"]==1 for a,b in pairs),
                "harms": sum(a["em"]==1 and b["em"]==0 for a,b in pairs),
                "mean_delta_em": mean(lambda a,b:b["em"]-a["em"]),
                "mean_delta_generated_tokens": mean(lambda a,b:b["generated_tokens_after_anchor"]-a["generated_tokens_after_anchor"]),
                "mean_delta_model_input_tokens": mean(lambda a,b:b["model_input_tokens_after_anchor"]-a["model_input_tokens_after_anchor"]),
                "mean_delta_search_calls": mean(lambda a,b:b["search_calls_after_anchor"]-a["search_calls_after_anchor"]),
                "arm_statuses": dict(collections.Counter(b["status"] for a,b in pairs))}
        by_role[role] = arms
    expected = {(c["pair_id"],c["role"]) for c in planned}
    recorded = {(r["pair_id"],r["role"]) for r in records}
    controls = [r for r in records if "original_control_exact" in r["checks"]]
    return {"planned_cases":len(planned),"recorded_cases":len(records),"completed_cases":len(complete),
        "statuses":dict(collections.Counter(r["status"] for r in records)), "by_role":by_role,
        "engineering_gate":{"all_planned_cases_completed":len(complete)==len(expected) and recorded==expected,
            "all_original_controls_and_history_exact":len(controls)==len(expected) and all(r["checks"]["original_control_exact"] and r["checks"]["source_history_exact"] for r in controls)},
        "interpretation":"At most four outcome-selected development pairs. Generic versus constraint review share original global tool/context and per-segment decoding caps; actual costs differ and are reported. No significance, population recovery rate, causal mechanism certification, deployable trigger, F confirmation or RL GO."}


def execute(case, context, name, generator, retriever):
    event = case["event"]
    row = context["trajectories"][event_key(event)]
    if value_hash(row) != case["source_branch_sha256"]:
        raise ValueError("Original perturbed branch changed")
    source = context["source"]
    result = continue_from_observation(row, context["clean"][event["sample_id"]], event["target_turn"],
        case["resume_after_turn"], name, generator, retriever, source["design"]["agent"], source["design"]["retrieval"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("outputs/e_review/reviewed.json"))
    parser.add_argument("--local", type=Path, default=Path("configs/local.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/e_smoke"))
    parser.add_argument("--pair-limit", type=int, default=1, help="first N finalized pairs, chosen before new arm outcomes; 1 smoke then up to 4 engineering")
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--check-only", action="store_true", help="offline source/plan review; no model or service call")
    args = parser.parse_args()
    try:
        if not 1 <= args.pair_limit <= 4 or not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
            raise ValueError("Invalid pair limit/timeout")
        plan, all_cases, context = reload_plan(args.plan)
        pair_ids = list(dict.fromkeys(c["pair_id"] for c in all_cases))[:args.pair_limit]
        planned = [c for c in all_cases if c["pair_id"] in pair_ids]
        if not planned:
            raise ValueError("No complete reviewed failed/stable pairs")
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        parser.error(str(exc))
    print("E frozen plan: {} pairs, {} cases; protocol={}".format(len(pair_ids),len(planned),plan["protocol_sha256"]),flush=True)
    if args.check_only:
        print("Offline source/plan checks passed. Runtime tokenizer, asset and live service validation remain required.")
        return 0
    if args.run_dir.exists():
        parser.error("Run directory exists; use a new path, preserving previous results")
    if git_output("status","--porcelain"):
        parser.error("Current core tree is dirty; preserve changes and use a committed E implementation")
    args.run_dir.mkdir(parents=True)
    source = context["source"]
    manifest = {"schema_version":1,"stage":"E_engineering","status":"initializing",
        "repository_commit":git_output("rev-parse","HEAD"),"working_tree_dirty":False,
        "submodules":git_output("submodule","status","--recursive"),
        "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256":file_hash(args.plan),"source_hashes":plan["source_hashes"],"protocol":plan["protocol"],
        "protocol_sha256":plan["protocol_sha256"],"source_design":source["design"],
        "source_checkpoint":source["checkpoint"],"source_asset_versions":source.get("declared_asset_versions"),
        "backend":source["backend"],"tensor_parallel_size":source["tensor_parallel_size"],
        "gpu_memory_utilization":source["gpu_memory_utilization"],"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_cases":[{"pair_id":c["pair_id"],"role":c["role"],"event_key":event_key(c["event"]),"resume_after_turn":c["resume_after_turn"]} for c in planned],
        "requested_pair_limit":args.pair_limit,"request_timeout":args.request_timeout}
    records = [{"pair_id":c["pair_id"],"role":c["role"],"event_key":event_key(c["event"]),
                "resume_after_turn":c["resume_after_turn"],"status":"planned","checks":{},"branches":{}} for c in planned]
    audited = None
    def atomic(path, value, jsonl=False):
        temporary = path.with_name(path.name+".tmp")
        if jsonl:
            temporary.write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in value),encoding="utf-8")
        else:
            write_json(temporary,value)
        temporary.replace(path)
    def persist():
        atomic(args.run_dir/"run_manifest.json",manifest)
        atomic(args.run_dir/"mechanism_metrics.json",metrics(records,planned))
        atomic(args.run_dir/"mechanism_rollouts.jsonl",records,True)
        atomic(args.run_dir/"retrieval_audit.json",{"policy":"normal live retrieval, no cache; parent perturbed observations are NOT retrieval references","calls":audited.calls if audited else []})
    atomic(args.run_dir/"frozen_plan.json",plan)
    manifest["frozen_plan_sha256"] = file_hash(args.run_dir/"frozen_plan.json")
    persist()
    active = None
    try:
        local = load_json(args.local)
        checkpoint, identity = check_runtime(source,local)
        manifest.update(runtime_identity=identity,retrieval_url=local["retrieval_url"],status="preflight")
        agent,retrieval = source["design"]["agent"],source["design"]["retrieval"]
        audited = AuditedRetriever(SearchR1Retriever(local["retrieval_url"],args.request_timeout),retrieval["visible_top_k"])
        for case in planned:
            event = case["event"]
            clean = context["clean"][event["sample_id"]]
            row = context["trajectories"][event_key(event)]
            for prefix in clean["prefixes"]:
                audited.seed(prefix["search_query"],prefix["visible_documents"])
            for index,prefix in enumerate(row["prefixes"]):
                if index != event["target_turn"]-1:
                    audited.seed(prefix["search_query"],prefix["visible_documents"])
        for query in list(audited.expected):
            audited.search(query,retrieval["candidate_top_k"])
        manifest["preflight_retrieval_calls"] = len(audited.calls)
        os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
        from .backends import HFGenerator,VLLMGenerator
        generator = VLLMGenerator(checkpoint,agent,source["tensor_parallel_size"],source["gpu_memory_utilization"]) if source["backend"]=="vllm" else HFGenerator(checkpoint)
        for case,record in zip(planned,records):
            event = case["event"]
            _,_,anchor = restore_after_observation(context["trajectories"][event_key(event)],context["clean"][event["sample_id"]],event["target_turn"],case["resume_after_turn"],agent,retrieval,generator)
            record["anchor"] = anchor
        manifest["status"] = "original_controls"
        persist()
        # All cases' controls pass before ANY general/constraint review executes.
        for index,(case,record) in enumerate(zip(planned,records)):
            active = record
            record["status"] = "control_running"
            for name in (("original_a","original_b") if index%2==0 else ("original_b","original_a")):
                record["branches"][name] = execute(case,context,name,generator,audited)
                persist()
                if record["branches"][name]["status"] == "infrastructure_error":
                    raise RuntimeError(record["branches"][name].get("error"))
            a,b = record["branches"]["original_a"],record["branches"]["original_b"]
            historical = {**context["trajectories"][event_key(case["event"])],"target_turn":case["resume_after_turn"]}
            record["checks"].update(original_control_exact=continuation_signature(a)==continuation_signature(b),
                                    source_history_exact=continuation_signature(a)==continuation_signature(historical))
            if not all(record["checks"].values()):
                record["status"] = manifest["status"] = "control_failed"
                print("E original control/history mismatch: stopped before every review arm.",flush=True)
                return 2
            record["status"] = "controls_passed"
            persist()
        manifest["status"] = "review_arms"
        for index,(case,record) in enumerate(zip(planned,records)):
            active = record
            record["status"] = "review_running"
            names = ("general_review","constraint_review") if index%2==0 else ("constraint_review","general_review")
            record["review_arm_order"] = names
            for name in names:
                row = execute(case,context,name,generator,audited)
                record["branches"][name] = row
                persist()
                if row["status"] == "infrastructure_error":
                    raise RuntimeError(row.get("error"))
                if row["e_metadata"]["anchor"] != record["anchor"] or row["prefixes"][:case["resume_after_turn"]] != record["branches"]["original_a"]["prefixes"][:case["resume_after_turn"]]:
                    raise ValueError("E review arm changed its frozen past/anchor")
                persist()
            record["status"] = "completed"
            persist()
            print("{} {} completed".format(case["pair_id"],case["role"]),flush=True)
        manifest["status"] = "completed"
        print("E engineering completed; no statistical, triggering, confirmation or RL GO.",flush=True)
        return 0
    except BaseException as exc:
        manifest.update(status="failed",error={"type":type(exc).__name__,"message":str(exc)})
        if active is not None:
            active.update(status="infrastructure_error",error=manifest["error"])
        raise
    finally:
        persist()


if __name__=="__main__":
    raise SystemExit(main())

"""Read-only CPU closeout audit of completed E v2 artifacts; no new policy execution."""
import argparse
import collections
import copy
from pathlib import Path

from .common import file_hash, load_json, write_json
from .datasets import read_rows
from .metrics import answer_scores
from .post_evidence import value_hash
from .prepare_e_probe import reload_probe_plan
from .replay import continuation_signature, observation, token_hash
from .rollout import parse_action
from .run_e_probe import metrics
from .semantic_review import event_key

ARMS = ("original_a", "original_b", "general_review", "constraint_review")
TERMINATIONS = ("generation_limit", "malformed_action", "context_limit")


def validate_branch(row, original, anchor, role, arm, source):
    turn = anchor["resume_after_turn"]
    agent = source["design"]["agent"]
    for key in ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts", "condition", "initial_prompt"):
        if row[key] != original[key]:
            raise ValueError("E branch identity/context changed: " + key)
    if row["branch"] != arm or row["target_turn"] != turn or row["parent_target_turn"] != original["target_turn"]:
        raise ValueError("E branch arm/anchor/parent identity mismatch")
    if row["e_metadata"]["anchor"] != anchor or row["steps"][:turn] != original["steps"][:turn] or row["prefixes"][:turn] != original["prefixes"][:turn]:
        raise ValueError("E branch changed its actual frozen past")
    instruction = source["audit_protocol"]["prompts"][arm]
    if row["e_metadata"]["instruction"] != instruction or row["e_metadata"]["instruction_tokens"] != len(row["e_metadata"]["instruction_token_ids"]):
        raise ValueError("E instruction/cost differs from frozen protocol")
    if arm.startswith("original_") and (row["e_metadata"]["instruction_token_ids"] or row["e_metadata"]["input_after_instruction_sha256"] != anchor["post_observation_token_sha256"]):
        raise ValueError("Original E control has an injected instruction or changed input hash")
    steps, prefixes = row["steps"], row["prefixes"]
    calls = row["search_calls"]
    if calls != len(prefixes) or not turn <= calls <= agent["max_search_turns"]:
        raise ValueError("E global search count drift")
    if any(any(type(t) is not int or t < 0 for t in step["token_ids"]) or len(step["token_ids"]) > agent["max_generation_tokens_per_turn"] for step in steps):
        raise ValueError("E generated IDs or per-segment limit invalid")
    if row["generated_tokens"] != sum(len(step["token_ids"]) for step in steps) or row["model_input_tokens"] != sum(step["input_tokens"] for step in steps):
        raise ValueError("E recorded full token costs disagree with raw steps")
    for key, expected in (("search_calls_after_anchor", calls-turn),
                          ("generated_tokens_after_anchor", sum(len(s["token_ids"]) for s in steps[turn:])),
                          ("model_input_tokens_after_anchor", sum(s["input_tokens"] for s in steps[turn:]))):
        if row[key] != expected:
            raise ValueError("E incremental cost mismatch: " + key)
    for index, prefix in enumerate(prefixes):
        kind, query = parse_action(steps[index]["text"])
        if kind != "search" or query != prefix["search_query"] or prefix["search_turn"] != index+1:
            raise ValueError("E search prefix/action mismatch")
        if prefix["prefix_token_sha256"] != token_hash(prefix["prefix_token_ids"]) or prefix["observation_text"] != observation(prefix["visible_documents"]):
            raise ValueError("E raw token checksum or observation mismatch")
        if prefix["observed_document_ids"] != [d["document_id"] for d in prefix["visible_documents"]]:
            raise ValueError("E observed IDs mismatch")
    status = row["status"]
    if status == "infrastructure_error":
        raise ValueError("Completed E closeout cannot hide infrastructure errors")
    final_kind, final_value = parse_action(steps[-1]["text"]) if len(steps) > turn else (None, None)
    if status == "answered":
        if final_kind != "answer" or final_value != row["final_answer"]:
            raise ValueError("E answered status/action inconsistent")
        scores = answer_scores(row["final_answer"], row["gold_answer"])
    else:
        if status not in (*TERMINATIONS, "search_budget_exhausted") or row["final_answer"]:
            raise ValueError("E terminal status/final-answer inconsistency")
        scores = {"em": 0, "f1": 0.0}
    if any(row[key] != value for key, value in scores.items()):
        raise ValueError("E score differs from raw final answer")
    if status == "search_budget_exhausted" and (calls != agent["max_search_turns"] or final_kind != "search" or len(steps) != calls+1):
        raise ValueError("Budget terminal receipt lacks the final over-budget refused search action")
    return {"status": status, "em": row["em"], "f1": row["f1"], "final_answer": row["final_answer"],
            "new_successful_searches": row["search_calls_after_anchor"],
            "generated_tokens_after_anchor": row["generated_tokens_after_anchor"],
            "model_input_tokens_after_anchor": row["model_input_tokens_after_anchor"],
            "instruction_tokens": row["e_metadata"]["instruction_tokens"],
            "last_action": {"kind": final_kind, "value": final_value},
            "refused_over_budget_query": final_value if status == "search_budget_exhausted" else None,
            "after_anchor_steps": copy.deepcopy(steps[turn:]),
            "later_observations": [{"search_turn": p["search_turn"], "query": p["search_query"], "documents": p["visible_documents"]} for p in prefixes[turn:]]}


def harm_kind(row):
    if row["status"] == "answered":
        return "answered_wrong"
    return "budget_search_refusal" if row["status"] == "search_budget_exhausted" else "termination_"+row["status"]


def audit_run(run_dir):
    manifest = load_json(run_dir/"run_manifest.json")
    if manifest["stage"] != "E_stratified_probe" or manifest["status"] != "completed":
        raise ValueError("Audit only completed v2 runs; preserve partial failures separately")
    frozen = run_dir/"frozen_plan.json"
    if file_hash(frozen) != manifest["frozen_plan_sha256"]:
        raise ValueError("Frozen plan hash mismatch")
    plan, selected, context = reload_probe_plan(frozen)
    group_ids = list(dict.fromkeys(c["pair_id"] for c in selected))[:manifest["requested_pair_limit"]]
    planned = [c for c in selected if c["pair_id"] in group_ids]
    expected = {(c["pair_id"], c["role"]): c for c in planned}
    rows = read_rows(run_dir/"mechanism_rollouts.jsonl")
    actual = {(r["pair_id"], r["role"]): r for r in rows}
    if not expected or expected.keys() != actual.keys() or len(rows) != len(actual):
        raise ValueError("Missing/extra/duplicate E case coverage")
    if manifest["source_hashes"] != plan["source_hashes"] or manifest["protocol"] != plan["protocol"] or manifest["protocol_sha256"] != plan["protocol_sha256"] or manifest["source_design"] != context["source"]["design"]:
        raise ValueError("Manifest source/design/protocol mismatch")
    costs_source = {**context["source"], "audit_protocol": plan["protocol"]}
    paired = collections.Counter()
    harms = {role: {arm: collections.Counter() for arm in ARMS[2:]} for role in ("failed_sufficient", "stable_insufficient")}
    cases = []
    for key, case in expected.items():
        row = actual[key]
        event = case["event"]
        original = context["trajectories"][event_key(event)]
        if value_hash(original) != case["source_branch_sha256"] or row["event_key"] != list(event_key(event)) or row["resume_after_turn"] != case["resume_after_turn"]:
            raise ValueError("E source branch/event/start mismatch")
        if row["status"] != "completed" or set(row["branches"]) != set(ARMS):
            raise ValueError("E planned case missing completed branches")
        a,b = row["branches"]["original_a"],row["branches"]["original_b"]
        historical = {**original, "target_turn":case["resume_after_turn"]}
        if not row["checks"]["original_control_exact"] or not row["checks"]["source_history_exact"] or continuation_signature(a) != continuation_signature(b) or continuation_signature(a) != continuation_signature(historical):
            raise ValueError("E original controls/history not exact")
        if (case["role"] == "failed_sufficient" and a["em"] != 0) or (case["role"] == "stable_insufficient" and a["em"] != 1):
            raise ValueError("E original score inconsistent with stratum")
        expected_anchor = case["anchor_reference"]
        if {k:v for k,v in row["anchor"].items() if k != "post_observation_token_sha256"} != {k:v for k,v in expected_anchor.items() if k != "post_observation_token_sha256"} or len(row["anchor"]["post_observation_token_sha256"]) != 64:
            raise ValueError("Saved E anchor differs from the frozen source reference")
        summaries = {arm: validate_branch(row["branches"][arm], original, row["anchor"], case["role"], arm, costs_source) for arm in ARMS}
        for arm in ARMS[2:]:
            if a["em"] == 1 and row["branches"][arm]["em"] == 0:
                harms[case["role"]][arm][harm_kind(row["branches"][arm])] += 1
        g,c = row["branches"]["general_review"]["em"],row["branches"]["constraint_review"]["em"]
        if case["role"] == "failed_sufficient":
            paired["both" if g and c else "general_only" if g else "constraint_only" if c else "neither"] += 1
        cases.append({"pair_id":key[0], "role":key[1], "event_key":row["event_key"],
                      "question":original["question"], "gold_answer":original["gold_answer"],
                      "anchor":row["anchor"], "initial_observations":case["event"]["visible_observations"],
                      "arms":summaries, "manual_review":{"decision":"pending",
                          "questions":["If repaired, was initial evidence reconsidered, repeated by later search, or supplemented by later facts? Quote actual body and generation; text explanations alone do not prove internal mechanisms.",
                                       "If terminal harm, inspect successful/attempted searches, budget at last step, action before refusal, and whether evidence/context changed."]}})
    recomputed = metrics(rows,planned)
    stored = load_json(run_dir/"mechanism_metrics.json")
    if stored != recomputed:
        different = [k for k in stored.keys() | recomputed.keys() if stored.get(k) != recomputed.get(k)]
        raise ValueError("Stored E metrics differ from raw recomputation: "+", ".join(sorted(different)))
    audit = load_json(run_dir/"retrieval_audit.json")
    if any(not call["matched_reference"] for call in audit["calls"]):
        raise ValueError("Retrieval audit reports drift")
    return {"schema_version":1, "stage":"E_v2_CPU_closeout", "source_run":str(run_dir.resolve()),
            "provenance":{"repository_commit":manifest["repository_commit"],
                          **{name:file_hash(run_dir/name) for name in ("run_manifest.json","frozen_plan.json","mechanism_rollouts.jsonl","mechanism_metrics.json","retrieval_audit.json")}},
            "mechanical_checks_passed":True, "raw_metrics":recomputed,
            "primary_paired_outcomes":dict(paired),
            "harm_categories":{role:{arm:dict(counter) for arm,counter in arms.items()} for role,arms in harms.items()},
            "cases":cases,
            "interpretation":"Read-only receipts and pending human evidence-path explanation. Saved anchor SHAs compared across branches, not independently re-tokenized offline. No semantic automation, population NetEM, mechanism certification, new policy/GPU execution, F or RL GO. Smoke and full engineering share questions; do not pool as independent observations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=Path("outputs/e_probe_engineering"))
    parser.add_argument("--output",type=Path,default=Path("outputs/e_probe_closeout/receipt.json"))
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output exists; preserve previous receipts")
        receipt = audit_run(args.run_dir)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        write_json(args.output,receipt)
        print("CPU closeout passed: {} cases, primary {}, harms {}. No new inference.".format(len(receipt["cases"]),receipt["primary_paired_outcomes"],receipt["harm_categories"]))
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

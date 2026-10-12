"""Freeze E development cases and stable review proposals from existing A1/A3/calibration."""
import argparse
import copy
from pathlib import Path

from .audit_pairs import CONDITIONS, audit_rows
from .common import file_hash, load_json, write_json
from .datasets import read_rows
from .post_evidence import PROTOCOL, restore_after_observation, value_hash
from .run_replay import load_source
from .semantic_review import event_key, indexed_events, raw_events, validate_reviews


def candidate(row, clean, key, resume, agent, retrieval, review=None):
    _, _, anchor = restore_after_observation(row, clean, key[2], resume, agent, retrieval)
    event = {"dataset": key[0], "sample_id": key[1], "target_turn": key[2], "condition": key[3],
             "question": row["question"], "gold_answer": row["gold_answer"],
             "gold_supporting_facts": copy.deepcopy(row["gold_supporting_facts"]),
             "visible_observations": [{"search_turn": p["search_turn"], "documents": copy.deepcopy(p["visible_documents"])} for p in row["prefixes"][:resume]],
             "review": copy.deepcopy(review) if review else {"decision": "pending", "reviewer": "", "reason": "", "earliest_sufficient_turn": None, "supporting_quotes": []}}
    event["review"]["prefix_only_verified"] = False
    return {"event": event, "resume_after_turn": resume, "source_branch_sha256": value_hash(row),
            "anchor_reference": anchor}


def build_plan(source_run, a3_path, packet_path, reviewed_path, max_failed=4, stable_proposals=3):
    if type(max_failed) is not int or not 1 <= max_failed <= 4 or type(stable_proposals) is not int or not 1 <= stable_proposals <= 3:
        raise ValueError("Engineering limits: 1..4 failed questions and 1..3 stable proposals each")
    source, clean_rows = load_source(source_run)
    agent, retrieval = source["design"]["agent"], source["design"]["retrieval"]
    if agent["max_search_turns"] != 4 or agent["max_generation_tokens_per_turn"] != 512 or retrieval["visible_top_k"] != 3 or retrieval["candidate_top_k"] != 20:
        raise ValueError("E v1 requires the frozen 4-search/512-segment/Top3-20 cohort")
    packet, reviewed = load_json(packet_path), load_json(reviewed_path)
    review_receipt = validate_reviews(packet, reviewed, allow_pending=True)
    a3_sha = file_hash(a3_path)
    if packet["provenance"]["a3_rollouts_sha256"] != a3_sha or packet["provenance"]["a1_rollouts_sha256"] != file_hash(source_run / "clean_rollouts.jsonl"):
        raise ValueError("Calibration is not bound to these original A1/A3 files")
    cases = read_rows(a3_path)
    mapping = packet["provenance"].get("input_branch_mapping", {})
    audit_rows(clean_rows, cases, [1, 2], mapping)
    trajectories = raw_events(cases, mapping)
    clean = {r["sample_id"]: r for r in clean_rows}
    order = {r["sample_id"]: i for i, r in enumerate(clean_rows)}
    events = indexed_events(reviewed)
    positives, exclusions = [], []
    for key, event in sorted(events.items(), key=lambda item: (order[item[0][1]], item[0][2], CONDITIONS.index(item[0][3]))):
        if event["review"]["decision"] != "sufficient":
            continue
        row = trajectories[key]
        visible = [{"search_turn": p["search_turn"], "documents": p["visible_documents"]} for p in row["prefixes"]]
        if visible != event["visible_observations"] or row["question"] != event["question"] or row["gold_answer"] != event["gold_answer"] or row["gold_supporting_facts"] != event["gold_supporting_facts"]:
            raise ValueError("Calibration event differs from actual perturbed trajectory")
        if row["status"] != "answered" or row["em"] != 0 or clean[key[1]]["em"] != 1:
            exclusions.append({"event_key": key, "reason": "not a nonempty answered degradation"})
            continue
        if key[1] in {p[0][1] for p in positives} or len(positives) >= max_failed:
            continue
        resume = max(key[2], event["review"]["earliest_sufficient_turn"])
        positives.append((key, candidate(row, clean[key[1]], key, resume, agent, retrieval, event["review"])))
    if not positives:
        raise ValueError("No independently reviewed sufficient answered-error development cases")
    excluded_ids = {key[1] for key, _ in positives}
    reserved, groups = set(), []
    for number, (key, failed) in enumerate(positives, 1):
        resume = failed["resume_after_turn"]
        length = failed["anchor_reference"]["post_observation_input_tokens"]
        eligible = []
        for other, row in trajectories.items():
            if other[0] != key[0] or other[2:] != key[2:] or other[1] in excluded_ids | reserved or row["status"] != "answered" or row["em"] != 1 or clean[other[1]]["em"] != 1:
                continue
            if len(row["prefixes"]) < resume or len(row["steps"]) <= resume:
                continue
            size = row["steps"][resume]["input_tokens"]
            if not 0.75 <= size / length <= 1.25:
                continue
            eligible.append((abs(size-length), order[other[1]], other, row))
        eligible.sort(key=lambda item: item[:2])
        stable = []
        for _, _, other, row in eligible:
            if other[1] in reserved:
                continue
            stable.append(candidate(row, clean[other[1]], other, resume, agent, retrieval))
            reserved.add(other[1])
            if len(stable) >= stable_proposals:
                break
        groups.append({"pair_id": "pair_{}".format(number), "decision": "pending", "reason": "",
                       "failed": failed, "stable_candidates": stable,
                       "match_basis": "same dataset/condition/parent turn/resume turn and remaining calls; post-observation length ratio 0.75..1.25; nearest length then frozen A1 order; one question globally"})
    plan = {"schema_version": 1, "stage": "E_engineering", "protocol": copy.deepcopy(PROTOCOL),
            "protocol_sha256": value_hash(PROTOCOL), "source_design": copy.deepcopy(source["design"]),
            "limits": {"max_failed": max_failed, "stable_proposals": stable_proposals},
            "sources": {"source_run": str(source_run.resolve()), "a3_rollouts": str(a3_path.resolve()),
                        "packet": str(packet_path.resolve()), "reviewed": str(reviewed_path.resolve())},
            "source_hashes": {"a1_manifest": file_hash(source_run / "run_manifest.json"),
                              "a1_rollouts": file_hash(source_run / "clean_rollouts.jsonl"), "a3_rollouts": a3_sha,
                              "packet": file_hash(packet_path), "reviewed": file_hash(reviewed_path)},
            "calibration_receipt": review_receipt, "excluded_reviewed_cases": exclusions, "groups": groups,
            "interpretation": "Outcome-selected engineering only. Existing four sufficient annotations may seed this plan; all runnable failed/stable members require independent sufficiency using only this displayed truncated prefix. No full C/D, triggering, significance or RL GO."}
    return plan, {"source": source, "clean": clean, "trajectories": trajectories}


def context_only(plan):
    value = copy.deepcopy(plan)
    for group in value["groups"]:
        group.pop("decision")
        group.pop("reason")
        for case in [group["failed"], *group["stable_candidates"]]:
            case["event"].pop("review")
    return value


def validate_plan(plan, expected):
    if context_only(plan) != context_only(expected):
        raise ValueError("Frozen E context/protocol changed; edit only decisions/reasons and event reviews")
    selected = []
    for group in plan["groups"]:
        if group["decision"] not in ("run", "skip") or not isinstance(group["reason"], str) or not group["reason"].strip():
            raise ValueError("Finalize every pair as run/skip with a reason before model execution")
        if group["decision"] == "skip":
            continue
        failed = group["failed"]
        stable = None
        for candidate_case in group["stable_candidates"]:
            decision = candidate_case["event"]["review"]["decision"]
            if decision == "pending":
                raise ValueError("Review stable proposals in their frozen order; no skipping a pending nearer match")
            one = {"events": [candidate_case["event"]]}
            validate_reviews(one, one)
            if decision == "sufficient":
                stable = candidate_case
                break
        if stable is None:
            raise ValueError("Runnable pair has no independently reviewed sufficient stable match")
        for role, case in (("failed", failed), ("stable", stable)):
            review = case["event"]["review"]
            validate_reviews({"events": [case["event"]]}, {"events": [case["event"]]})
            if review["decision"] != "sufficient" or review.get("prefix_only_verified") is not True or review["earliest_sufficient_turn"] > case["resume_after_turn"]:
                raise ValueError("Both members need independent sufficient review at the actual prefix-only anchor")
            selected.append({"pair_id": group["pair_id"], "role": role, **copy.deepcopy(case)})
    return selected


def reload_plan(path):
    plan = load_json(path)
    paths = {k: Path(v) for k, v in plan["sources"].items()}
    expected, context = build_plan(paths["source_run"], paths["a3_rollouts"], paths["packet"], paths["reviewed"], **plan["limits"])
    return plan, validate_plan(plan, expected), context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=Path("outputs/a1_clean_scale500"))
    parser.add_argument("--a3-rollouts", type=Path, default=Path("outputs/a3_scale500_merged/paired_rollouts.jsonl"))
    parser.add_argument("--packet", type=Path, default=Path("outputs/semantic_audit/body_diagnostics.json"))
    parser.add_argument("--reviewed", type=Path, default=Path("outputs/semantic_audit/body_reviewed.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/e_review/proposals.json"))
    parser.add_argument("--max-failed", type=int, default=4)
    parser.add_argument("--stable-proposals", type=int, default=3)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    try:
        if args.validate:
            _, selected, _ = reload_plan(args.validate)
            print("E plan validated: {} complete failed/stable pairs. Offline check only.".format(len(selected)//2))
        else:
            if args.output.exists():
                raise ValueError("Output exists; preserve prior proposals")
            plan, _ = build_plan(args.source_run, args.a3_rollouts, args.packet, args.reviewed, args.max_failed, args.stable_proposals)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, plan)
            print("Prepared {} pair groups. Review prefix-only sufficiency and stable proposals; no GPU used.".format(len(plan["groups"])))
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

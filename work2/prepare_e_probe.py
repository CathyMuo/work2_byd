"""Prepare explicit E v2 strata from an immutable, all-skip E v1 review plan."""
import argparse
import copy
from pathlib import Path

from .common import file_hash, load_json, write_json
from .post_evidence import PROTOCOL, value_hash
from .prepare_e import reload_plan
from .semantic_review import validate_reviews

PROBE_PROTOCOL = {**copy.deepcopy(PROTOCOL),
    "version": "work2_E_stratified_probe_v2",
    "selection": "Fixed v1 all-skip pool; sufficient failed primary and prefix-insufficient stable secondary. Each action contrast is within the actual state. No population NetEM, knowledge-source attribution or deployable trigger.",
    "case_structure": "at most four sufficient failed plus one nearest independently reviewed insufficient stable per group; secondary absence does not block primary",
    "secondary_evidence": "identify missing original-question constraints and consider alternative derivations; absent gold string alone is inadequate",
    "old_gate": "E v1 remains all-skip; this probe does not pass or loosen its sufficient-pair gate"}
ROLES = ("failed_sufficient", "stable_insufficient")


def checked_review(case):
    event = case["event"]
    validate_reviews({"events": [event]}, {"events": [event]})
    return event["review"]


def probe_case(group, role, original):
    case = copy.deepcopy(original)
    case.update(pair_id=group["pair_id"], role=role, decision="pending", reason="",
                prior_review=copy.deepcopy(case["event"]["review"]))
    case["event"]["review"]["prefix_only_verified"] = False
    if role == "stable_insufficient":
        case["event"]["review"].update(missing_question_constraints=[], alternative_derivations_considered="")
    return case


def build_probe(parent_path):
    parent, selected, context = reload_plan(parent_path)
    if selected or any(group["decision"] != "skip" for group in parent["groups"]):
        raise ValueError("E v2 entry requires a preserved finalized all-skip v1 plan; do not overwrite or reinterpret v1")
    cases, pool_receipts = [], []
    for group in parent["groups"]:
        failed = group["failed"]
        review = checked_review(failed)
        if review["decision"] != "sufficient" or review.get("prefix_only_verified") is not True or review["earliest_sufficient_turn"] > failed["resume_after_turn"]:
            raise ValueError("All-skip v1 must still contain independently sufficient prefix-only failed cases")
        cases.append(probe_case(group, "failed_sufficient", failed))
        candidates = group["stable_candidates"]
        reviews = [checked_review(case) for case in candidates]
        # Freeze the first already-reviewed insufficient candidate, before any new action outcomes.
        nearest = next((case for case, review in zip(candidates, reviews) if review["decision"] == "insufficient"), None)
        if nearest is not None:
            cases.append(probe_case(group, "stable_insufficient", nearest))
        pool_receipts.append({"pair_id": group["pair_id"], "v1_decision": group["decision"],
                              "v1_reason": group["reason"], "reviewed_stable_candidates": len(candidates),
                              "candidate_decisions": [review["decision"] for review in reviews],
                              "secondary_proposed": nearest is not None})
    plan = {"schema_version": 2, "stage": "E_stratified_probe", "protocol": copy.deepcopy(PROBE_PROTOCOL),
            "protocol_sha256": value_hash(PROBE_PROTOCOL), "source_design": copy.deepcopy(parent["source_design"]),
            "source_hashes": copy.deepcopy(parent["source_hashes"]),
            "parent_plan": str(parent_path.resolve()), "parent_plan_sha256": file_hash(parent_path),
            "v1_pool_receipts": pool_receipts, "cases": cases,
            "interpretation": "E v1 has zero sufficient failed/stable pairs. V2 primary repair and secondary prefix-insufficient stable harm are distinct within-state descriptive development probes, not pooled NetEM or evidence for parametric knowledge/B3 harm mechanisms. Prefix review judgments are not automatically verified semantic entailment. No F/RL GO."}
    return plan, context


def context_only(plan):
    frozen = copy.deepcopy(plan)
    for case in frozen["cases"]:
        case.pop("decision")
        case.pop("reason")
        case["event"].pop("review")
    return frozen


def validate_probe(plan, expected):
    if context_only(plan) != context_only(expected):
        raise ValueError("Frozen v2 context changed; edit only case run/skip reasons and event reviews")
    selected = []
    for case in plan["cases"]:
        if case["decision"] not in ("run", "skip") or not isinstance(case["reason"], str) or not case["reason"].strip():
            raise ValueError("Finalize every v2 case as run/skip with a reason before inference")
        if case["decision"] == "skip":
            continue
        review = checked_review(case)
        if review.get("prefix_only_verified") is not True:
            raise ValueError("Runnable v2 cases need independent prefix-only reaffirmation")
        if case["role"] == "failed_sufficient":
            if review["decision"] != "sufficient" or review["earliest_sufficient_turn"] > case["resume_after_turn"]:
                raise ValueError("Primary cases require sufficient evidence at the frozen actual anchor")
        elif case["role"] == "stable_insufficient":
            gaps = review.get("missing_question_constraints")
            alternatives = review.get("alternative_derivations_considered")
            if review["decision"] != "insufficient" or not isinstance(gaps, list) or not gaps or any(not isinstance(gap, str) or not gap.strip() for gap in gaps) or not isinstance(alternatives, str) or not alternatives.strip():
                raise ValueError("Secondary cases require insufficient review, explicit missing constraints and alternative derivation audit; uncertainty must be skipped")
        else:
            raise ValueError("Unknown v2 stratum")
        selected.append(copy.deepcopy(case))
    if selected and not any(case["role"] == "failed_sufficient" for case in selected):
        raise ValueError("V2 is a primary repair probe; secondary-only execution is not authorized by this protocol")
    return selected


def reload_probe_plan(path):
    plan = load_json(path)
    parent = Path(plan["parent_plan"])
    if file_hash(parent) != plan["parent_plan_sha256"]:
        raise ValueError("Preserved v1 reviewed plan changed")
    expected, context = build_probe(parent)
    return plan, validate_probe(plan, expected), context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, default=Path("outputs/e_review/reviewed.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/e_probe_review/proposals.json"))
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    try:
        if args.validate:
            _, selected, _ = reload_probe_plan(args.validate)
            counts = {role: sum(case["role"] == role for case in selected) for role in ROLES}
            print("V2 validated (CPU only): {}. V1 sufficient-pair gate remains unpassed.".format(counts))
        else:
            if args.output.exists():
                raise ValueError("Output exists; preserve old plans and use a new path")
            plan, _ = build_probe(args.parent_plan)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, plan)
            print("Prepared {} v2 cases from the fixed all-skip v1 pool. Reaffirm each prefix-only review and freeze run/skip.".format(len(plan["cases"])))
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

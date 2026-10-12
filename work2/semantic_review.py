"""Body-only lexical diagnostics and validation of independent evidence reviews.

This is the public implementation, not a reconstruction of unavailable 113ca69.
A lexical score NEVER assigns a semantic sufficient/insufficient decision.
Gold supporting sentences are used for offline audit, never a deployable trigger.
"""
import argparse
import collections
import copy
import re
from pathlib import Path

from .audit_pairs import condition_branches, parse_mapping
from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows

DECISIONS = ("sufficient", "insufficient", "uncertain")
LABELS = ("evidence_use_candidate", "budget_exhaustion", "termination_failure",
          "retrieval_action_candidate", "uncertain", "other")
ANSWER_TYPES = ("entity", "yes_no", "numeric", "other", "unknown")


def event_key(event):
    return (event["dataset"], event["sample_id"], event["target_turn"], event["condition"])


def body(document):
    # Exactly the body rendered by Search-R1 format_documents; the first line
    # is displayed as the title, whether quoted or unquoted.
    return document["contents"].partition("\n")[2]


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def overlap(sentence, text):
    expected = collections.Counter(tokens(sentence))
    return sum((expected & collections.Counter(tokens(text))).values()) / sum(expected.values()) if expected else None


def diagnose(packet, threshold=0.6):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0,1]")
    result = copy.deepcopy(packet)
    for event in result["events"]:
        diagnostics = []
        for index, fact in enumerate(event["gold_supporting_facts"]):
            scores = []
            for observation in event["visible_observations"]:
                for document in observation["documents"]:
                    score = overlap(fact["sentence"], body(document)) if fact.get("sentence") else None
                    scores.append({"search_turn": observation["search_turn"], "document_id": document["document_id"], "token_recall": score})
            valid = [s for s in scores if s["token_recall"] is not None]
            best = max(valid, key=lambda s: s["token_recall"]) if valid else None
            diagnostics.append({"fact_id": index, "best_body_match": best,
                                "proxy_above_threshold": best is not None and best["token_recall"] >= threshold})
        event["lexical_diagnostics"] = {"method": "body_token_multiset_recall_v1", "threshold": threshold,
                                        "facts": diagnostics, "semantic_decision_assigned": False,
                                        "uses_gold_supporting_sentences": True}
    return result


def indexed_events(packet):
    events = packet["events"]
    result = {event_key(e): e for e in events}
    if not result or len(result) != len(events):
        raise ValueError("Review packet events empty or duplicated")
    return result


def validate_reviews(original, reviewed, allow_pending=False):
    expected, actual = indexed_events(original), indexed_events(reviewed)
    if expected.keys() != actual.keys() or {k: v for k, v in original.items() if k != "events"} != {k: v for k, v in reviewed.items() if k != "events"}:
        raise ValueError("Packet coverage/provenance changed")
    for key, event in actual.items():
        if {k: v for k, v in expected[key].items() if k != "review"} != {k: v for k, v in event.items() if k != "review"}:
            raise ValueError("Event context changed; edit only review fields")
        review = event["review"]
        if review["decision"] == "pending" and allow_pending:
            continue
        if review["decision"] not in DECISIONS:
            raise ValueError("Finalize every semantic decision; proxy scores do not decide")
        if any(not isinstance(review.get(k), str) or not review[k].strip() for k in ("reviewer", "reason")) or review.get("independent_of_studied_agent") is not True:
            raise ValueError("Every decision needs an identified independent reviewer and reason")
        turn = review.get("earliest_sufficient_turn")
        if review["decision"] != "sufficient":
            if turn is not None:
                raise ValueError("Insufficient/uncertain decisions must not assert a sufficient turn")
            continue
        observations = {o["search_turn"]: o for o in event["visible_observations"]}
        if type(turn) is not int or turn not in observations or not review.get("supporting_quotes"):
            raise ValueError("Sufficient decisions require a real turn and body quotes")
        if review.get("question_constraints_verified") is not True or not isinstance(review.get("answer_derivation"), str) or not review["answer_derivation"].strip():
            raise ValueError("Sufficient decisions need original-question constraint review and answer derivation")
        quoted_turns = []
        for quote in review["supporting_quotes"]:
            at = quote["search_turn"]
            docs = observations.get(at, {}).get("documents", [])
            matching = [d for d in docs if d["document_id"] == quote["document_id"]]
            text = quote.get("quote", "")
            normalize = lambda value: " ".join(value.casefold().split())
            if type(at) is not int or at > turn or not isinstance(text, str) or not text.strip() or not any(normalize(text) in normalize(body(d)) for d in matching):
                raise ValueError("Quote is not in actual visible body at/before the claimed sufficient turn")
            quoted_turns.append(at)
        if max(quoted_turns) != turn:
            raise ValueError("At least one supporting quote must locate the claimed earliest sufficient turn")
    event_counts = collections.Counter(e["review"]["decision"] for e in actual.values())
    states = collections.defaultdict(set)
    by_condition = collections.defaultdict(collections.Counter)
    for key, event in actual.items():
        states[key[:3]].add(event["review"]["decision"])
        by_condition[key[3]][event["review"]["decision"]] += 1
    state_counts = collections.Counter("sufficient_any_event" if "sufficient" in ds else "pending_without_sufficient" if "pending" in ds else "all_insufficient" if ds == {"insufficient"} else "uncertain_without_sufficient" for ds in states.values())
    return {"events": len(actual), "states": len(states), "event_decisions": dict(event_counts),
            "reviewed_events": len(actual) - event_counts["pending"],
            "all_events_reviewed": event_counts["pending"] == 0,
            "states_with_pending_events": sum("pending" in ds for ds in states.values()),
            "state_decisions": dict(state_counts), "by_condition_event_decisions": dict(by_condition),
            "scope": "Reviewer semantic judgments with mechanically checked provenance/quotes; not automatic entailment certification or recoverability proof. States use an explicit any-sufficient-event rule."}


def raw_events(cases, mapping):
    result = {}
    for case in cases:
        if not case.get("branches"):
            continue
        for condition, row in condition_branches(case, mapping).items():
            key = (case["dataset"], case["sample_id"], case["target_turn"], condition)
            if key in result:
                raise ValueError("Duplicate raw A3 event")
            result[key] = row
    return result


def validate_mechanisms(reviewed, rows, trajectories):
    events = indexed_events(reviewed)
    supplied = {event_key(r): r for r in rows}
    if len(supplied) != len(rows) or supplied.keys() != events.keys():
        raise ValueError("Mechanism labels must cover every reviewed condition event exactly once")
    primary, secondary, answer_types = collections.Counter(), collections.Counter(), collections.Counter()
    state_labels, raw_statuses = collections.defaultdict(set), collections.Counter()
    by_condition = collections.defaultdict(collections.Counter)
    evidence_use_states = set()
    for key, entry in supplied.items():
        event, row = events[key], trajectories[key]
        visible = [{"search_turn": p["search_turn"], "documents": p["visible_documents"]} for p in row["prefixes"]]
        if visible != event["visible_observations"] or row["question"] != event["question"] or row["gold_answer"] != event["gold_answer"] or row["gold_supporting_facts"] != event["gold_supporting_facts"]:
            raise ValueError("Mechanism raw trajectory differs from evidence-review context")
        label, answer_type = entry["primary_label"], entry["answer_type"]
        if label not in LABELS or answer_type not in ANSWER_TYPES or any(not isinstance(entry.get(k), str) or not entry[k].strip() for k in ("reviewer", "reason")):
            raise ValueError("Mechanisms need one primary label, separate answer type, reviewer and reason")
        extras = entry.get("secondary_labels", [])
        if not isinstance(extras, list) or len(set(extras)) != len(extras) or any(e not in LABELS or e == label for e in extras):
            raise ValueError("Invalid secondary labels")
        status = row["status"]
        if status == "search_budget_exhausted" and label != "budget_exhaustion":
            raise ValueError("Budget-exhausted events need a budget primary label; answer type is a separate axis")
        if status in ("generation_limit", "malformed_action", "context_limit") and label != "termination_failure":
            raise ValueError("Terminal failures need a termination primary label; possible upstream causes belong in secondary labels")
        if label == "budget_exhaustion" and status != "search_budget_exhausted":
            raise ValueError("Budget label disagrees with raw terminal status")
        if label == "termination_failure" and status not in ("generation_limit", "malformed_action", "context_limit"):
            raise ValueError("Termination label disagrees with raw terminal status")
        if label == "evidence_use_candidate":
            review = event["review"]
            if review["decision"] != "sufficient" or status != "answered" or not row["final_answer"].strip() or row["em"] != 0 or len(row["steps"]) <= review["earliest_sufficient_turn"]:
                raise ValueError("Evidence-use candidate requires reviewed sufficiency, a subsequent generation and a nonempty wrong answer")
        primary[label] += 1
        by_condition[key[3]][label] += 1
        if label == "evidence_use_candidate":
            evidence_use_states.add(key[:3])
        secondary.update(extras)
        answer_types[answer_type] += 1
        raw_statuses[status] += 1
        state_labels[key[:3]].add(label)
    states = collections.Counter(next(iter(labels)) if len(labels) == 1 else "mixed_across_conditions" for labels in state_labels.values())
    return {"primary_event_counts": dict(primary), "secondary_multilabel_event_counts": dict(secondary),
            "answer_type_event_counts": dict(answer_types), "raw_terminal_status_counts": dict(raw_statuses),
            "exclusive_state_primary_counts": dict(states), "distinct_states": len(state_labels),
            "by_condition_primary_event_counts": dict(by_condition),
            "evidence_use_candidate_states_any_condition": len(evidence_use_states),
            "scope": "Terminal labels checked against raw status; evidence-use remains a reviewed candidate, not a verified causal mechanism. Different primary labels across conditions form an explicit mixed state. Answer type is not a failure cause."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("diagnose", "validate"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path)
    parser.add_argument("--allow-pending", action="store_true", help="validate a calibration subset while counting unreviewed events explicitly; not a complete C/D receipt")
    parser.add_argument("--mechanisms", type=Path, help="explicit event-level primary/secondary label JSONL")
    parser.add_argument("--a3-rollouts", type=Path)
    parser.add_argument("--branch", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output exists; preserve prior artifacts")
        packet = load_json(args.packet)
        indexed_events(packet)
        if args.mode == "diagnose":
            result = diagnose(packet, args.threshold)
        else:
            if not args.reviewed:
                raise ValueError("Validation requires --reviewed")
            reviewed = load_json(args.reviewed)
            result = {"semantic_review": validate_reviews(packet, reviewed, args.allow_pending), "reviewed_sha256": file_hash(args.reviewed)}
            if args.mechanisms:
                if not result["semantic_review"]["all_events_reviewed"]:
                    raise ValueError("Finalize semantic reviews before a complete mechanism receipt")
                if not args.a3_rollouts or packet["provenance"]["a3_rollouts_sha256"] != file_hash(args.a3_rollouts):
                    raise ValueError("Mechanism validation requires the unchanged original A3 file")
                mapping = parse_mapping(args.branch) if args.branch else packet["provenance"].get("input_branch_mapping", {})
                result["mechanisms"] = validate_mechanisms(reviewed, read_rows(args.mechanisms), raw_events(read_rows(args.a3_rollouts), mapping))
                result["mechanism_labels_sha256"] = file_hash(args.mechanisms)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result["validation_provenance" if args.mode == "validate" else "diagnostic_provenance"] = {
            "packet_sha256": file_hash(args.packet), "analysis_commit": git_output("rev-parse", "HEAD"),
            "note": "No model/service call; no automatic semantic decisions. Original sources unchanged."}
        write_json(args.output, result)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        parser.error(str(exc))
    print("Wrote {}. Lexical diagnostics are not semantic certification; sufficient evidence is not witnessed recovery.".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

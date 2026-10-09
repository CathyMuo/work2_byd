"""A2 engineering controls and descriptive paired changes, with explicit denominators."""
import collections


def summarize_replay(records, selected, target_turns):
    def group(items, planned, planned_cases, planned_eligible):
        eligible = [r for r in items if r["status"] != "unreachable"]
        controls = [r for r in eligible if "clean_control_exact" in r.get("checks", {})]
        pairs = [r for r in items if r["status"] == "completed"]
        count = len(pairs)
        mean = lambda field: sum(r["checks"][field] for r in pairs) / count if count else None
        return {"selected_questions": planned, "recorded_target_cases": len(items),
                "planned_target_cases": planned_cases, "eligible_target_prefixes": planned_eligible,
                "recorded_eligible_prefixes": len(eligible),
                "planned_unreachable_target_turns": planned_cases - planned_eligible,
                "unreachable_target_turn": sum(r["status"] == "unreachable" for r in items),
                "clean_control_pairs": len(controls),
                "clean_control_mismatches": sum(not r["checks"]["clean_control_exact"] for r in controls),
                "permutation_pairs": count,
                "degenerate_permutation_skips": sum(r["status"] == "permutation_unchanged" for r in items),
                "statuses": dict(collections.Counter(r["status"] for r in items)),
                "clean_em": mean("clean_em"), "permuted_em": mean("permuted_em"),
                "degradation_rate": mean("degradation"), "improvement_rate": mean("improvement"),
                "mean_delta_search_calls": mean("delta_search_calls"),
                "source_history_mismatches_in_controls": sum(not r["checks"].get("source_history_exact", True) for r in controls),
                "query_string_comparison_note": "String equality is descriptive only; not OffTarget or a semantic drift label."}
    expected_cases = len(selected) * len(target_turns)
    controls = [r for r in records if "clean_control_exact" in r.get("checks", {})]
    expected_keys = {(r["sample_id"], t) for r in selected for t in target_turns}
    complete = (len(records) == expected_cases
                and {(r["sample_id"], r["target_turn"]) for r in records} == expected_keys
                and all(r["status"] in ("completed", "unreachable", "permutation_unchanged") for r in records))
    eligibility = lambda rows, turns: sum(len(r["prefixes"]) >= t for r in rows for t in turns)
    by_turn = {str(t): group([r for r in records if r["target_turn"] == t], len(selected), len(selected),
                            eligibility(selected, [t])) for t in target_turns}
    # Sanity gates are exact implementation invariants, not effect-size/significance thresholds.
    sufficient = all(by_turn[str(t)]["clean_control_pairs"] > 0 and by_turn[str(t)]["permutation_pairs"] > 0 for t in target_turns)
    gate = complete and sufficient and all(r["checks"]["clean_control_exact"] for r in controls)
    datasets = sorted(set(r["dataset"] for r in selected))
    return {"schema_version": 1, "overall": group(records, len(selected), expected_cases, eligibility(selected, target_turns)), "by_target_turn": by_turn,
            "by_dataset_and_turn": {d: {str(t): group([r for r in records if r["dataset"] == d and r["target_turn"] == t],
                                                      sum(s["dataset"] == d for s in selected),
                                                      sum(s["dataset"] == d for s in selected),
                                                      eligibility([s for s in selected if s["dataset"] == d], [t])) for t in target_turns} for d in datasets},
            "engineering_gate": {"all_planned_cases_recorded": complete,
                                 "each_requested_turn_has_control_and_permutation": sufficient,
                                 "all_clean_controls_exact": bool(controls) and all(r["checks"]["clean_control_exact"] for r in controls),
                                 "ready_for_a3_engineering_review": gate},
            "interpretation": "10-question A2 sanity only; inspect artifacts before A3. No statistical, recovery, novelty, or RL GO conclusion."}

"""A3 frozen review packets and conservative, auditable single-slot substitutions."""
import copy

from .common import file_hash, load_json
from .datasets import read_rows
from .metrics import lexical_fact_hits
from .replay import compare_branches, continuation_signature, observation, validate_prefix
from .replay_metrics import summarize_replay
from .run_replay import index_family, select_rows

CONDITIONS = ("E2_distractor", "E3_bridge_removal", "E4_redundancy")
POLICY = {"candidate_ranks": [4, 20], "replacement_character_ratio": [0.5, 2.0],
          "fact_screen": "conservative title+sentence lexical proxy plus independent semantic review",
          "selection": "first N per dataset in frozen A1 order; no outcome filtering",
          "purpose": "10-question engineering pilot; not a research GO or semantic recovery measurement"}


def a2_gate(source_run, a2_run, source, rows):
    """Verify local raw A2 controls/provenance; never rewrite historical metadata."""
    manifest = load_json(a2_run / "run_manifest.json")
    if manifest.get("stage") != "A2" or manifest.get("status") != "completed" or manifest.get("working_tree_dirty"):
        raise ValueError("A3 requires a completed clean-tree A2 run")
    for key, name in (("source_manifest_sha256", "run_manifest.json"),
                      ("source_rollouts_sha256", "clean_rollouts.jsonl")):
        if manifest.get(key) != file_hash(source_run / name):
            raise ValueError("A2 provenance does not match immutable A1: " + key)
    ids = [item["sample_id"] for item in manifest["selected"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate A2 selected IDs")
    selected = [r for r in rows if r["sample_id"] in ids]
    if [r["sample_id"] for r in selected] != ids:
        raise ValueError("A2 selection order differs from A1")
    if ids != [r["sample_id"] for r in select_rows(rows, 5)] or manifest["target_turns"] != [1, 2]:
        raise ValueError("A3 engineering expects the fixed first-five-per-dataset A2 with target turns 1/2")
    cases = read_rows(a2_run / "replay_sanity.jsonl")
    by_id = {r["sample_id"]: r for r in selected}
    for case in cases:
        if case["status"] == "completed":
            actual = compare_branches(case["branches"], case["target_turn"])
            if not actual["clean_control_exact"] or any(case["checks"].get(k) != v for k, v in actual.items()):
                raise ValueError("A2 raw branches disagree with control checks")
            historical = {**by_id[case["sample_id"]], "target_turn": case["target_turn"]}
            if continuation_signature(case["branches"]["clean_a"]) != continuation_signature(historical):
                raise ValueError("A2 clean continuation differs from A1 history; review before A3")
            original = historical["prefixes"][case["target_turn"] - 1]
            if case["branches"]["clean_a"]["prefixes"][case["target_turn"] - 1] != original:
                raise ValueError("A2 clean target prefix differs from A1")
    recomputed = summarize_replay(cases, selected, manifest["target_turns"])
    if recomputed != load_json(a2_run / "replay_metrics.json") or not recomputed["engineering_gate"]["ready_for_a3_engineering_review"]:
        raise ValueError("A2 raw engineering gate has not passed or metrics changed")
    calls = load_json(a2_run / "retrieval_audit.json")["calls"]
    if not calls or any(c.get("matched_reference") is not True for c in calls):
        raise ValueError("A2 retrieval audit missing or contains drift")
    canonical = index_family(source.get("declared_asset_versions", {}).get("actual_index_type"))
    if canonical is None:
        raise ValueError("Ambiguous A1 index declaration")
    prior = manifest.get("runtime_identity", {}).get("actual_index_type")
    if prior != canonical and not (canonical == "flat" and prior == "hnsw"
                                   and str(manifest.get("repository_commit", "")).startswith("a17f1f3")):
        raise ValueError("Unexpected A2 index label mismatch; review provenance")
    return {"a2_manifest_sha256": file_hash(a2_run / "run_manifest.json"),
            "a2_rollouts_sha256": file_hash(a2_run / "replay_sanity.jsonl"),
            "a2_metrics_sha256": file_hash(a2_run / "replay_metrics.json"),
            "a2_audit_sha256": file_hash(a2_run / "retrieval_audit.json"),
            "index_label_erratum": {"historical_a2_label": prior, "normalized_a1_declaration": canonical,
                                    "label_correction_needed": prior != canonical,
                                    "basis": "Legacy negated index declaration parser corrected; no binary FAISS inspection here. Original manifests unchanged."}}


def make_packet(source_run, a2_run, source, rows, limit=5):
    provenance = a2_gate(source_run, a2_run, source, rows)
    agent, retrieval = source["design"]["agent"], source["design"]["retrieval"]
    if retrieval["visible_top_k"] != 3 or retrieval["candidate_top_k"] != 20:
        raise ValueError("A3 requires the frozen Top-3 / Top-20 cohort")
    if limit != 5:
        raise ValueError("This engineering iteration is fixed at the first five questions per dataset")
    cases = []
    for row in select_rows(rows, limit):
        for turn in (1, 2):
            reachable = len(row["prefixes"]) >= turn
            prefix = validate_prefix(row, turn, agent, retrieval) if reachable else None
            # Deliberately omit clean answers, EM and continuations from reviewer view.
            cases.append({"sample_id": row["sample_id"], "dataset": row["dataset"], "target_turn": turn,
                          "question": row["question"], "gold_answer": row["gold_answer"], "gold_supporting_facts": row["gold_supporting_facts"],
                          "reachable": reachable,
                          "prefix_token_sha256": prefix["prefix_token_sha256"] if prefix else None,
                          "query": prefix["search_query"] if prefix else None,
                          "previous_visible_documents": [p["visible_documents"] for p in row["prefixes"][:turn-1]] if reachable else [],
                          "visible_documents": prefix["visible_documents"] if prefix else [],
                          "candidate_documents": prefix["candidate_documents"] if prefix else [],
                          "visible_lexical_fact_ids": lexical_fact_hits(prefix["visible_documents"], row["gold_supporting_facts"]) if prefix else [],
                          "reviews": {c: {"decision": "pending" if reachable else "skip",
                                          "reason": "" if reachable else "unreachable_target_turn",
                                          "reviewer": "", "independent_of_studied_agent": False,
                                          "replacement_slot": None, "candidate_rank": None,
                                          "bridge_fact_ids": [], "redundant_fact_ids": [],
                                          "semantic_checks": {}, "evidence": ""} for c in CONDITIONS}})
    return {"schema_version": 1, "stage": "A3_engineering", "policy": copy.deepcopy(POLICY),
            "source_run": str(source_run.resolve()), "a2_run": str(a2_run.resolve()),
            "source_manifest_sha256": file_hash(source_run / "run_manifest.json"),
            "source_rollouts_sha256": file_hash(source_run / "clean_rollouts.jsonl"),
            "provenance": provenance, "cases": cases}


def substitution(case, condition):
    review = case["reviews"][condition]
    if review["decision"] != "approve":
        raise ValueError("Substitution requires an approved review")
    slot, rank = review["replacement_slot"], review["candidate_rank"]
    if type(slot) is not int or not 0 <= slot < 3 or type(rank) is not int or not 4 <= rank <= 20:
        raise ValueError("Use zero-based slot 0..2 and source candidate rank 4..20")
    candidates = case["candidate_documents"]
    if rank > len(candidates) or candidates[rank-1].get("rank") != rank:
        raise ValueError("Candidate rank unavailable or inconsistent")
    original = case["visible_documents"]
    removed, candidate = original[slot], candidates[rank-1]
    if candidate["document_id"] in [d["document_id"] for d in original] or candidate["contents"] in [d["contents"] for d in original]:
        raise ValueError("Replacement must be a distinct real candidate")
    ratio = len(candidate["contents"]) / max(1, len(removed["contents"]))
    if not POLICY["replacement_character_ratio"][0] <= ratio <= POLICY["replacement_character_ratio"][1]:
        raise ValueError("Replacement character-length ratio outside frozen [0.5, 2.0]")
    result = copy.deepcopy(original)
    result[slot] = copy.deepcopy(candidate)
    facts = case["gold_supporting_facts"]
    old = set(lexical_fact_hits(original, facts))
    new = set(lexical_fact_hits(result, facts))
    remaining = set(lexical_fact_hits([d for i, d in enumerate(original) if i != slot], facts))
    candidate_hits = set(lexical_fact_hits([candidate], facts))
    if condition in ("E2_distractor", "E4_redundancy") and not old <= new:
        raise ValueError("Replacement removes a known supporting-fact proxy")
    if condition == "E2_distractor" and candidate_hits:
        raise ValueError("E2 candidate contains a known gold-fact proxy")
    if condition == "E3_bridge_removal":
        bridge = review["bridge_fact_ids"]
        if not bridge or any(type(i) is not int or not 0 <= i < len(facts) for i in bridge):
            raise ValueError("E3 requires valid zero-based bridge fact IDs")
        previous = set(lexical_fact_hits([d for docs in case["previous_visible_documents"] for d in docs], facts))
        if not set(bridge) <= old - new or set(bridge) & previous or candidate_hits:
            raise ValueError("E3 bridge must be present originally, absent after replacement and in earlier observations; candidate must contain no gold-fact proxy")
        if old - new != set(bridge):
            raise ValueError("E3 must declare every removed known fact; no hidden collateral removal")
    if condition == "E4_redundancy":
        redundant = review["redundant_fact_ids"]
        if not redundant or any(type(i) is not int or not 0 <= i < len(facts) for i in redundant) or not set(redundant) <= remaining & candidate_hits:
            raise ValueError("E4 requires a fact proxy shared by replacement and a retained passage")
    if observation(result) == observation(original):
        raise ValueError("Observation did not change")
    return result, {"type": condition, "replacement_slot": slot, "candidate_rank": rank,
                    "removed_document_id": removed["document_id"], "replacement_document_id": candidate["document_id"],
                    "replacement_character_ratio": ratio, "review": copy.deepcopy(review),
                    "document_multiset_preserved": False}


def validate_packet(packet, expected):
    """Bind all immutable context to source; only reviews may be edited."""
    without_reviews = lambda p: {**p, "cases": [{k: v for k, v in c.items() if k != "reviews"} for c in p["cases"]]}
    if without_reviews(packet) != without_reviews(expected):
        raise ValueError("Review packet context/provenance changed; edit only reviews")
    approved = 0
    for case in packet["cases"]:
        if set(case["reviews"]) != set(CONDITIONS):
            raise ValueError("Keep all E2/E3/E4 review entries")
        for condition, review in case["reviews"].items():
            if review["decision"] not in ("approve", "skip"):
                raise ValueError("Finalize every review as approve or skip before any model execution")
            if not isinstance(review.get("reason"), str) or not review["reason"].strip():
                raise ValueError("Every approved/skipped intervention needs a reason")
            if not case["reachable"]:
                if review["decision"] != "skip":
                    raise ValueError("Unreachable states cannot be approved")
                continue
            if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
                raise ValueError("Reachable decisions need an identified reviewer")
            if review["decision"] == "approve":
                if review.get("independent_of_studied_agent") is not True or not isinstance(review.get("evidence"), str) or not review["evidence"].strip():
                    raise ValueError("Approval requires independent semantic review and quoted evidence")
                required = ["query_relevant", "critical_facts_preserved"] if condition != "E3_bridge_removal" else ["query_relevant", "bridge_role_verified", "bridge_absent_from_prior_history", "replacement_has_no_answer_or_bridge", "other_critical_facts_preserved"]
                if condition == "E2_distractor":
                    required += ["replacement_has_no_answer", "removed_passage_noncritical"]
                if condition == "E4_redundancy":
                    required += ["redundant_evidence_verified", "removed_passage_noncritical"]
                if any(review.get("semantic_checks", {}).get(k) is not True for k in required):
                    raise ValueError("Missing independent semantic checks for " + condition + ": " + ", ".join(required))
                substitution(case, condition)
                approved += 1
    return approved

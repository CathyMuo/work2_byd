"""Body-proxy, reviewer-provenance and event/state regression tests, CPU only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from work2.semantic_review import (body, diagnose, event_key, main, overlap,
                                   validate_mechanisms, validate_reviews)


class SemanticReviewTests(unittest.TestCase):
    def setUp(self):
        self.event = {"sample_id": "hotpotqa:fixture", "dataset": "hotpotqa", "target_turn": 1,
                      "condition": "E1_permutation", "question": "Fixture question?", "gold_answer": "Ada",
                      "gold_supporting_facts": [{"title": "Page", "sentence": "Ada was born in Paris."}],
                      "visible_observations": [{"search_turn": 1, "documents": [
                          {"document_id": "doc", "title": "Page", "contents": '"Page"\nAda was born in Paris.'}]}],
                      "review": {"decision": "pending", "reviewer": "", "earliest_sufficient_turn": None,
                                 "supporting_quotes": [], "reason": ""}}
        self.packet = {"schema_version": 1, "provenance": {"fixture": True}, "events": [self.event]}

    def reviewed(self, packet=None):
        result = copy.deepcopy(packet or self.packet)
        for event in result["events"]:
            event["review"] = {"decision": "sufficient", "reviewer": "Independent CPU fixture",
                               "independent_of_studied_agent": True, "earliest_sufficient_turn": 1,
                               "supporting_quotes": [{"search_turn": 1, "document_id": "doc", "quote": "Ada was born in Paris."}],
                               "question_constraints_verified": True, "answer_derivation": "Fixture explanation, not research evidence.",
                               "reason": "CPU fixture only"}
        return result

    def trajectory(self, event=None, status="answered"):
        event = event or self.event
        return {"status": status, "final_answer": "wrong" if status == "answered" else "", "em": 0,
                "question": event["question"], "gold_answer": event["gold_answer"], "gold_supporting_facts": event["gold_supporting_facts"],
                "steps": [{}, {}], "prefixes": [{"search_turn": 1, "visible_documents": event["visible_observations"][0]["documents"]}]}

    def mechanism(self, event=None, label="evidence_use_candidate"):
        event = event or self.event
        return {**{k: event[k] for k in ("sample_id", "dataset", "target_turn", "condition")},
                "primary_label": label, "secondary_labels": [], "answer_type": "entity",
                "reviewer": "CPU fixture", "reason": "Fixture only"}

    def test_title_only_match_is_removed_from_proxy(self):
        self.assertEqual(body({"contents": '"Ada was born in Paris."\nUnrelated body.'}), "Unrelated body.")
        self.assertEqual(overlap("Ada was born in Paris.", "Unrelated body."), 0)
        self.assertEqual(body({"contents": '"Ada"'}), "")

    def test_high_overlap_missing_negation_does_not_assign_semantic_decision(self):
        packet = copy.deepcopy(self.packet)
        packet["events"][0]["gold_supporting_facts"][0]["sentence"] = "Ada was not born in Paris."
        result = diagnose(packet)
        fact = result["events"][0]["lexical_diagnostics"]["facts"][0]
        self.assertTrue(fact["proxy_above_threshold"])
        self.assertEqual(result["events"][0]["review"]["decision"], "pending")
        self.assertFalse(result["events"][0]["lexical_diagnostics"]["semantic_decision_assigned"])
        self.assertNotIn("lexical_diagnostics", packet["events"][0])

    def test_independent_review_and_body_quotes_required(self):
        good = self.reviewed()
        self.assertEqual(validate_reviews(self.packet, good)["event_decisions"], {"sufficient": 1})
        bad = copy.deepcopy(good)
        bad["events"][0]["review"]["supporting_quotes"][0]["quote"] = "Page"
        with self.assertRaisesRegex(ValueError, "visible body"):
            validate_reviews(self.packet, bad)
        bad = copy.deepcopy(good)
        bad["events"][0]["review"]["independent_of_studied_agent"] = False
        with self.assertRaisesRegex(ValueError, "independent"):
            validate_reviews(self.packet, bad)

    def test_context_corruption_and_duplicate_events_rejected(self):
        bad = self.reviewed()
        bad["events"][0]["question"] = "different question"
        with self.assertRaisesRegex(ValueError, "context changed"):
            validate_reviews(self.packet, bad)
        duplicated = {**self.packet, "events": [self.event, self.event]}
        with self.assertRaisesRegex(ValueError, "duplicated"):
            validate_reviews(duplicated, duplicated)

    def test_state_aggregation_is_explicit_and_condition_overlap_not_double_counted(self):
        second = {**copy.deepcopy(self.event), "condition": "E2_distractor"}
        packet = {**self.packet, "events": [self.event, second]}
        reviewed = self.reviewed(packet)
        reviewed["events"][1]["review"].update(decision="uncertain", earliest_sufficient_turn=None)
        result = validate_reviews(packet, reviewed)
        self.assertEqual(result["events"], 2)
        self.assertEqual(result["states"], 1)
        self.assertEqual(result["state_decisions"], {"sufficient_any_event": 1})

    def test_empty_budget_answer_cannot_be_evidence_use_candidate(self):
        reviewed = self.reviewed()
        rows = [self.mechanism()]
        raw = {event_key(self.event): self.trajectory(status="search_budget_exhausted")}
        with self.assertRaisesRegex(ValueError, "budget primary"):
            validate_mechanisms(reviewed, rows, raw)
        rows[0].update(primary_label="budget_exhaustion", answer_type="yes_no")
        result = validate_mechanisms(reviewed, rows, raw)
        self.assertEqual(result["primary_event_counts"], {"budget_exhaustion": 1})
        self.assertEqual(result["answer_type_event_counts"], {"yes_no": 1})

    def test_mixed_mechanisms_have_one_exclusive_state_count(self):
        second = {**copy.deepcopy(self.event), "condition": "E2_distractor"}
        packet = {**self.packet, "events": [self.event, second]}
        reviewed = self.reviewed(packet)
        rows = [self.mechanism(), self.mechanism(second, "budget_exhaustion")]
        raw = {event_key(self.event): self.trajectory(), event_key(second): self.trajectory(second, "search_budget_exhausted")}
        result = validate_mechanisms(reviewed, rows, raw)
        self.assertEqual(sum(result["primary_event_counts"].values()), 2)
        self.assertEqual(result["exclusive_state_primary_counts"], {"mixed_across_conditions": 1})
        self.assertEqual(result["evidence_use_candidate_states_any_condition"], 1)

    def test_evidence_use_requires_generation_after_sufficient_observation(self):
        row = self.trajectory()
        row["steps"] = [{}]
        with self.assertRaisesRegex(ValueError, "subsequent generation"):
            validate_mechanisms(self.reviewed(), [self.mechanism()], {event_key(self.event): row})

    def test_uncertain_nonentity_review_is_retained_in_denominator(self):
        reviewed = self.reviewed()
        reviewed["events"][0]["review"].update(decision="uncertain", earliest_sufficient_turn=None)
        result = validate_reviews(self.packet, reviewed)
        self.assertEqual(result["states"], 1)
        self.assertEqual(result["state_decisions"], {"uncertain_without_sufficient": 1})

    def test_partial_calibration_keeps_pending_count_and_is_not_a_complete_receipt(self):
        result = validate_reviews(self.packet, self.packet, allow_pending=True)
        self.assertFalse(result["all_events_reviewed"])
        self.assertEqual(result["reviewed_events"], 0)
        self.assertEqual(result["state_decisions"], {"pending_without_sufficient": 1})
        with self.assertRaisesRegex(ValueError, "Finalize"):
            validate_reviews(self.packet, self.packet)

    def test_cli_diagnostics_and_review_receipt_preserve_input_files(self):
        from work2.common import load_json, write_json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packet, diagnostics, reviewed, receipt = [root / name for name in ("packet.json", "diagnostics.json", "reviewed.json", "receipt.json")]
            write_json(packet, self.packet)
            original = packet.read_bytes()
            with mock.patch("sys.argv", ["semantic", "diagnose", "--packet", str(packet), "--output", str(diagnostics)]):
                self.assertEqual(main(), 0)
            prepared = load_json(diagnostics)
            write_json(reviewed, self.reviewed(prepared))
            with mock.patch("sys.argv", ["semantic", "validate", "--packet", str(diagnostics), "--reviewed", str(reviewed), "--output", str(receipt)]):
                self.assertEqual(main(), 0)
            self.assertTrue(load_json(receipt)["semantic_review"]["all_events_reviewed"])
            self.assertEqual(packet.read_bytes(), original)

    def test_native_a3_mechanism_receipt_binds_actual_raw_branch_and_hash(self):
        import test_factory_audit
        from work2.audit_pairs import audit_rows
        from work2.common import file_hash, load_json, write_json
        fixture = test_factory_audit.FactoryAuditTests()
        fixture.setUp()
        _, events, _ = audit_rows([fixture.source], [fixture.case], [2], {})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a3, packet, reviewed, mechanisms, receipt = [root / name for name in ("a3.jsonl", "packet.json", "reviewed.json", "mechanisms.jsonl", "receipt.json")]
            a3.write_text(json.dumps(fixture.case) + "\n")
            value = {"schema_version": 1, "provenance": {"a3_rollouts_sha256": file_hash(a3)}, "events": events}
            write_json(packet, value)
            independent = copy.deepcopy(value)
            independent["events"][0]["review"] = {"decision": "sufficient", "reviewer": "Independent CPU fixture",
                "independent_of_studied_agent": True, "earliest_sufficient_turn": 1,
                "supporting_quotes": [{"search_turn": 1, "document_id": "1", "quote": "Fact 1"}],
                "question_constraints_verified": True, "answer_derivation": "CPU structural fixture only", "reason": "Fixture only"}
            write_json(reviewed, independent)
            mechanisms.write_text(json.dumps(self.mechanism(events[0])) + "\n")
            before = a3.read_bytes()
            with mock.patch("sys.argv", ["semantic", "validate", "--packet", str(packet), "--reviewed", str(reviewed),
                    "--mechanisms", str(mechanisms), "--a3-rollouts", str(a3), "--output", str(receipt)]):
                self.assertEqual(main(), 0)
            self.assertEqual(load_json(receipt)["mechanisms"]["primary_event_counts"], {"evidence_use_candidate": 1})
            self.assertEqual(a3.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

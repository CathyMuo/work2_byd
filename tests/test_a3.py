"""CPU protocol fixtures only: never scientific intervention annotations/results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_a1 import FakeGenerator
import test_a2
from test_a2 import DistinctRetriever
from work2.common import file_hash, load_json, write_json
from work2.interventions import CONDITIONS, make_packet, substitution, validate_packet
from work2.metrics import summarize
from work2.rollout import run_sample
from work2.run_interventions import main, run_case, summarize_interventions
from work2.run_replay import index_family, load_source, main as a2_main


class FactRetriever(DistinctRetriever):
    def search(self, query, top_k):
        docs = super().search(query, top_k)
        # Independent synthetic duplicate, with a distinct upstream ID/content.
        docs[4].update(title="Title1", contents="Title1\nFact 1 extra")
        return docs


class InterventionTests(unittest.TestCase):
    def setUp(self):
        fixture = test_a2.ReplayTests()
        fixture.setUp()
        self.agent, self.retrieval = fixture.agent, fixture.retrieval
        self.sample = {**fixture.sample, "gold_supporting_facts": [
            {"title": "Title0", "sentence": "Fact 0", "sentence_id": 0},
            {"title": "Title1", "sentence": "Fact 1", "sentence_id": 0}]}
        self.row = run_sample(self.sample, FakeGenerator([
            "<search>first</search>", "<search>second</search>", "<answer>SECRET_GOLD</answer>"]),
            FactRetriever(), self.agent, self.retrieval)

    def packet_fixture(self, root):
        a1, a2 = root / "a1", root / "a2"
        a1.mkdir()
        pilot = root / "pilot.jsonl"
        pilot.write_text(json.dumps(self.sample)+"\n")
        (a1 / "clean_rollouts.jsonl").write_text(json.dumps(self.row)+"\n")
        source = {"stage": "A1", "condition": "E0_clean", "status": "completed", "working_tree_dirty": False,
                  "selected_ids": [self.row["sample_id"]], "samples_path": str(pilot), "samples_sha256": file_hash(pilot),
                  "design": {"agent": self.agent, "retrieval": self.retrieval}, "repository_commit": "fixture",
                  "checkpoint_key": "agent_checkpoint", "checkpoint": {}, "backend": "vllm",
                  "tensor_parallel_size": 1, "gpu_memory_utilization": 0.75,
                  "declared_asset_versions": {"actual_index_type": "Flat (e5_Flat.index); NOT HNSW"}}
        write_json(a1 / "run_manifest.json", source)
        write_json(a1 / "baseline_metrics.json", summarize([self.row], [self.row]))
        local = root / "local.json"
        write_json(local, {"retrieval_url": "http://localhost/retrieve"})
        replies = ["<search>second</search>", "<answer>SECRET_GOLD</answer>"] * 3 + ["<answer>SECRET_GOLD</answer>"]*3
        argv = ["a2", "--source-run", str(a1), "--local", str(local), "--run-dir", str(a2)]
        with mock.patch("sys.argv", argv), mock.patch("work2.run_replay.check_runtime", return_value=(root, {"actual_index_type": "hnsw"})), \
                mock.patch("work2.run_replay.SearchR1Retriever", return_value=FactRetriever()), \
                mock.patch("work2.backends.VLLMGenerator", return_value=FakeGenerator(replies)), \
                mock.patch("work2.run_replay.git_output", side_effect=lambda *a: "" if a[0] == "status" else "a17f1f39c0aac5b83fbf6c7f0e17d6711084802f"):
            self.assertEqual(a2_main(), 0)
        packet = make_packet(a1, a2, source, [self.row])
        return a1, a2, local, packet

    def reviewed(self, packet):
        packet = copy.deepcopy(packet)
        for case in packet["cases"]:
            for condition, review in case["reviews"].items():
                review.update(decision="skip", reason="fixture: bridge already in prior history", reviewer="CPU fixture reviewer")
                if case["target_turn"] == 1:
                    review.update(decision="approve", reason="CPU fixture only", independent_of_studied_agent=True,
                                  replacement_slot=0 if condition == "E3_bridge_removal" else 2,
                                  candidate_rank=5 if condition == "E4_redundancy" else 4,
                                  bridge_fact_ids=[0] if condition == "E3_bridge_removal" else [],
                                  redundant_fact_ids=[1] if condition == "E4_redundancy" else [],
                                  evidence="Synthetic fixture quotes: Fact 0, Fact 1; not research evidence.",
                                  semantic_checks={k: True for k in ("query_relevant", "critical_facts_preserved", "replacement_has_no_answer", "removed_passage_noncritical", "bridge_role_verified", "bridge_absent_from_prior_history", "replacement_has_no_answer_or_bridge", "other_critical_facts_preserved", "redundant_evidence_verified")})
        return packet

    def test_negated_index_labels_and_ambiguity(self):
        for text, expected in [("Flat (e5_Flat.index); NOT HNSW", "flat"), ("HNSW", "hnsw"),
                               ("Flat", "flat"), ("HNSW; NOT Flat", "hnsw"), ("e5_Flat.index", "flat"),
                               ("Flat / HNSW", None), ("unknown", None), ("not hnsw", None), (None, None)]:
            with self.subTest(text=text):
                self.assertEqual(index_family(text), expected)

    def test_f1_disagreement_includes_exact_values_and_types(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = test_a2.ReplayTests()
            fixture.setUp()
            a1 = fixture.fixture_source(Path(folder))
            metrics = load_json(a1 / "baseline_metrics.json")
            metrics["overall"]["f1"] = 0.9999999999999999
            write_json(a1 / "baseline_metrics.json", metrics)
            with self.assertRaisesRegex(ValueError, r"f1: recorded=0.9999999999999999 \(float\), recomputed=1.0 \(float\)"):
                load_source(a1)

    def test_reviews_are_required_and_context_is_bound_to_history(self):
        with tempfile.TemporaryDirectory() as folder:
            a1, a2, local, packet = self.packet_fixture(Path(folder))
            self.assertTrue(packet["provenance"]["index_label_erratum"]["label_correction_needed"])
            self.assertEqual(packet["provenance"]["index_label_erratum"]["normalized_a1_declaration"], "flat")
            self.assertNotIn("em", packet["cases"][0])
            with self.assertRaisesRegex(ValueError, "Finalize"):
                validate_packet(packet, packet)
            reviewed = self.reviewed(packet)
            self.assertEqual(validate_packet(reviewed, packet), 3)
            reviewed["cases"][0]["candidate_documents"][3]["contents"] = "made-up intervention"
            with self.assertRaisesRegex(ValueError, "context/provenance"):
                validate_packet(reviewed, packet)

    def test_semantic_review_cannot_be_replaced_by_lexical_hits(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, packet = self.packet_fixture(Path(folder))
            reviewed = self.reviewed(packet)
            reviewed["cases"][0]["reviews"]["E3_bridge_removal"]["semantic_checks"]["bridge_role_verified"] = False
            with self.assertRaisesRegex(ValueError, "semantic checks"):
                validate_packet(reviewed, packet)

    def test_failed_a2_raw_metrics_cannot_enter_review_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            a1, a2, _, packet = self.packet_fixture(Path(folder))
            metrics = load_json(a2 / "replay_metrics.json")
            metrics["engineering_gate"]["ready_for_a3_engineering_review"] = False
            write_json(a2 / "replay_metrics.json", metrics)
            source, rows = load_source(a1)
            with self.assertRaisesRegex(ValueError, "raw engineering gate"):
                make_packet(a1, a2, source, rows)

    def test_bridge_previously_seen_and_collateral_loss_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, packet = self.packet_fixture(Path(folder))
            reviewed = self.reviewed(packet)
            first, second = reviewed["cases"]
            second["reviews"]["E3_bridge_removal"] = copy.deepcopy(first["reviews"]["E3_bridge_removal"])
            with self.assertRaisesRegex(ValueError, "earlier observations"):
                substitution(second, "E3_bridge_removal")
            first["reviews"]["E2_distractor"]["replacement_slot"] = 0
            with self.assertRaisesRegex(ValueError, "removes a known"):
                substitution(first, "E2_distractor")

    def test_length_and_redundancy_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, packet = self.packet_fixture(Path(folder))
            case = self.reviewed(packet)["cases"][0]
            case["candidate_documents"][3]["contents"] *= 10
            with self.assertRaisesRegex(ValueError, "length ratio"):
                substitution(case, "E2_distractor")
            case["reviews"]["E4_redundancy"]["redundant_fact_ids"] = [0]
            with self.assertRaisesRegex(ValueError, "shared"):
                substitution(case, "E4_redundancy")

    def test_only_one_observation_changes_and_budget_remains_global(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, packet = self.packet_fixture(Path(folder))
            case = self.reviewed(packet)["cases"][0]
            original = copy.deepcopy(self.row)
            replies = ["<search>second</search>", "<answer>SECRET_GOLD</answer>"]*3
            replies += ["<search>second</search>", "<search>third</search>", "<search>fourth</search>", "<search>refused</search>"]*3
            result = run_case(self.row, case, FakeGenerator(replies), FactRetriever(), self.agent, self.retrieval, 0)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(self.row, original)
            for condition in CONDITIONS:
                row = result["branches"][condition]
                self.assertEqual(row["status"], "search_budget_exhausted")
                self.assertEqual(row["search_calls"], 4)
                self.assertEqual(row["prefixes"][0]["prefix_token_ids"], original["prefixes"][0]["prefix_token_ids"])
                self.assertTrue(all(result["checks"][condition].values()))

    def test_control_failure_stops_all_intervention_branches(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, packet = self.packet_fixture(Path(folder))
            case = self.reviewed(packet)["cases"][0]
            generator = FakeGenerator(["<search>second</search>", "<answer>SECRET_GOLD</answer>",
                                       "<search>second</search>", "<think>changed</think><answer>SECRET_GOLD</answer>"])
            result = run_case(self.row, case, generator, FactRetriever(), self.agent, self.retrieval, 0)
            self.assertEqual(result["status"], "control_failed")
            self.assertEqual(set(result["branches"]), {"clean_a", "clean_b"})
            summary = summarize_interventions([result], self.reviewed(packet))
            self.assertEqual(summary["by_condition_and_turn"]["E3_bridge_removal"]["1"]["approved_states"], 1)
            self.assertEqual(summary["by_condition_and_turn"]["E3_bridge_removal"]["1"]["recorded_pairs"], 0)
            self.assertFalse(summary["engineering_gate"]["all_planned_states_recorded"])

    def test_cli_artifacts_and_runtime_parameters_preserve_original_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a1, a2, local, packet = self.packet_fixture(root)
            original_hashes = {str(p): file_hash(p) for d in (a1, a2) for p in d.iterdir()}
            reviews = root / "reviewed.json"
            write_json(reviews, self.reviewed(packet))
            run = root / "a3"
            argv = ["a3", "--source-run", str(a1), "--a2-run", str(a2), "--reviews", str(reviews), "--local", str(local), "--run-dir", str(run)]
            replies = ["<search>second</search>", "<answer>SECRET_GOLD</answer>"]*6
            with mock.patch("sys.argv", argv), mock.patch("work2.run_interventions.check_runtime", return_value=(root, {"actual_index_type": "flat"})), \
                    mock.patch("work2.run_interventions.SearchR1Retriever", return_value=FactRetriever()), \
                    mock.patch("work2.backends.VLLMGenerator", return_value=FakeGenerator(replies)) as backend:
                self.assertEqual(main(), 0)
                self.assertEqual(backend.call_args.args[3], 0.75)
            self.assertEqual({name: file_hash(Path(name)) for name in original_hashes}, original_hashes)
            self.assertEqual(load_json(run / "run_manifest.json")["status"], "completed")
            metrics = load_json(run / "intervention_metrics.json")
            self.assertTrue(metrics["engineering_gate"]["all_planned_states_recorded"])
            self.assertEqual(metrics["engineering_gate"]["condition_coverage"], dict.fromkeys(CONDITIONS, 1))
            self.assertIsNone(metrics["semantic_recovery"])
            for condition in CONDITIONS:
                self.assertEqual(metrics["by_condition_and_turn"][condition]["1"]["same_subset_e1_em"], 1.0)


if __name__ == "__main__":
    unittest.main()

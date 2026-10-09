import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_a1 import FakeGenerator
from work2.common import file_hash, load_json, write_json
from work2.metrics import summarize
from work2.replay import (AuditedRetriever, RetrieverDriftError, compare_branches, continuation_signature,
                          replay_branch, validate_prefix)
from work2.replay_metrics import summarize_replay
from work2.rollout import run_sample
from work2.run_replay import check_runtime, load_source, main, select_rows


class DistinctRetriever:
    def __init__(self):
        self.queries = []
        self.changed = False

    def search(self, query, top_k):
        self.queries.append(query)
        return [{"document_id": str(i), "title": "Title" + str(i),
                 "contents": "Title{}\nFact {}{}".format(i, i, "CHANGED" if self.changed else ""),
                 "rank": i + 1, "score": 1.0} for i in range(top_k)]


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.sample = {"sample_id": "hotpotqa:1", "dataset": "hotpotqa", "question": "Where?",
                       "gold_answer": "SECRET_GOLD", "gold_supporting_facts": []}
        self.agent = {"max_search_turns": 4, "max_context_tokens": 10000,
                      "max_generation_tokens_per_turn": 512, "seed": 42, "temperature": 0}
        self.retrieval = {"visible_top_k": 3, "candidate_top_k": 20}
        self.source = run_sample(self.sample, FakeGenerator([
            "<search>first</search>", "<search>second</search>", "<answer>SECRET_GOLD</answer>"]),
            DistinctRetriever(), self.agent, self.retrieval)

    def branch(self, name, turn=2, responses=None):
        generator = FakeGenerator(responses or ["<answer>SECRET_GOLD</answer>"])
        result = replay_branch(self.source, turn, name, generator, DistinctRetriever(), self.agent, self.retrieval)
        return result, generator

    def test_same_raw_prefix_only_target_observation_changes(self):
        original = copy.deepcopy(self.source)
        a, ga = self.branch("clean_a")
        b, gb = self.branch("clean_b")
        p, gp = self.branch("permuted_a")
        target = original["prefixes"][1]
        self.assertEqual(a["prefixes"][1]["prefix_token_ids"], target["prefix_token_ids"])
        self.assertEqual(a["prefixes"][1]["observation_text"], target["observation_text"])
        self.assertEqual(p["prefixes"][1]["observed_document_ids"], ["1", "2", "0"])
        self.assertEqual(ga.inputs[0], target["prefix_token_ids"] + ga.encode(target["observation_text"]))
        self.assertEqual(gp.inputs[0][:len(target["prefix_token_ids"])], target["prefix_token_ids"])
        self.assertEqual(p["prefixes"][0], original["prefixes"][0])
        self.assertEqual(self.source, original)
        checks = compare_branches({"clean_a": a, "clean_b": b, "permuted_a": p}, 2)
        self.assertTrue(checks["clean_control_exact"])
        self.assertTrue(checks["document_multiset_preserved"])
        self.assertEqual(continuation_signature(a), continuation_signature({**self.source, "target_turn": 2}))

    def test_spent_budget_is_not_reset_and_later_docs_are_normal(self):
        generator = FakeGenerator(["<search>third</search>", "<search>fourth</search>", "<search>refused</search>"])
        retriever = DistinctRetriever()
        row = replay_branch(self.source, 2, "permuted_a", generator, retriever, self.agent, self.retrieval)
        self.assertEqual(row["status"], "search_budget_exhausted")
        self.assertEqual(retriever.queries, ["third", "fourth"])
        self.assertEqual(row["search_calls"], 4)
        self.assertEqual(row["search_calls_after_target"], 2)
        for prefix in row["prefixes"][2:]:
            self.assertEqual(prefix["observed_document_ids"], ["0", "1", "2"])
            self.assertEqual(prefix["visible_documents"], prefix["candidate_documents"][:3])

    def test_control_detects_token_differences_with_same_answer(self):
        a, _ = self.branch("clean_a")
        b, _ = self.branch("clean_b", responses=["<think>extra</think><answer>SECRET_GOLD</answer>"])
        p, _ = self.branch("permuted_a")
        checks = compare_branches({"clean_a": a, "clean_b": b, "permuted_a": p}, 2)
        self.assertFalse(checks["clean_control_exact"])
        self.assertTrue(checks["clean_control_answer_equal"])

    def test_corrupt_prefix_or_observation_is_rejected_before_generation(self):
        for key in ("prefix_token_sha256", "observation_text", "search_query"):
            corrupt = copy.deepcopy(self.source)
            corrupt["prefixes"][1][key] = "corrupt"
            generator = FakeGenerator([])
            with self.assertRaises(ValueError):
                replay_branch(corrupt, 2, "clean_a", generator, DistinctRetriever(), self.agent, self.retrieval)
            self.assertEqual(generator.inputs, [])

    def test_changed_tokenizer_is_rejected(self):
        generator = FakeGenerator([])
        generator.encode = lambda text: list(text.encode()) + [999]
        with self.assertRaisesRegex(ValueError, "token"):
            validate_prefix(self.source, 2, self.agent, self.retrieval, generator)

    def test_generated_ids_are_preserved_without_reencoding(self):
        original = copy.deepcopy(self.source)
        original["steps"][1]["token_ids"] = [999]
        original["prefixes"][1]["prefix_token_ids"] = (original["prefixes"][0]["prefix_token_ids"]
            + FakeGenerator([]).encode(original["prefixes"][0]["observation_text"]) + [999])
        from work2.replay import token_hash
        original["prefixes"][1]["prefix_token_sha256"] = token_hash(original["prefixes"][1]["prefix_token_ids"])
        original["prefixes"][1]["generated_tokens_before_observation"] = len(original["steps"][0]["token_ids"]) + 1
        generator = FakeGenerator(["<answer>x</answer>"])
        row = replay_branch(original, 2, "clean_a", generator, DistinctRetriever(), self.agent, self.retrieval)
        self.assertEqual(row["prefixes"][1]["prefix_token_ids"][-1], 999)
        self.assertEqual(generator.inputs[0][len(original["prefixes"][1]["prefix_token_ids"]) - 1], 999)

    def test_live_retrieval_drift_is_exposed_not_cached(self):
        live = DistinctRetriever()
        audited = AuditedRetriever(live, 3)
        audited.seed("query", live.search("query", 20))
        audited.search("query", 20)
        live.changed = True
        with self.assertRaises(RetrieverDriftError):
            audited.search("query", 20)
        self.assertEqual(live.queries, ["query", "query", "query"])
        self.assertFalse(audited.calls[-1]["matched_reference"])

    def test_selection_is_independent_of_outcomes_and_turn_eligibility(self):
        short = {**self.source, "sample_id": "hotpotqa:short", "prefixes": self.source["prefixes"][:1], "em": 0}
        other = {**self.source, "sample_id": "hotpotqa:other", "em": 1}
        self.assertEqual(select_rows([short, other], 1), [short])
        records = [{"sample_id": short["sample_id"], "dataset": short["dataset"], "target_turn": 2,
                    "status": "unreachable", "checks": {}, "branches": {}}]
        result = summarize_replay(records, [short], [2])
        self.assertEqual(result["by_target_turn"]["2"]["eligible_target_prefixes"], 0)
        self.assertEqual(result["by_target_turn"]["2"]["unreachable_target_turn"], 1)
        self.assertFalse(result["engineering_gate"]["ready_for_a3_engineering_review"])

    def fixture_source(self, root):
        source = root / "a1"
        source.mkdir()
        pilot = root / "pilot.jsonl"
        pilot.write_text(json.dumps(self.sample) + "\n")
        (source / "clean_rollouts.jsonl").write_text(json.dumps(self.source) + "\n")
        manifest = {"stage": "A1", "condition": "E0_clean", "status": "completed",
                    "selected_ids": [self.sample["sample_id"]], "samples_path": str(pilot),
                    "samples_sha256": file_hash(pilot), "design": {"agent": self.agent, "retrieval": self.retrieval},
                    "repository_commit": "fixture", "checkpoint": {}, "checkpoint_key": "agent_checkpoint",
                    "backend": "vllm", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.8}
        write_json(source / "run_manifest.json", manifest)
        write_json(source / "baseline_metrics.json", summarize([self.source], [self.source]))
        return source

    def test_changed_frozen_pilot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.fixture_source(root)
            load_source(source)
            (root / "pilot.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "pilot"):
                load_source(source)

    def test_runtime_identity_accepts_unchanged_assets_and_rejects_checkpoint_change(self):
        from work2.run_baseline import asset_identity
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model"
            model.mkdir()
            for name in ("config.json", "tokenizer_config.json"):
                (model / name).write_text('{}')
            (model / "model.safetensors").write_text('CPU fixture, not a model')
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"fixture": "model.safetensors"}}))
            corpus, index = root / "corpus", root / "index"
            corpus.write_text('fixture')
            index.write_text('fixture')
            local = {"agent_checkpoint": str(model), "retriever_model": str(model),
                     "corpus_file": str(corpus), "index_file": str(index), "actual_index_type": "Flat",
                     "retrieval_url": "http://localhost/retrieve"}
            source = {"checkpoint_key": "agent_checkpoint", "checkpoint": asset_identity(model, True),
                      "retriever_model": asset_identity(model, True), "corpus": asset_identity(corpus), "index": asset_identity(index),
                      "declared_asset_versions": {"actual_index_type": "FAISS Flat"}, "backend": "vllm",
                      "retrieval_url": local["retrieval_url"], "packages": {p: "fixture" for p in ("torch", "transformers", "vllm")}}
            with mock.patch("work2.run_replay.importlib.metadata.version", return_value="fixture"):
                self.assertEqual(check_runtime(source, local)[0], model)
                with self.assertRaisesRegex(ValueError, "index type"):
                    check_runtime(source, {**local, "actual_index_type": "HNSW"})
                (model / "tokenizer_config.json").write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    check_runtime(source, local)

    def test_partial_controls_keep_planned_eligibility_denominator(self):
        record = {"sample_id": self.source["sample_id"], "dataset": self.source["dataset"],
                  "target_turn": 1, "status": "control_failed", "checks": {"clean_control_exact": False}, "branches": {}}
        result = summarize_replay([record], [self.source], [1, 2])
        self.assertEqual(result["by_target_turn"]["2"]["eligible_target_prefixes"], 1)
        self.assertEqual(result["by_target_turn"]["2"]["recorded_eligible_prefixes"], 0)
        self.assertFalse(result["engineering_gate"]["ready_for_a3_engineering_review"])

    def test_cli_completed_and_control_failed_do_not_execute_invalid_permutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.fixture_source(root)
            local = root / "local.json"
            local.write_text(json.dumps({"retrieval_url": "http://localhost/retrieve"}))
            for mismatch in (False, True):
                run = root / ("mismatch" if mismatch else "ok")
                replies = (["<answer>SECRET_GOLD</answer>", "<think>extra</think><answer>SECRET_GOLD</answer>"]
                           if mismatch else ["<answer>SECRET_GOLD</answer>"] * 3)
                generator = FakeGenerator(replies)
                argv = ["replay", "--source-run", str(source), "--local", str(local),
                        "--run-dir", str(run), "--per-dataset-limit", "1", "--target-turns", "2"]
                with mock.patch("sys.argv", argv), mock.patch("work2.run_replay.check_runtime", return_value=(root, {})), \
                        mock.patch("work2.run_replay.SearchR1Retriever", return_value=DistinctRetriever()), \
                        mock.patch("work2.backends.VLLMGenerator", return_value=generator):
                    self.assertEqual(main(), 2 if mismatch else 0)
                status = load_json(run / "run_manifest.json")["status"]
                self.assertEqual(status, "control_failed" if mismatch else "completed")
                case = json.loads((run / "replay_sanity.jsonl").read_text())
                self.assertEqual(len(case["branches"]), 2 if mismatch else 3)
                self.assertEqual(load_json(run / "replay_metrics.json")["engineering_gate"]["ready_for_a3_engineering_review"], not mismatch)


if __name__ == "__main__":
    unittest.main()

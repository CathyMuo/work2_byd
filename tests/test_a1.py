import copy
import json
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from work2.common import load_json
from work2.datasets import canonicalize, freeze_pilot, read_rows
from work2.metrics import answer_scores, lexical_fact_hits, summarize
from work2.retrieval import SearchR1Retriever
from work2.rollout import run_sample
from work2.run_baseline import selected_samples


class FakeGenerator:
    """CPU-only protocol fixture, never exposed as an experiment backend."""
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def encode(self, text):
        return list(text.encode())

    def initial_prompt(self, text):
        return text

    def generate(self, tokens, limit, seed):
        self.inputs.append(list(tokens))
        text = next(self.responses)
        return {"text": text, "token_ids": self.encode(text), "finish_reason": "stop"}


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        return [{"document_id": str(i), "title": "Bridge", "contents": "Bridge\nKnown fact.",
                 "rank": i + 1, "score": 1.0} for i in range(top_k)]


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.sample = {"sample_id": "hotpotqa:1", "dataset": "hotpotqa", "question": "Where?",
                       "gold_answer": "SECRET_GOLD", "gold_supporting_facts": [
                           {"title": "Bridge", "sentence_id": 0, "sentence": "Known fact."}]}
        self.agent = {"max_search_turns": 4, "max_context_tokens": 10000,
                      "max_generation_tokens_per_turn": 512, "seed": 42}
        self.retrieval = {"visible_top_k": 3, "candidate_top_k": 20}

    def test_raw_prefix_and_gold_isolation(self):
        generator = FakeGenerator(["<search>bridge</search>", "<search>second</search>", "<answer>place</answer>"])
        retriever = FakeRetriever()
        row = run_sample(self.sample, generator, retriever, self.agent, self.retrieval)
        self.assertEqual(row["status"], "answered")
        self.assertEqual(row["search_calls"], 2)
        self.assertEqual(len(row["prefixes"][0]["candidate_documents"]), 20)
        self.assertEqual(len(row["prefixes"][0]["visible_documents"]), 3)
        first = row["prefixes"][0]
        self.assertEqual(first["prefix_token_ids"], generator.encode(first["prefix_text"]))
        self.assertEqual(generator.inputs[1], first["prefix_token_ids"] + generator.encode(first["observation_text"]))
        for tokens in generator.inputs:
            self.assertNotIn("SECRET_GOLD", bytes(tokens).decode())
            self.assertNotIn("gold_supporting_facts", bytes(tokens).decode())
        self.assertEqual(first["visible_lexical_fact_ids"], [0])
        self.assertEqual(first["original_document_ids"], first["observed_document_ids"])

    def test_fifth_search_is_refused(self):
        generator = FakeGenerator(["<search>q</search>"] * 5)
        retriever = FakeRetriever()
        row = run_sample(self.sample, generator, retriever, self.agent, self.retrieval)
        self.assertEqual(row["status"], "search_budget_exhausted")
        self.assertEqual(len(retriever.queries), 4)
        self.assertEqual(row["em"], 0)

    def test_answer_after_four_searches(self):
        generator = FakeGenerator(["<search>q</search>"] * 4 + ["<answer>SECRET_GOLD</answer>"])
        row = run_sample(self.sample, generator, FakeRetriever(), self.agent, self.retrieval)
        self.assertEqual(row["status"], "answered")
        self.assertEqual(row["em"], 1)

    def test_prefix_keeps_original_generated_token_ids(self):
        generator = FakeGenerator([])
        outputs = iter([
            {"text": "<search>q</search>", "token_ids": [999], "finish_reason": "stop"},
            {"text": "<answer>x</answer>", "token_ids": [998], "finish_reason": "stop"}])
        generator.generate = lambda tokens, limit, seed: next(outputs)
        row = run_sample(self.sample, generator, FakeRetriever(), self.agent, self.retrieval)
        self.assertEqual(row["prefixes"][0]["prefix_token_ids"][-1], 999)
        self.assertNotEqual(row["prefixes"][0]["prefix_token_ids"], generator.encode(row["prefixes"][0]["prefix_text"]))

    def test_context_does_not_silently_truncate(self):
        self.agent["max_context_tokens"] = 10
        generator = FakeGenerator([])
        row = run_sample(self.sample, generator, FakeRetriever(), self.agent, self.retrieval)
        self.assertEqual(row["status"], "context_limit")
        self.assertEqual(generator.inputs, [])

    def test_malformed_and_infrastructure_denominators(self):
        row = run_sample(self.sample, FakeGenerator(["untagged answer"]), FakeRetriever(), self.agent, self.retrieval)
        self.assertEqual(row["status"], "malformed_action")
        error = copy.deepcopy(row)
        error.update(status="infrastructure_error", em=None, f1=None)
        summary = summarize([row, error], [self.sample, self.sample])["overall"]
        self.assertEqual(summary["evaluable_samples"], 1)
        self.assertEqual(summary["infrastructure_errors"], 1)
        self.assertEqual(summary["em"], 0)

    def test_missing_answer_never_scores_correct(self):
        self.sample["gold_answer"] = "A"
        row = run_sample(self.sample, FakeGenerator(["untagged"]), FakeRetriever(), self.agent, self.retrieval)
        self.assertEqual(row["em"], 0)
        self.assertEqual(row["f1"], 0.0)

    def test_fact_title_alone_is_not_evidence(self):
        self.assertEqual(lexical_fact_hits([{"title": "Bridge", "contents": "Bridge\nDifferent fact."}],
                                        self.sample["gold_supporting_facts"]), [])

    def test_answer_metrics(self):
        self.assertEqual(answer_scores("The Paris!", "Paris"), {"em": 1, "f1": 1.0})
        self.assertAlmostEqual(answer_scores("New York", "York")["f1"], 2 / 3)
        self.assertEqual(answer_scores("yes maybe", "yes")["f1"], 0)

    def test_local_selection_is_stable_and_preserves_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local = {}
            for dataset, key in (("hot", "hotpotqa_file"), ("wiki", "two_wiki_file")):
                path = root / (dataset + ".json")
                rows = [{"_id": str(i), "question": dataset + str(i), "answer": "a",
                         "context": [["Title", ["Fact."]]], "supporting_facts": [["Title", 0]]}
                        for i in range(10)]
                path.write_text(json.dumps(rows))
                local[key] = str(path)
            a = freeze_pilot(local, root / "one", count=5)
            b = freeze_pilot(local, root / "two", count=5)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(len(selected_samples(a, 2)), 4)
            self.assertEqual(read_rows(a)[0]["gold_supporting_facts"][0]["sentence"], "Fact.")
            with self.assertRaises(ValueError):
                freeze_pilot(local, root / "one", count=5)
            self.assertIn("pilot_sha256", load_json(a.parent / "selection_manifest.json"))

    def test_scored_http_retrieval_schema(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"result": [[{"document": {"id": "42", "contents": '"Title"\nFact.'}, "score": 0.2}]]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            docs = SearchR1Retriever("http://127.0.0.1:{}/retrieve".format(server.server_port)).search("q", 20)
            self.assertEqual(received, [{"queries": ["q"], "topk": 20, "return_scores": True}])
            self.assertEqual(docs[0]["document_id"], "42")
            self.assertEqual(docs[0]["title"], "Title")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_cli_records_complete_and_failed_runs(self):
        from work2.run_baseline import main
        from work2.common import ROOT
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{}')
            source = root / "pilot.jsonl"
            source.write_text(json.dumps(self.sample) + '\n')
            local = root / "local.json"
            local.write_text(json.dumps({"agent_checkpoint": str(model), "retriever_model": str(model),
                                         "corpus_file": str(root / "not_here"), "index_file": str(root / "not_here"),
                                         "retrieval_url": "http://localhost:8000/retrieve"}))
            for fail in (False, True):
                run = root / ("failed" if fail else "complete")
                argv = ["runner", "--local", str(local), "--samples", str(source),
                        "--design", str(ROOT / "configs/diagnostic.json"), "--run-dir", str(run)]
                generator = FakeGenerator(["<answer>SECRET_GOLD</answer>"])
                if fail:
                    generator.generate = mock.Mock(side_effect=RuntimeError("fixture backend failure"))
                with mock.patch("sys.argv", argv), mock.patch("work2.run_baseline.SearchR1Retriever", return_value=FakeRetriever()), \
                        mock.patch("work2.backends.VLLMGenerator", return_value=generator):
                    if fail:
                        with self.assertRaises(RuntimeError):
                            main()
                    else:
                        self.assertEqual(main(), 0)
                self.assertEqual(load_json(run / "run_manifest.json")["status"], "failed" if fail else "completed")
                metrics = load_json(run / "baseline_metrics.json")["overall"]
                self.assertEqual(metrics["recorded_samples"], 1)
                self.assertEqual(metrics["infrastructure_errors"], int(fail))
                self.assertEqual(metrics["em"], None if fail else 1.0)


if __name__ == "__main__":
    unittest.main()

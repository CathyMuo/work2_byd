"""CPU tests for CT1 (interventional credit pilot); no torch/model/GPU."""
import argparse
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from work2 import credit_analysis as ca
from work2 import credit_signals as cs
from work2 import credit_testbed as ct
from work2.common import ROOT, file_hash, load_json, write_json
from work2.rollout import run_sample

PROTOCOL = load_json(ROOT / "configs/credit_v1.json")
AGENT = dict(PROTOCOL["agent"], temperature=0)
RETRIEVAL = PROTOCOL["retrieval"]


class BatchRetriever:
    def __init__(self):
        self.batches = []

    def _docs(self, top_k):
        docs = ["Alpha\nAlpha fact.", "Beta\nBeta text here.", "Gamma\nGamma text here."]
        docs += ["Filler{}\nFiller text {}.".format(i, i) for i in range(4, top_k + 1)]
        return [{"document_id": "d{}".format(i), "title": c.split("\n")[0], "contents": c, "rank": i + 1, "score": 1.0}
                for i, c in enumerate(docs[:top_k])]

    def search(self, query, top_k):
        return self._docs(top_k)

    def search_batch(self, queries, top_k):
        self.batches.append(list(queries))
        return [self._docs(top_k) for _ in queries]


class EvidenceGen:
    """Searches `searches` times, then answers Paris iff the LAST observation shows Alpha's fact."""
    def __init__(self, searches=1):
        self.searches, self.batch_sizes = searches, []

    def encode(self, text):
        return list(text.encode())

    def initial_prompt(self, text):
        return text

    def _one(self, tokens):
        text = bytes(tokens).decode()
        seen = text.count("\n\n<information>")
        if seen < self.searches:
            out = "<search>q{}</search>".format(seen)
        else:
            last = text.rsplit("\n\n<information>", 1)[1]
            out = "<answer>{}</answer>".format("Paris" if "Alpha fact." in last else "London")
        return {"text": out, "token_ids": self.encode(out), "finish_reason": "stop"}

    def generate(self, tokens, limit, seed):
        return self._one(tokens)

    def sample_batch(self, prompts, max_tokens, seeds, temperature, top_p=1.0):
        self.batch_sizes.append(len(prompts))
        return [self._one(p) for p in prompts]


def sample(i=0, dataset="hotpotqa"):
    return {"sample_id": "{}:{}".format(dataset, i), "dataset": dataset, "question": "Where {}?".format(i),
            "gold_answer": "Paris", "gold_supporting_facts": [{"title": "Alpha", "sentence_id": 0, "sentence": "Alpha fact."}]}


class EngineTests(unittest.TestCase):
    def test_batched_matches_sequential_rollout(self):
        gen, s = EvidenceGen(2), sample()
        seq = run_sample(s, gen, BatchRetriever(), AGENT, RETRIEVAL)
        prompt = gen.encode(seq["initial_prompt"])
        r = ct.new_rollout(s, prompt, 0, 1)
        ct.batched_continue([r], gen, ct.LiveSearcher(BatchRetriever(), RETRIEVAL), AGENT, RETRIEVAL, 1.0, 1.0)
        self.assertEqual((r["status"], r["final_answer"], r["search_calls"], r["em"]),
                         (seq["status"], seq["final_answer"], seq["search_calls"], seq["em"]))
        self.assertEqual(len(r["tokens"]), seq["final_prompt_token_count"])
        self.assertEqual(r["queries"], [p["search_query"] for p in seq["prefixes"]])

    def test_spent_budget_carries_over(self):
        gen = EvidenceGen(10)
        r = ct.new_rollout(sample(), gen.encode("Q"), 3, 1)
        ct.batched_continue([r], gen, ct.LiveSearcher(BatchRetriever(), RETRIEVAL), AGENT, RETRIEVAL, 1.0, 1.0)
        self.assertEqual(r["search_calls"], 4)
        self.assertEqual(r["status"], "search_budget_exhausted")

    def test_null_mode_never_calls_retriever(self):
        retr, gen = BatchRetriever(), EvidenceGen(2)
        null = BatchRetriever()._docs(20)[1:4]
        r = ct.new_rollout(sample(), gen.encode("Q"), 0, 1, mode="null", null_docs=null)
        ct.batched_continue([r], gen, ct.LiveSearcher(retr, RETRIEVAL), AGENT, RETRIEVAL, 1.0, 1.0)
        self.assertEqual(retr.batches, [])
        self.assertEqual((r["search_calls"], r["em"]), (2, 0))

    def test_null_documents_exclude_answer_and_facts(self):
        s = dict(sample(), gold_answer="Gamma")
        docs = ct.null_documents(ct.LiveSearcher(BatchRetriever(), RETRIEVAL), "decoy", s, RETRIEVAL)
        self.assertEqual([d["title"] for d in docs], ["Beta", "Filler4", "Filler5"])

    def test_run_question_values_and_segments(self):
        gen, retr = EvidenceGen(1), BatchRetriever()
        row = ct.run_question(sample(), "Where 9?", gen, ct.LiveSearcher(retr, RETRIEVAL), PROTOCOL)
        k = PROTOCOL["sampling"]["k"]
        t = row["trajectory"]
        self.assertEqual((t["search_calls"], t["em"], t["reproduced"]), (1, 1, True))
        self.assertEqual(row["values"]["V"], [[1] * k, [1] * k])
        self.assertEqual(row["values"]["N"], [[0] * k])
        self.assertEqual(row["values"]["C"], [0] * k)
        text = bytes(t["token_ids"][:ct.obs_end(t, 1)]).decode()
        self.assertTrue(text.endswith("</information>\n\n"))
        self.assertTrue(bytes(t["token_ids"][:ct.query_end(t, 1)]).decode().endswith("</search>"))
        self.assertEqual(t["turns"][0]["fact_ids"], [0])
        self.assertEqual(row["branch_rollouts"], 2 * k + k + k)
        self.assertIn(row["branch_rollouts"], gen.batch_sizes)

    def test_drift_raises(self):
        class Drifty(BatchRetriever):
            n = 0
            def search_batch(self, queries, top_k):
                Drifty.n += 1
                docs = super().search_batch(queries, top_k)
                if Drifty.n > 1:
                    docs[0][0] = dict(docs[0][0], contents="Changed\nchanged.")
                return docs
        s = ct.LiveSearcher(Drifty(), RETRIEVAL)
        s.search_many(["a"])
        with self.assertRaises(RuntimeError):
            s.search_many(["a"])


class SignalTests(unittest.TestCase):
    def trajectory(self):
        return ct.run_question(sample(), "Where 9?", EvidenceGen(2), ct.LiveSearcher(BatchRetriever(), RETRIEVAL), PROTOCOL)

    def test_spans(self):
        row = self.trajectory(); t = row["trajectory"]
        enc = lambda x: list(x.encode())
        spans = cs.gold_spans(t, enc, "Paris")
        self.assertEqual(len(spans), 3)
        for s in spans:
            a, b = s["target"]
            self.assertEqual(bytes(s["ids"][a:b]).decode(), "Paris")
            self.assertTrue(bytes(s["ids"][:a]).decode().endswith("<answer>"))
        e = cs.ealr_spans(t)
        self.assertEqual([x["turn"] for x in e], [1, 2])
        rows = cs.masked_rows(e[0])
        self.assertEqual(rows[0][0], e[0]["target"][0] - 1)
        self.assertEqual(len(rows), e[0]["target"][1] - e[0]["target"][0])

    def test_score_record_with_fake_scorer(self):
        class Fake:
            encode = staticmethod(lambda x: list(x.encode()))
            def target_logprobs(self, ids, start, end, rows=None):
                return [-0.5 if not rows else -1.5] * (end - start)
        out = cs.score_record(Fake(), self.trajectory(), "policy")
        self.assertEqual(len(out["gold_mean_logprob"]), 3)
        self.assertEqual(out["ealr"], {"1": 1.0, "2": 1.0})


class AnalysisTests(unittest.TestCase):
    def test_rank_statistics(self):
        self.assertEqual(ca.ranks([3, 1, 1, 2]), [4.0, 1.5, 1.5, 3.0])
        self.assertAlmostEqual(ca.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(ca.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertEqual(ca.auroc([0.9, 0.8, 0.1], [True, True, False]), 1.0)

    def build_run(self, root):
        run = root / "run"; run.mkdir()
        recs = []
        for d in ("hotpotqa", "2wiki"):
            for i in range(8):
                k = 8
                v1 = [1] * (i + 1) + [0] * (k - i - 1)   # varying evidence value across questions
                recs.append({"sample_id": "{}:{}".format(d, i), "dataset": d, "null_eligible": True,
                             "trajectory": {"search_calls": 1, "em": 1, "reproduced": True,
                                            "turns": [{"fact_ids": [0] if i % 2 else [], "answer_hit": bool(i % 2)}]},
                             "values": {"V": [[0] * k, v1], "N": [[0] * k], "C": [0] * k}})
        with (run / "records.jsonl").open("w") as f:
            for r in recs: f.write(json.dumps(r) + "\n")
        pol = [{"sample_id": r["sample_id"], "gold_mean_logprob": [-3.0, -3.0 + 0.1 * int(r["sample_id"].split(":")[1])],
                "ealr": {"1": -0.1 * j}} for j, r in enumerate(recs)]
        with (run / "signals_policy.jsonl").open("w") as f:
            for r in pol: f.write(json.dumps(r) + "\n")
        write_json(run / "run_manifest.json", {"status": "completed", "records_sha256": file_hash(run / "records.jsonl"),
                                               "mc_wall_seconds": 600, "signals": {"policy": {"sha256": file_hash(run / "signals_policy.jsonl")}}})
        return run

    def test_analyze_and_gates(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ca, "git_output", return_value="fixture"):
            report, rows = ca.analyze(self.build_run(Path(tmp)), resamples=50)
            self.assertEqual(report["turns"], 16)
            self.assertTrue(report["signals"]["ig"]["available"])
            self.assertFalse(report["signals"]["trace"]["available"])
            self.assertGreater(report["signals"]["ig"]["vs_E"]["spearman"], 0.5)
            self.assertIn("G_a_truth_reliability_ge_0.6", report["gates"])
            self.assertTrue(report["gates"]["G_c_cost_le_8h_for_600"])
            self.assertIn("CT1", ca.markdown(report))

    def test_tampered_signals_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.build_run(Path(tmp))
            with (run / "signals_policy.jsonl").open("a") as f: f.write("\n")
            with self.assertRaises(ValueError):
                ca.analyze(run, resamples=0)


class RunnerTests(unittest.TestCase):
    def test_prepare_worker_merge(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ct, "git_output", return_value="fixture"):
            root = Path(tmp)
            src = root / "sources.jsonl"
            with src.open("w") as f:
                for d in ("hotpotqa", "2wiki"):
                    for i in range(5):
                        f.write(json.dumps(sample(i, d)) + "\n")
            ct.prepare(argparse.Namespace(protocol=ROOT / "configs/credit_v1.json", sources=src, per_dataset=3, output=root / "plan"))
            plan = load_json(root / "plan" / "plan.json")
            self.assertEqual(len(plan["questions"]), 6)
            self.assertNotEqual(plan["questions"][0]["decoy_sample_id"], plan["questions"][0]["sample_id"])
            local = root / "local.json"; write_json(local, {"agent_checkpoint": str(root), "retrieval_url": "http://x"})
            run_dir = root / "run"; (run_dir / "logs").mkdir(parents=True)
            write_json(run_dir / "run_plan.json", {"plan_path": str(root / "plan" / "plan.json"), "plan_sha256": file_hash(root / "plan" / "plan.json"),
                       "question_ids": [q["sample_id"] for q in plan["questions"]], "gpus": ["0", "1"],
                       "gpu_memory_utilization": .5, "request_timeout": 5, "local_file": str(local),
                       "local_file_sha256": file_hash(local), "repository_commit": "fixture"})
            write_json(run_dir / "run_manifest.json", {"status": "running"})
            for index in range(2):
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(index)}), \
                     mock.patch("work2.backends.VLLMGenerator", side_effect=lambda *a: EvidenceGen(1)), \
                     mock.patch("work2.retrieval.SearchR1Retriever", side_effect=lambda *a: BatchRetriever()):
                    ct.worker(run_dir, index)
            m = ct.merge(run_dir)
            self.assertEqual((m["recorded"], m["trajectory_reproduced"], m["null_eligible"]), (6, 6, 6))


if __name__ == "__main__":
    unittest.main()

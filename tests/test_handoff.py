"""CPU tests for HR1 handoff replay (fake policies, no model/GPU)."""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_audit import raw
from test_credit import BatchRetriever, EvidenceGen
from work2 import handoff as hr
from work2 import handoff_analysis as ha
from work2.common import ROOT, file_hash, load_json, write_json
from work2.credit_testbed import LiveSearcher

PROTOCOL = load_json(ROOT / "configs/handoff_v1.json")


class InstructGen(EvidenceGen):
    """Searches like R but never uses the evidence (always answers London)."""
    def _one(self, tokens):
        out = super()._one(tokens)
        if "<answer>" in out["text"]:
            out = {"text": "<answer>London</answer>", "token_ids": self.encode("<answer>London</answer>"), "finish_reason": "stop"}
        return out


def sample(i=0, d="hotpotqa"):
    return {"sample_id": "{}:{}".format(d, i), "dataset": d, "question": "Where {}?".format(i), "gold_answer": "Paris",
            "gold_supporting_facts": [{"title": "Alpha", "sentence_id": 0, "sentence": "Alpha fact."}], "decoy_question": "decoy"}


class StateTests(unittest.TestCase):
    def test_arms_build_expected_prefixes(self):
        gen = EvidenceGen(2)
        traj = hr.sample_trajectory(sample(), gen, LiveSearcher(BatchRetriever(), PROTOCOL["retrieval"]), PROTOCOL, "R")
        null = gen.encode("\n\n<information>NULL DOCS</information>\n\n")
        real = bytes(hr.state_tokens(traj, 2, "real", null, gen.encode)).decode()
        pl = bytes(hr.state_tokens(traj, 2, "placebo", null, gen.encode)).decode()
        ev = bytes(hr.state_tokens(traj, 2, "evonly", null, gen.encode)).decode()
        self.assertEqual(real.count("Alpha fact."), 2)
        self.assertNotIn("Alpha fact.", pl.split("Question:")[1])
        self.assertEqual(pl.count("NULL DOCS"), 2)
        self.assertEqual(ev.count("Alpha fact."), 2)
        self.assertEqual(ev.count("<search> q0 </search>"), 1)
        self.assertEqual(hr.state_tokens(traj, 0, "placebo", null, gen.encode), traj["token_ids"][:traj["segments"][0]["end"]])
        self.assertEqual(hr.points(traj, [0, 1, 2, 3]), [0, 1, 2])


class PipelineTests(unittest.TestCase):
    def test_three_phase_run_merge_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(hr, "git_output", return_value="fixture"), \
             mock.patch.object(ha, "git_output", return_value="fixture"):
            root = Path(tmp)
            write_json(root / "hp.json", raw("h", 8)); write_json(root / "tw.json", raw("w", 8))
            for f in ("hp.json", "tw.json"):   # make questions look like the fake policies expect
                rows = load_json(root / f)
                for i, r in enumerate(rows):
                    r["question"] = "Where {}?".format(i * 2 if f == "hp.json" else i * 2 + 100)
                write_json(root / f, rows)
            hist = root / "hist.jsonl"; hist.write_text(json.dumps({"sample_id": "hotpotqa:none", "dataset": "hotpotqa", "question": "x"}) + "\n")
            local = root / "local.json"
            write_json(local, {"hotpotqa_file": str(root / "hp.json"), "two_wiki_file": str(root / "tw.json"),
                               "agent_checkpoint": str(root / "r_model"), "instruct_checkpoint": str(root / "instruct_model"),
                               "retrieval_url": "http://x"})
            hr.freeze(argparse.Namespace(protocol=ROOT / "configs/handoff_v1.json", local=local, exclude=[hist], per_dataset=3, output=root / "fresh"))
            samples = root / "fresh" / "samples.jsonl"
            self.assertEqual(len(samples.read_text().splitlines()), 6)
            run_dir = root / "run"; (run_dir / "logs").mkdir(parents=True)
            write_json(run_dir / "run_plan.json", {"protocol": PROTOCOL, "protocol_sha256": hr.sha(PROTOCOL), "samples_path": str(samples),
                       "samples_sha256": file_hash(samples), "gpus": ["0", "1"], "gpu_memory_utilization": .5, "request_timeout": 5,
                       "local_file": str(local), "local_file_sha256": file_hash(local), "repository_commit": "fixture"})
            write_json(run_dir / "run_manifest.json", {"status": "running"})
            factory = lambda path, *a: InstructGen(1) if "instruct" in str(path) else EvidenceGen(1)
            for phase in hr.PHASES:
                for index in range(2):
                    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(index)}), \
                         mock.patch("work2.backends.VLLMGenerator", side_effect=factory), \
                         mock.patch("work2.retrieval.SearchR1Retriever", side_effect=lambda *a: BatchRetriever()):
                        hr.worker(run_dir, index, phase)
            m = hr.merge(run_dir)
            self.assertEqual((m["recorded"], m["template_identical"]), (6, 6))
            report = ha.analyze(run_dir, resamples=50)
            c = report["components"]
            self.assertAlmostEqual(c["gain_end"]["estimate"], 1.0)
            self.assertAlmostEqual(c["utilization"]["estimate"], 1.0)
            self.assertAlmostEqual(c["acquisition"]["estimate"], 0.0)
            self.assertAlmostEqual(c["evidence_free"]["estimate"], 0.0)
            self.assertTrue(report["gates"]["G_A1_protocol_valid"])
            self.assertTrue(report["gates"]["GO"])
            self.assertIn("HR1", ha.markdown(report))

    def test_phase_order_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            samples = run_dir / "s.jsonl"; samples.write_text(json.dumps(sample()) + "\n")
            local = run_dir / "local.json"; write_json(local, {})
            write_json(run_dir / "run_plan.json", {"protocol": PROTOCOL, "protocol_sha256": hr.sha(PROTOCOL), "samples_path": str(samples),
                       "samples_sha256": file_hash(samples), "gpus": ["0"], "local_file": str(local), "local_file_sha256": file_hash(local),
                       "repository_commit": "fixture"})
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), mock.patch.object(hr, "git_output", return_value="fixture"):
                with self.assertRaises(FileNotFoundError):
                    hr.worker(run_dir, 0, "i2")


if __name__ == "__main__":
    unittest.main()
"""CPU tests for HR2 (handoff replay on v0.3 with the evonly_placebo arm; fake policies, no GPU)."""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_audit import raw
from test_credit import BatchRetriever, EvidenceGen
from test_handoff import InstructGen, sample
from work2 import handoff as hr
from work2 import handoff_analysis as ha
from work2.common import ROOT, file_hash, load_json, write_json
from work2.credit_testbed import LiveSearcher

PROTOCOL = load_json(ROOT / "configs/handoff_v2.json")


def pipeline(root, r_name, samples=None):
    """Freeze (once) and run all three phases for one R checkpoint; returns (run_dir, samples)."""
    local = root / "local_{}.json".format(r_name)
    if samples is None:
        write_json(root / "hp.json", raw("h", 8)); write_json(root / "tw.json", raw("w", 8))
        for f in ("hp.json", "tw.json"):
            rows = load_json(root / f)
            for i, r in enumerate(rows):
                r["question"] = "Where {}?".format(i * 2 if f == "hp.json" else i * 2 + 100)
            write_json(root / f, rows)
    write_json(local, {"hotpotqa_file": str(root / "hp.json"), "two_wiki_file": str(root / "tw.json"),
                       "agent_checkpoint": str(root / r_name), "instruct_checkpoint": str(root / "instruct_model"),
                       "retrieval_url": "http://x"})
    if samples is None:
        hist = root / "hist.jsonl"
        hist.write_text(json.dumps({"sample_id": "hotpotqa:none", "dataset": "hotpotqa", "question": "x"}) + "\n")
        hr.freeze(argparse.Namespace(protocol=ROOT / "configs/handoff_v2.json", local=local, exclude=[hist], per_dataset=3,
                                     output=root / "fresh"))
        samples = root / "fresh" / "samples.jsonl"
    run_dir = root / ("run_" + r_name); (run_dir / "logs").mkdir(parents=True)
    write_json(run_dir / "run_plan.json", {"protocol": PROTOCOL, "protocol_sha256": hr.sha(PROTOCOL), "samples_path": str(samples),
               "samples_sha256": file_hash(samples), "gpus": ["0", "1"], "gpu_memory_utilization": .5, "request_timeout": 5,
               "local_file": str(local), "local_file_sha256": file_hash(local), "repository_commit": "fixture",
               "assets": {"agent_checkpoint": {"path": r_name}, "instruct_checkpoint": {"path": "instruct_model"}}})
    write_json(run_dir / "run_manifest.json", {"status": "running"})
    factory = lambda path, *a: InstructGen(1) if "instruct" in str(path) else EvidenceGen(1)
    for phase in hr.PHASES:
        for index in range(2):
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(index)}), \
                 mock.patch("work2.backends.VLLMGenerator", side_effect=factory), \
                 mock.patch("work2.retrieval.SearchR1Retriever", side_effect=lambda *a: BatchRetriever()):
                hr.worker(run_dir, index, phase)
    hr.merge(run_dir)
    return run_dir, samples


class HR2Tests(unittest.TestCase):
    def test_protocol_and_evonly_placebo_prefix(self):
        self.assertEqual(hr.arms_of(PROTOCOL), ("real", "placebo", "evonly", "evonly_placebo"))
        self.assertEqual(hr.arms_of(load_json(ROOT / "configs/handoff_v1.json")), hr.ARMS)
        gen = EvidenceGen(2)
        traj = hr.sample_trajectory(sample(), gen, LiveSearcher(BatchRetriever(), PROTOCOL["retrieval"]), PROTOCOL, "R")
        null = gen.encode("\n\n<information>NULL DOCS</information>\n\n")
        evp = bytes(hr.state_tokens(traj, 2, "evonly_placebo", null, gen.encode)).decode()
        self.assertNotIn("Alpha fact.", evp.split("Question:")[1])
        self.assertEqual(evp.count("NULL DOCS"), 2)
        self.assertEqual(evp.count("<search> q1 </search>"), 1)

    def test_bad_arm_set_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            write_json(path, dict(PROTOCOL, arm_set=["placebo", "real"]))
            with self.assertRaises(ValueError):
                hr.load_protocol(path)

    def test_run_analysis_and_checkpoint_comparison(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(hr, "git_output", return_value="fixture"), \
             mock.patch.object(ha, "git_output", return_value="fixture"):
            root = Path(tmp)
            run_a, samples = pipeline(root, "r_ppo")
            report = ha.analyze(run_a, resamples=50)
            c, g = report["components"], report["gates"]
            self.assertAlmostEqual(c["gain_end"]["estimate"], 1.0)
            self.assertAlmostEqual(c["evidence_free_evonly_placebo"]["estimate"], 0.0)
            self.assertEqual(g["evidence_free_primary"], "evidence_free_evonly_placebo")
            self.assertAlmostEqual(g["G_A1_detail"]["evonly_minus_evonly_placebo_R_end"]["estimate"], 1.0)
            self.assertTrue(g["G_A1_protocol_valid"])
            self.assertIn("HR2", ha.markdown(report))
            self.assertIn("evonly_placebo", ha.markdown(report))
            run_b, _ = pipeline(root, "r_grpo", samples)
            cmp = ha.compare(run_a, run_b, resamples=50)
            self.assertEqual(cmp["questions"], 6)
            self.assertAlmostEqual(cmp["share_difference_a_minus_b"]["utilization"]["estimate"], 0.0)
            self.assertFalse(cmp["G_A3_iii"])
            self.assertIn("G_A3_iii", ha.compare_markdown(cmp))
            with self.assertRaises(ValueError):
                ha.compare(run_a, run_a, resamples=10)

    def test_freeze_excludes_hr1_and_u1_history(self):
        self.assertIn("outputs/hr1_fresh/samples.jsonl", hr.DEFAULT_HISTORY)
        self.assertIn("outputs/u1_fresh/samples.jsonl", hr.DEFAULT_HISTORY)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "hp.json", raw("h", 6)); write_json(root / "tw.json", raw("w", 6))
            local = root / "local.json"
            write_json(local, {"hotpotqa_file": str(root / "hp.json"), "two_wiki_file": str(root / "tw.json")})
            u1 = root / "u1.jsonl"   # U1 history mixes FlashRAG datasets with HotpotQA
            u1.write_text("".join(json.dumps(r) + "\n" for r in (
                {"sample_id": "nq:1", "dataset": "nq", "question": "h question 0?"},
                {"sample_id": "hotpotqa:h1", "dataset": "hotpotqa", "question": "h question 1?"})))
            hr.freeze(argparse.Namespace(protocol=ROOT / "configs/handoff_v2.json", local=local, exclude=[u1], per_dataset=2,
                                         output=root / "fresh"))
            chosen = [json.loads(x)["question"] for x in (root / "fresh" / "samples.jsonl").read_text().splitlines()]
            self.assertEqual(len(chosen), 4)
            self.assertNotIn("h question 0?", chosen)
            self.assertNotIn("h question 1?", chosen)


if __name__ == "__main__":
    unittest.main()

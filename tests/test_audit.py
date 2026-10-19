"""CPU tests for AC1: fresh-question freezing, E0 source workers, reflect arms, verdicts."""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_pcr import AGENT, RETRIEVAL, DocRetriever, OrderGenerator, source
from work2 import audit_confirm as ac
from work2 import pcr_analysis
from work2.common import ROOT, file_hash, load_json, write_json
from work2.placebo import ARM_GROUPS, REFLECT_PROMPT, run_state


def raw(prefix, n, dup_question=None):
    rows = []
    for i in range(n):
        q = dup_question if (dup_question and i == 0) else "{} question {}?".format(prefix, i)
        rows.append({"_id": "{}{}".format(prefix, i), "question": q, "answer": "Paris",
                     "context": [["Alpha", ["Alpha fact."]]], "supporting_facts": [["Alpha", 0]]})
    return rows


class FreezeTests(unittest.TestCase):
    def test_freeze_excludes_history_by_id_and_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "hp.json", raw("h", 12)); write_json(root / "tw.json", raw("w", 12, dup_question="h question 5?"))
            local = root / "local.json"; write_json(local, {"hotpotqa_file": str(root / "hp.json"), "two_wiki_file": str(root / "tw.json")})
            hist = root / "hist.jsonl"
            with hist.open("w") as f:
                f.write(json.dumps({"sample_id": "hotpotqa:h0", "dataset": "hotpotqa", "question": "x"}) + "\n")
                f.write(json.dumps({"sample_id": "2wiki:other", "dataset": "2wiki", "question": "w question 3?"}) + "\n")
            ac.freeze(argparse.Namespace(protocol=ROOT / "configs/audit_confirm_v1.json", local=local, exclude=[hist],
                                         per_dataset=5, output=root / "fresh"))
            rows = [json.loads(l) for l in (root / "fresh" / "samples.jsonl").read_text().splitlines()]
            ids = [r["sample_id"] for r in rows]
            self.assertEqual(len(rows), 10)
            self.assertNotIn("hotpotqa:h0", ids)
            self.assertNotIn("w question 3?", [r["question"] for r in rows])
            self.assertEqual(len({r["question"] for r in rows}), 10)
            with self.assertRaises(ValueError):
                ac.freeze(argparse.Namespace(protocol=ROOT / "configs/audit_confirm_v1.json", local=local, exclude=[hist],
                                             per_dataset=5, output=root / "fresh"))


class SourcesTests(unittest.TestCase):
    def test_worker_merge_produce_pcr_compatible_sources(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ac, "git_output", return_value="fixture"):
            root = Path(tmp)
            samples = root / "samples.jsonl"
            with samples.open("w") as f:
                for d in ("hotpotqa", "2wiki"):
                    for i in range(3):
                        f.write(json.dumps({"sample_id": "{}:{}".format(d, i), "dataset": d, "question": "Where {}?".format(i),
                                            "gold_answer": "Paris", "gold_supporting_facts": [{"title": "Alpha", "sentence_id": 0, "sentence": "Alpha fact."}]}) + "\n")
            local = root / "local.json"; write_json(local, {"agent_checkpoint": str(root), "retrieval_url": "http://x"})
            protocol = load_json(ROOT / "configs/audit_confirm_v1.json")
            run_dir = root / "run"; (run_dir / "logs").mkdir(parents=True)
            write_json(run_dir / "run_plan.json", {"protocol": protocol, "protocol_sha256": ac.sha(protocol), "samples_path": str(samples),
                       "samples_sha256": file_hash(samples), "gpus": ["0", "1"], "gpu_memory_utilization": .5, "request_timeout": 5,
                       "local_file": str(local), "local_file_sha256": file_hash(local), "repository_commit": "fixture"})
            write_json(run_dir / "run_manifest.json", {"status": "running"})
            for index in range(2):
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(index)}), \
                     mock.patch("work2.backends.VLLMGenerator", side_effect=lambda *a: OrderGenerator()), \
                     mock.patch("work2.retrieval.SearchR1Retriever", side_effect=lambda *a: DocRetriever()):
                    ac.worker(run_dir, index)
            ac.merge(run_dir)
            from work2.pcr import source_rows
            rows = source_rows(run_dir / "sources.jsonl")
            self.assertEqual([r["sample_id"] for r in rows], [json.loads(l)["sample_id"] for l in samples.read_text().splitlines()])
            self.assertEqual(load_json(run_dir / "run_manifest.json")["clean_em"], 1.0)


class ReflectTests(unittest.TestCase):
    def test_reflect_arms_and_matched_placebo(self):
        row = run_state(source(), 1, ARM_GROUPS["audit"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)
        arms = row["arms"]
        self.assertEqual(len(arms), 20)
        r, n = arms["reflect"]["metadata"], arms["neutral_reflect"]["metadata"]
        self.assertEqual(r["instruction"], REFLECT_PROMPT)
        self.assertEqual(r["instruction_tokens"], n["instruction_tokens"])
        self.assertEqual(n["matched_tokens"], r["instruction_tokens"])
        self.assertEqual(arms["perm1_reflect"]["metadata"]["order"], arms["perm_1"]["metadata"]["order"])

    def test_recovery_analysis_includes_reflect(self):
        states = [run_state(source(i, d), 1, ARM_GROUPS["audit"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)
                  for d in ("hotpotqa", "2wiki") for i in range(0, 12, 2)]
        derived = [pcr_analysis.derive(s) for s in states]
        rec = pcr_analysis.recovery(derived, "sem", 20)
        self.assertIn("reflect", rec["selected_fragile"]["methods"])
        self.assertIn("repair_minus_reperturb", rec["selected_fragile"]["methods"]["review"])
        self.assertIn("reflect", rec["natural_states"])


class VerdictTests(unittest.TestCase):
    def test_verdicts(self):
        pcr = {"recovery": {"semantic": {"available": True, "selected_fragile": {"degraded_states": 40, "methods": {
                   "review": {"placebo_explained_fraction": 1.3, "direction_reversed": False},
                   "reflect": {"placebo_explained_fraction": 0.1, "direction_reversed": False}}}}},
               "deletion": {"semantic": {"available": True, "matched_states": 300, "all_matched": {"support_minus_nonsupport": {"ci95": [-0.14, -0.05]}}}},
               "luck": {"semantic": {"lucky_share_of_accuracy": {}, "clean_em": {}}}}
        sig = lambda s, hi: {"available": True, "vs_E": {"spearman": s, "ci95": [s - .05, hi]}}
        ct1 = {"reliability": {"E": {"spearman_brown": .88}, "A": {"spearman_brown": .5}}, "truth": {"mean_A": 0, "mean_E": .05},
               "signals": {}, "exploratory": {"informative_signals": {"ig": sig(.08, .16), "ealr": sig(.09, .18), "fact_hit": sig(.17, .26)}}}
        v = ac.verdicts(pcr, ct1)
        self.assertEqual(v["H1_recovery_explained_by_placebo"]["per_method"], {"review": True, "reflect": False})
        self.assertTrue(v["H1_recovery_explained_by_placebo"]["confirmed"])
        self.assertTrue(v["H2_support_deletion_hurts_more"]["confirmed"])
        self.assertTrue(v["H3_evidence_signals_do_not_track_E"]["confirmed"])
        ct1["exploratory"]["informative_signals"]["fact_hit"] = sig(.35, .45)
        self.assertFalse(ac.verdicts(pcr, ct1)["H3_evidence_signals_do_not_track_E"]["confirmed"])
        self.assertEqual(ac.verdicts(pcr, ct1)["n_confirmed"], 2)


if __name__ == "__main__":
    unittest.main()

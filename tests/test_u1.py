"""CPU tests for U1: multi-answer scoring, FlashRAG data, official-prompt replay, h1 arms, freeze, summary."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_credit import BatchRetriever, EvidenceGen
from test_pcr import AGENT, RETRIEVAL, DocRetriever, OrderGenerator
from work2 import credit_testbed as ct
from work2 import pcr, u1
from work2.common import ROOT, file_hash, load_json, write_json
from work2.datasets import canonicalize_flashrag
from work2.metrics import answer_scores, primary_gold
from work2.pcr_analysis import derive, recovery
from work2.placebo import ARM_GROUPS, contains_answer, run_state
from work2.rollout import OFFICIAL_PROMPT_VERSION, run_sample


def nq_sample(i, gold=("Paris", "City of Paris")):
    return {"sample_id": "nq:{}".format(i), "dataset": "nq", "question": "Where {}?".format(i),
            "gold_answer": list(gold), "gold_supporting_facts": []}


class ScoringTests(unittest.TestCase):
    def test_alias_lists(self):
        self.assertEqual(answer_scores("city of paris", ["Paris", "City of Paris"])["em"], 1)
        self.assertEqual(answer_scores("Paris France", ["Paris", "Lyon"])["em"], 0)
        self.assertGreater(answer_scores("Paris France", ["Paris", "Lyon"])["f1"], 0.6)
        self.assertEqual(answer_scores("Paris", "Paris")["em"], 1)
        self.assertEqual(primary_gold(["A", "B"]), "A")
        self.assertTrue(contains_answer({"contents": "T\nIt is in the City of Paris."}, ["Lyon", "City of Paris"]))
        self.assertFalse(contains_answer({"contents": "T\nLyon."}, ["Paris"]))

    def test_flashrag_canonicalization(self):
        r = canonicalize_flashrag({"id": "q7", "question": " Who? ", "golden_answers": ["A", "A", " B "]}, "nq")
        self.assertEqual((r["sample_id"], r["question"], r["gold_answer"], r["gold_supporting_facts"]), ("nq:q7", "Who?", ["A", "B"], []))
        with self.assertRaises(ValueError):
            canonicalize_flashrag({"id": "x", "question": "Q", "golden_answers": []}, "nq")


class ReplayTests(unittest.TestCase):
    def official_source(self, i=0):
        agent = dict(AGENT, prompt_variant="official")
        return run_sample(nq_sample(i), OrderGenerator(), DocRetriever(), agent, RETRIEVAL)

    def test_official_prompt_sources_replay_with_h1_arms(self):
        src = self.official_source()
        self.assertEqual(src["prompt_version"], OFFICIAL_PROMPT_VERSION)
        self.assertEqual(src["em"], 1)                      # alias list scored
        row = run_state(src, 1, ARM_GROUPS["h1"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)
        self.assertEqual(set(row["arms"]), set(ARM_GROUPS["h1"]))
        self.assertEqual(row["deletion_eligibility"]["drop_support"]["reason"], "no_support_slot")
        rec = recovery([derive(row)], "sem", 10)
        self.assertTrue(rec["available"])
        self.assertIn("reflect", rec["selected_fragile"]["methods"])

    def test_pcr_prepare_multi_dataset_with_short_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources.jsonl"
            agent = dict(AGENT, prompt_variant="official")
            rows = [run_sample(nq_sample(i), OrderGenerator(), DocRetriever(), agent, RETRIEVAL) for i in range(4)]
            rows += [run_sample(dict(nq_sample(10 + i), sample_id="bamboogle:{}".format(i), dataset="bamboogle"),
                                OrderGenerator(), DocRetriever(), agent, RETRIEVAL) for i in range(2)]
            sources.write_text("".join(json.dumps(r) + "\n" for r in rows))
            pcr.prepare(argparse.Namespace(protocol=ROOT / "configs/pcr_u1_official.json", sources=sources, output=root / "plan", per_dataset=3))
            plan = load_json(root / "plan" / "plan.json")
            self.assertEqual(plan["datasets"], ["nq", "bamboogle"])
            self.assertEqual(plan["short_datasets"], {"bamboogle": 2})
            self.assertEqual(len(plan["question_ids"]), 5)
            mixed = root / "mixed.jsonl"
            bounded = run_sample(nq_sample(99), OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)
            mixed.write_text("".join(json.dumps(r) + "\n" for r in rows + [bounded]))
            with self.assertRaises(ValueError):
                pcr.source_rows(mixed)


class FreezeAndSummaryTests(unittest.TestCase):
    def test_freeze_reuses_ac1_and_adds_flashrag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ac1 = root / "ac1"; ac1.mkdir()
            rows = [{"sample_id": "hotpotqa:{}".format(i), "dataset": "hotpotqa", "question": "Hq {}?".format(i), "gold_answer": "A",
                     "gold_supporting_facts": []} for i in range(3)]
            (ac1 / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            write_json(ac1 / "selection_manifest.json", {"samples_sha256": file_hash(ac1 / "samples.jsonl")})
            nq = root / "nq.jsonl"
            nq.write_text("".join(json.dumps({"id": str(i), "question": "Hq 0?" if i == 0 else "N {}?".format(i), "golden_answers": ["x"]}) + "\n" for i in range(6)))
            local = root / "local.json"; write_json(local, {"nq_file": str(nq)})
            proto = dict(load_json(ROOT / "configs/u1_protocol.json"))
            proto["new_datasets"] = {"nq": {"local_key": "nq_file", "per_dataset": 4}, "musique": {"local_key": "musique_file", "per_dataset": 2}}
            pfile = root / "proto.json"; write_json(pfile, proto)
            hist = root / "hist.jsonl"; hist.write_text(json.dumps({"sample_id": "hotpotqa:old", "dataset": "hotpotqa", "question": "N 1?"}) + "\n")
            u1.freeze(argparse.Namespace(protocol=pfile, local=local, ac1_fresh=ac1, exclude=[hist], output=root / "fresh"))
            m = load_json(root / "fresh" / "selection_manifest.json")
            got = [json.loads(l) for l in (root / "fresh" / "samples.jsonl").read_text().splitlines()]
            self.assertEqual(m["counts"], {"hotpotqa": 3, "2wiki": 0, "nq": 4})
            self.assertIn("musique", m["missing_datasets"])
            qs = [r["question"] for r in got if r["dataset"] == "nq"]
            self.assertNotIn("Hq 0?", qs)          # cross-dataset duplicate of an AC1 question
            self.assertNotIn("N 1?", qs)           # historical question text
            self.assertIsInstance(got[-1]["gold_answer"], list)

    def test_summary_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(u1, "git_output", return_value="fixture"):
            root = Path(tmp) / "u1"
            d = root / "v03_7b_ppo"
            for sub in ("sources", "pcr", "ct"):
                (d / sub).mkdir(parents=True)
            agent = dict(AGENT, prompt_variant="official")
            srcs = [run_sample(nq_sample(i), OrderGenerator(), DocRetriever(), agent, RETRIEVAL) for i in range(0, 24, 2)]
            (d / "sources" / "sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in srcs))
            states = [run_state(s, 1, ARM_GROUPS["h1"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL) for s in srcs]
            (d / "pcr" / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
            proto = load_json(ROOT / "configs/credit_u1_official.json")
            proto["sampling"]["k"] = 4
            recs = [ct.run_question(nq_sample(i, ("Paris",)), "decoy", EvidenceGen(1), ct.LiveSearcher(BatchRetriever(), RETRIEVAL), proto)
                    for i in range(12)]
            (d / "ct" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            sig = [{"sample_id": r["sample_id"], "gold_mean_logprob": [-3.0, -3.0 + .1 * i], "ealr": {"1": .05 * (i % 3)}}
                   for i, r in enumerate(recs)]
            (d / "ct" / "signals_policy.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sig))
            roster = Path(tmp) / "roster.json"; write_json(roster, {"agents": [{"label": "v03_7b_ppo"}, {"label": "absent"}]})
            u1.summary(argparse.Namespace(roster=roster, root=root, resamples=40, output=root / "u1_summary.json"))
            r = load_json(root / "u1_summary.json")
            self.assertEqual(r["included_agents"], ["v03_7b_ppo"])
            a = r["agents"][0]
            self.assertEqual(a["clean_em"]["all"], 1.0)
            self.assertIn("nq", a["by_dataset"])
            self.assertTrue(a["pooled"]["H1"]["available"] if "available" in a["pooled"]["H1"] else True)
            self.assertIn("U1", (root / "u1_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

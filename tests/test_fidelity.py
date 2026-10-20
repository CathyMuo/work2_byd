"""CPU tests for FC1: official prompt variant and the fidelity summary."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_credit import EvidenceGen
from test_pcr import DocRetriever
from work2 import audit_confirm as ac
from work2 import fidelity
from work2.common import ROOT, file_hash, load_json, write_json
from work2.rollout import OFFICIAL_PROMPT_VERSION, PROMPT_VERSION, question_prompt, run_sample


class PromptTests(unittest.TestCase):
    def test_official_template_is_verbatim_search_r1(self):
        text = question_prompt("Who?", 4, "official")
        self.assertIn("You can search as many times as your want.", text)
        self.assertIn("you can directly provide the answer inside <answer> and </answer>", text)
        self.assertNotIn("at most", text)
        self.assertTrue(text.endswith("Question: Who?\n"))
        self.assertIn("You may make at most 4 search calls.", question_prompt("Who?", 4))
        with self.assertRaises(ValueError):
            question_prompt("Who?", 4, "other")

    def test_run_sample_records_prompt_version(self):
        sample = {"sample_id": "hotpotqa:1", "dataset": "hotpotqa", "question": "Where 1?", "gold_answer": "Paris",
                  "gold_supporting_facts": [{"title": "Alpha", "sentence_id": 0, "sentence": "Alpha fact."}]}
        agent = dict(load_json(ROOT / "configs/fidelity_official.json")["agent"])
        row = run_sample(sample, EvidenceGen(1), DocRetriever(), agent, {"visible_top_k": 3, "candidate_top_k": 20})
        self.assertEqual(row["prompt_version"], OFFICIAL_PROMPT_VERSION)
        self.assertIn("as many times as your want", row["initial_prompt"])
        agent["prompt_variant"] = "bounded"
        self.assertEqual(run_sample(sample, EvidenceGen(1), DocRetriever(), agent, {"visible_top_k": 3, "candidate_top_k": 20})["prompt_version"], PROMPT_VERSION)

    def test_fc1_configs_accepted_and_differ_only_in_variant(self):
        b, o = (ac.load_protocol(ROOT / "configs/fidelity_{}.json".format(v)) for v in ("bounded", "official"))
        strip = lambda p: {k: v for k, v in p.items() if k != "scope"}
        self.assertEqual({**strip(b), "agent": {**b["agent"], "prompt_variant": None}},
                         {**strip(o), "agent": {**o["agent"], "prompt_variant": None}})


class SummaryTests(unittest.TestCase):
    def make_run(self, root, name, ems, variant="bounded"):
        d = root / name; d.mkdir()
        with (d / "sources.jsonl").open("w") as f:
            for i, em in enumerate(ems):
                ds = "hotpotqa" if i % 2 == 0 else "2wiki"
                f.write(json.dumps({"sample_id": "{}:{}".format(ds, i), "dataset": ds, "em": em, "status": "answered",
                                    "search_calls": 1, "generated_tokens": 10, "prompt_version": variant}) + "\n")
        write_json(d / "run_manifest.json", {"status": "completed", "sources_sha256": file_hash(d / "sources.jsonl")})
        write_json(d / "run_plan.json", {"assets": {"agent_checkpoint": {"path": "/m/" + name}},
                                         "protocol": {"agent": {"prompt_variant": variant}}})
        return d

    def test_summary_and_pairs(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fidelity, "git_output", return_value="fixture"):
            root = Path(tmp)
            a = self.make_run(root, "a", [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], "official")
            b = self.make_run(root, "b", [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
            r = fidelity.summarize(["a_off=" + str(a), "b_bnd=" + str(b)], ["a_off-b_bnd"], 200)
            self.assertEqual(r["shared_questions"], 12)
            self.assertAlmostEqual(r["runs"]["a_off"]["all"]["estimate"], 11 / 12)
            self.assertEqual(r["runs"]["a_off"]["prompt_variant"], "official")
            p = r["pairs"]["a_off-b_bnd"]
            self.assertAlmostEqual(p["estimate"], (11 - 3) / 12)
            self.assertTrue(p["ge_5pp_ci_lo_gt0"])
            self.assertIn("FC1", fidelity.markdown(r))
            with self.assertRaises(ValueError):
                fidelity.summarize(["a_off=" + str(a)], ["a_off-zzz"], 10)


if __name__ == "__main__":
    unittest.main()

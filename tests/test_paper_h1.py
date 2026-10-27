"""CPU tests for the pooled H1 paper table (synthetic PCR states)."""
import json
import tempfile
import unittest
from pathlib import Path

from work2 import paper_h1
from work2.common import file_hash, write_json

ARMS = ("clean_a", "clean_b", "perm_1", "perm_2", "perm1_review", "perm1_neutral", "perm1_budget2",
        "perm1_reflect", "perm1_neutral_reflect")


def state(i, dataset, outcome):
    arms = {a: {"em": outcome.get(a, outcome["clean_a"]), "final_answer": "Paris" if outcome.get(a, outcome["clean_a"]) else "Rome",
                "status": "answered"} for a in ARMS}
    return {"sample_id": "{}:{}".format(dataset, i), "dataset": dataset, "target_turn": 1, "gold_answer": "Paris",
            "known_fact_ids": [], "candidate_fact_ids": [], "target_visible_fact_ids": [], "source_search_calls": 2,
            "deletion_eligibility": {}, "arms": arms}


def write_run(d, states):
    d.mkdir(parents=True)
    (d / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    write_json(d / "run_manifest.json", {"status": "completed", "states_sha256": file_hash(d / "states.jsonl")})


class PooledTests(unittest.TestCase):
    def test_per_agent_and_pooled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # degraded states: clean right, perm_1 wrong; review repairs exactly as often as perm_2.
            s = []
            for i in range(8):
                rep = int(i % 2 == 0)
                s.append(state(i, "hotpotqa" if i < 4 else "nq",
                               {"clean_a": 1, "perm_1": 0, "perm_2": rep, "perm1_review": rep, "perm1_neutral": 0,
                                "perm1_reflect": 1, "perm1_neutral_reflect": rep, "perm1_budget2": 0}))
            s += [state(10 + i, "hotpotqa" if i < 2 else "nq", {"clean_a": 1}) for i in range(4)]   # stable
            s += [state(20 + i, "hotpotqa" if i < 2 else "nq", {"clean_a": 0, "perm_1": int(i == 0)}) for i in range(4)]
            write_run(root / "a1" / "pcr", s)
            write_run(root / "a2" / "pcr", s)
            r = paper_h1.analyze(root, ["a1", "a2", "missing"], 50)
            b = r["agents"]["a1"]["sem"]
            self.assertEqual((b["degraded"], b["stable"], b["clean_wrong"]), (8, 4, 4))
            self.assertAlmostEqual(b["methods"]["review"]["minus_P1"]["estimate"], 0.0)
            self.assertAlmostEqual(b["methods"]["reflect"]["minus_P2"]["estimate"], 0.5)
            self.assertAlmostEqual(b["noise"]["perm1_rescue_on_clean_wrong"], 0.25)
            self.assertIsNone(r["agents"]["missing"])
            self.assertEqual(r["pooled"]["sem"]["degraded"], 16)
            self.assertIn("**合并**", paper_h1.markdown(r))


if __name__ == "__main__":
    unittest.main()

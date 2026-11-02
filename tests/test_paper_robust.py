"""CPU tests for the exploratory H1 robustness cuts (synthetic PCR states)."""
import json
import tempfile
import unittest
from pathlib import Path

from work2 import paper_robust
from work2.common import file_hash, write_json

ARMS = ("clean_a", "clean_b", "perm_1", "perm_2", "perm1_review", "perm1_neutral", "perm1_budget2",
        "perm1_reflect", "perm1_neutral_reflect")


def state(i, dataset, outcome, **extra):
    arms = {a: {"em": outcome.get(a, outcome["clean_a"]), "final_answer": "Paris" if outcome.get(a, outcome["clean_a"]) else "Rome",
                "status": "answered"} for a in ARMS}
    return dict({"sample_id": "{}:{}".format(dataset, i), "dataset": dataset, "target_turn": 1, "gold_answer": "Paris",
                 "known_fact_ids": [], "candidate_fact_ids": [], "target_visible_fact_ids": [], "source_search_calls": 2,
                 "deletion_eligibility": {}, "arms": arms}, **extra)


def write_run(d, states, mismatches=0):
    d.mkdir(parents=True)
    (d / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    write_json(d / "run_manifest.json", {"status": "completed", "states_sha256": file_hash(d / "states.jsonl"),
                                         "control_mismatch_states": mismatches})


def rows():
    s = []
    for i in range(8):   # degraded: review repairs all of them, its filler half
        s.append(state(i, "hotpotqa" if i < 4 else "nq", {"clean_a": 1, "perm_1": 0, "perm_2": 0, "perm1_review": 1, "perm1_neutral": int(i % 2 == 0),
                                                          "perm1_reflect": 0, "perm1_neutral_reflect": 0}))
    for i in range(8):   # stable: review breaks a quarter of them, its filler none
        s.append(state(10 + i, "hotpotqa" if i < 4 else "nq", {"clean_a": 1, "perm1_review": int(i % 4 != 0)}))
    for i in range(4):   # clean wrong: review and filler both stay wrong
        s.append(state(20 + i, "hotpotqa" if i < 2 else "nq", {"clean_a": 0}))
    return s


class RobustTests(unittest.TestCase):
    def test_boot_keeps_a_question_together(self):
        r = [{"dataset": "a", "sample_id": "q1"}, {"dataset": "a", "sample_id": "q1"}, {"dataset": "a", "sample_id": "q2"}]
        e = paper_robust.boot(r, lambda x: 1.0, 50)
        self.assertEqual((e["n_states"], e["n_questions"], e["estimate"]), (3, 2, 1.0))
        self.assertEqual(e["ci95"], [1.0, 1.0])
        self.assertIsNone(paper_robust.boot([], lambda x: 1.0, 50))

    def test_labels(self):
        self.assertEqual([paper_robust.size_of(x) for x in ("zs_llama32_3b", "v03_14b_it_grpo", "research_32b", "zs_qwen25_7b")], [3, 14, 32, 7])
        self.assertEqual([paper_robust.family_of(x) for x in ("v01_7b_ppo", "research_7b", "stepsearch_7b")], ["Search-R1", "ReSearch", "other"])

    def test_cuts_and_put_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "v03_7b_grpo" / "pcr", rows())
            drifted = state(30, "nq", {"clean_a": 1, "perm_1": 0, "perm1_review": 0, "perm1_neutral": 1}, controls_exact=False,
                            control_mismatch={"arms": {}, "controls_agree": True})
            write_run(root / "research_32b" / "pcr", rows() + [drifted], mismatches=1)
            r = paper_robust.analyze(root, ["v03_7b_grpo", "research_32b", "missing_3b"], 50)
            self.assertEqual(r["missing"], ["missing_3b"])
            b = r["agents"]["v03_7b_grpo"]
            self.assertEqual((b["degraded"], b["stable"], b["clean_wrong"]), (8, 8, 4))
            m = b["methods"]["review"]
            self.assertAlmostEqual(m["repair_minus_P2"]["estimate"], 0.5)
            self.assertAlmostEqual(m["repair_minus_P1"]["estimate"], 1.0)
            self.assertAlmostEqual(m["harm_minus_P2"]["estimate"], 0.25)
            self.assertAlmostEqual(m["rescue_minus_P2"]["estimate"], 0.0)
            self.assertAlmostEqual(m["all_minus_P2"]["estimate"], (4 - 2) / 20)
            self.assertEqual(r["pooled"]["sem"]["states"], 40)
            self.assertEqual(sorted(r["by_size"]), ["32B", "<=7B"])
            self.assertEqual(r["by_size"]["32B"]["states"], 20)
            self.assertEqual(sorted(r["by_family"]), ["ReSearch", "Search-R1"])
            self.assertEqual(r["leave_one_out"]["research_32b"]["states"], 20)
            self.assertEqual(r["mismatch"]["per_agent"], {"research_32b": {"states": 1, "degraded": 1, "stable": 0, "clean_wrong": 0, "controls_agree": 1}})
            self.assertEqual(r["mismatch"]["pooled_put_back"]["states"], 41)
            self.assertEqual(r["mismatch"]["agents_put_back"]["research_32b"]["degraded"], 9)
            text = paper_robust.markdown(r)
            self.assertIn("去掉 research_32b", text)
            self.assertIn("E0 不复现的状态放回去", text)


if __name__ == "__main__":
    unittest.main()

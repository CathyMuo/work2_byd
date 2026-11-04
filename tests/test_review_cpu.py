"""CPU tests for work2/review_cpu.py (AS2 amendment 1): four-way tables, weighting, alert vs review, question-level policy,
semantic-view audit. Synthetic runs only."""
import json
import tempfile
import unittest
from pathlib import Path

from work2 import review_cpu
from work2.common import file_hash, write_json
from work2.pcr import rank_key

H1_ARMS = ("clean_a", "clean_b", "perm_1", "perm_2", "perm1_review", "perm1_neutral", "perm1_reflect", "perm1_neutral_reflect")


def arm(em, answer=None, tokens=10, searches=1):
    return {"em": em, "status": "answered", "final_answer": answer or ("Paris" if em else "Rome"), "continuation_sha256": "x",
            "generated_tokens_after_target": tokens, "search_calls_after_target": searches, "metadata": {}}


def state(sid, ds, arms, turn=1):
    return {"sample_id": sid, "dataset": ds, "target_turn": turn, "question": "Where?", "gold_answer": "Paris", "controls_exact": True,
            "known_fact_ids": [], "candidate_fact_ids": [], "target_visible_fact_ids": [], "source_search_calls": 1,
            "deletion_eligibility": {}, "arms": arms}


def write_run(d, states):
    d.mkdir(parents=True)
    (d / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    write_json(d / "run_manifest.json", {"status": "completed", "states_sha256": file_hash(d / "states.jsonl"), "control_mismatch_states": 0})


def h1_state(i, ds):
    if i < 10:      # degraded: review repairs even i, its filler repairs i < 5
        y = {"clean_a": 1, "perm_1": 0, "perm_2": int(i < 6), "perm1_review": int(i % 2 == 0), "perm1_neutral": int(i < 5),
             "perm1_reflect": 1, "perm1_neutral_reflect": 1}
    elif i < 16:    # stable: review breaks i in 10..11, filler never
        y = {"clean_a": 1, "perm_1": 1, "perm_2": 1, "perm1_review": int(i >= 12), "perm1_neutral": 1, "perm1_reflect": 1,
             "perm1_neutral_reflect": 1}
    else:           # clean wrong
        y = {a: 0 for a in H1_ARMS}
    y["clean_b"] = y["clean_a"]
    arms = {a: arm(v) for a, v in y.items()}
    if i == 12:     # a surface-form flip the semantic view absorbs
        arms["perm_2"] = arm(0, "Paris France")
    return state("{}:{}".format(ds, i), ds, arms)


class ReviewCpuTests(unittest.TestCase):
    def test_all_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "u1"
            for label in ("v03_7b_ppo", "research_7b"):
                write_run(root / label / "pcr", [h1_state(i, "nq" if i % 2 else "hotpotqa") for i in range(20)])
                as1 = []
                for i in range(20):
                    s = h1_state(i, "nq" if i % 2 else "hotpotqa")
                    s["arms"].update(perm1_alert=arm(1 if i < 10 else s["arms"]["perm1_review"]["em"]), perm1_neutral_alert=arm(0 if i < 10 else 1),
                                     perm1_hint=arm(1), perm1_neutral_hint=arm(0))
                    as1.append(s)
                write_run(root / label / "as1", as1)
                # U1N: 12 questions in one dataset; even ones searched, odd ones answered without searching (em 1)
                sources = [{"sample_id": "nq:{}".format(i), "dataset": "nq", "prefixes": [{}] if i % 2 == 0 else [], "em": 1 if i % 2 else 0}
                           for i in range(12)]
                (root / label / "sources").mkdir(parents=True)
                (root / label / "sources" / "sources.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sources))
                searched = sorted((s for s in sources if s["prefixes"]), key=lambda s: rank_key(7, s["sample_id"]))
                (root / label / "u1n_plan").mkdir()
                write_json(root / label / "u1n_plan" / "plan.json", {
                    "question_ids": [s["sample_id"] for s in searched], "datasets": ["nq"], "protocol": {"seed": 7},
                    "sources_path": str(root / label / "sources" / "sources.jsonl")})
                u1n = [state(s["sample_id"], "nq", {"clean_a": arm(0, tokens=5, searches=0), "clean_b": arm(0), "perm_1": arm(0),
                                                    "review": arm(1, tokens=20, searches=1), "neutral": arm(0), "reflect": arm(0),
                                                    "neutral_reflect": arm(0)}) for s in searched]
                write_run(root / label / "u1n", u1n)
            out = Path(tmp) / "out"
            out.mkdir()
            r = review_cpu.analyze(root, ["v03_7b_ppo", "research_7b"], 50, out)

            fw = r["fourway"]["sem"]["methods"]["review"]["filler"]
            self.assertEqual(fw["repair_counts"], {"both": 6, "method_only": 4, "other_only": 4, "neither": 6})
            self.assertAlmostEqual(fw["overlap_of_method_repairs"]["estimate"], 0.6)
            self.assertEqual(fw["break_counts"]["method_only"], 4)

            w = r["weights"]["estimands"]["review_repair_minus_filler"]
            self.assertAlmostEqual(w["state_weighted"]["estimate"], 0.0)
            self.assertAlmostEqual(w["agent_equal"]["estimate"], 0.0)
            self.assertAlmostEqual(w["recipe_equal"]["estimate"], 0.0)
            self.assertEqual(r["weights"]["recipes"], {"ReSearch": ["research_7b"], "Search-R1": ["v03_7b_ppo"]})
            h = r["weights"]["estimands"]["review_harm_minus_filler"]["state_weighted"]["estimate"]
            self.assertAlmostEqual(h, 2 / 6)

            self.assertAlmostEqual(r["alert"]["alert_minus_review_repair"]["estimate"], 0.5)

            p = r["policy"]
            last = max(sorted(range(12), key=lambda i: rank_key(7, "nq:{}".format(i))).index(i) for i in range(0, 12, 2))
            expected_no_search = sum(1 for i in sorted(range(12), key=lambda i: rank_key(7, "nq:{}".format(i)))[:last + 1] if i % 2)
            self.assertEqual(p["receipts"]["v03_7b_ppo"]["no_search_questions"], expected_no_search)
            n = 6 + expected_no_search
            self.assertAlmostEqual(p["contrasts"]["review-clean_a"]["raw"]["estimate"], 6 / n)
            self.assertAlmostEqual(p["added_searches"]["review"], 6 / n)

            s = r["semantic"]
            self.assertEqual(s["per_agent_arm"]["v03_7b_ppo"]["perm_2"]["changed"], 1)
            self.assertTrue((out / "semantic_cases.jsonl").read_text().strip())
            self.assertIn("四格表", review_cpu.markdown(r))


if __name__ == "__main__":
    unittest.main()

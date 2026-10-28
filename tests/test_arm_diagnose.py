"""CPU test for the PCR insertion-arm diagnostic."""
import json
import tempfile
import unittest
from pathlib import Path

from work2 import arm_diagnose

ARMS = ("clean_a", "perm_1", "perm_2", "perm1_review", "perm1_neutral", "perm1_reflect", "perm1_neutral_reflect", "perm1_budget2")


def state(i, broken):
    def arm(name):
        bad = broken and name in ("perm1_review", "perm1_neutral", "perm1_reflect", "perm1_neutral_reflect")
        return {"status": "malformed_action" if bad else "answered", "em": 0 if bad else 1, "final_answer": "" if bad else "Paris",
                "search_calls_after_target": 0, "generated_tokens_after_target": 3 if bad else 20,
                "post_target_texts": ["<|im_end|>" if bad else "<answer>Paris</answer>"]}
    return {"sample_id": "nq:{}".format(i), "gold_answer": ["Paris"], "arms": {a: arm(a) for a in ARMS}}


class DiagnoseTests(unittest.TestCase):
    def test_insertion_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "x" / "pcr"; run.mkdir(parents=True)
            (run / "states.jsonl").write_text("".join(json.dumps(state(i, True)) + "\n" for i in range(5)))
            d = arm_diagnose.diagnose(run)
            self.assertEqual(d["stable_states"], 5)
            self.assertEqual(d["arms"]["perm1_neutral"]["answered"], 0.0)
            self.assertEqual(d["arms"]["perm_2"]["em"], 1.0)
            self.assertEqual(d["arms"]["perm1_neutral"]["statuses"], {"malformed_action": 5})
            self.assertIn("<|im_end|>", arm_diagnose.markdown("x", d))
            self.assertFalse(d["insertion_valid"])
            self.assertIn("插入有效性", arm_diagnose.validity_table({"x": d}))
            (run.parent.parent / "y" / "pcr").mkdir(parents=True)
            (run.parent.parent / "y" / "pcr" / "states.jsonl").write_text("".join(json.dumps(state(i, False)) + "\n" for i in range(5)))
            self.assertTrue(arm_diagnose.diagnose(run.parent.parent / "y" / "pcr")["insertion_valid"])


if __name__ == "__main__":
    unittest.main()

"""CPU tests for AS1 / U1N: assay and natural arms, AS1 state selection, and the analysis readings."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from work2 import as1u1n, pcr
from work2.common import file_hash, write_json
from work2.placebo import ARM_GROUPS, run_state
from tests.test_pcr import AGENT, RETRIEVAL, DocRetriever, OrderGenerator, source


class HintReader(OrderGenerator):
    """Also answers correctly when the inserted text names the answer."""
    def generate(self, tokens, limit, seed):
        out = super().generate(tokens, limit, seed)
        text = bytes(tokens).decode()
        if "<answer>" in out["text"] and "is \"Paris\"" in text.rsplit("\n\n<information>", 1)[-1]:
            out = {"text": "<answer>Paris</answer>", "token_ids": self.encode("<answer>Paris</answer>"), "finish_reason": "stop"}
        return out


class ArmTests(unittest.TestCase):
    def test_assay_arms_hint_alert_and_matched_fillers(self):
        row = run_state(source(), 1, ARM_GROUPS["assay"], HintReader(), DocRetriever(), AGENT, RETRIEVAL)
        arms = row["arms"]
        self.assertEqual(set(arms), set(ARM_GROUPS["assay"]))
        self.assertEqual(arms["perm1_hint"]["em"], 1)
        self.assertEqual(arms["perm1_neutral_hint"]["em"], 0)
        for method, filler in (("perm1_alert", "perm1_neutral_alert"), ("perm1_hint", "perm1_neutral_hint"), ("perm1_review", "perm1_neutral")):
            self.assertEqual(arms[method]["metadata"]["instruction_tokens"], arms[filler]["metadata"]["instruction_tokens"])
            self.assertEqual(arms[filler]["metadata"]["matched_tokens"], arms[method]["metadata"]["instruction_tokens"])
        self.assertIn('"Paris"', arms["perm1_hint"]["metadata"]["instruction"])
        for filler in ("perm1_neutral_alert", "perm1_neutral_hint"):
            self.assertNotIn("Paris", arms[filler]["metadata"]["instruction"])
            self.assertNotIn("wrong", arms[filler]["metadata"]["instruction"])

    def test_h1_arms_unchanged_by_refactor(self):
        a = run_state(source(), 1, ARM_GROUPS["h1"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)["arms"]
        b = run_state(source(), 1, ARM_GROUPS["assay"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)["arms"]
        for arm in ("clean_a", "perm_1", "perm1_review", "perm1_neutral"):
            self.assertEqual(a[arm]["continuation_sha256"], b[arm]["continuation_sha256"])
            self.assertEqual(a[arm]["metadata"], b[arm]["metadata"])

    def test_natural_arms_keep_clean_order(self):
        arms = run_state(source(), 1, ARM_GROUPS["natural"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)["arms"]
        self.assertEqual(set(arms), set(ARM_GROUPS["natural"]))
        for arm in ("review", "neutral", "reflect", "neutral_reflect"):
            self.assertFalse(arms[arm]["observation_changed"])
            self.assertEqual(arms[arm]["metadata"]["base_observation"], "clean")


def write_run(d, states, mismatches=0):
    d.mkdir(parents=True)
    (d / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    write_json(d / "run_manifest.json", {"status": "completed", "states_sha256": file_hash(d / "states.jsonl"),
                                         "control_mismatch_states": mismatches})


class SelectionTests(unittest.TestCase):
    def test_select_all_degraded_and_hash_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [source(i) for i in range(12)]
            (root / "sources.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sources))
            states = [run_state(s, 1, ARM_GROUPS["h1"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL) for s in sources]
            write_run(root / "pcr", states)
            protocol = dict(json.loads(Path("configs/pcr_u1_bounded.json").read_text()), arm_group="assay")
            (root / "as1.json").write_text(json.dumps(protocol))
            args = argparse.Namespace(protocol=root / "as1.json", sources=root / "sources.jsonl", output=root / "plan",
                                      select_from=root / "pcr", stable_sample=1, wrong_sample=0, per_dataset=0)
            pcr.prepare(args)
            plan = json.loads((root / "plan" / "plan.json").read_text())
            sel = plan["selected_from"]
            self.assertEqual(sel["selected"]["degraded"], sel["available"]["degraded"])
            self.assertEqual(sel["selected"]["stable"], min(1, sel["available"]["stable"]))
            self.assertEqual(len(plan["states"]), sel["selected"]["degraded"] + sel["selected"]["stable"])
            self.assertEqual(set(plan["question_ids"]), {s["sample_id"] for s in plan["states"]})
            with self.assertRaises(ValueError):
                pcr.prepare(args)


def state(i, dataset, outcome, sha="x"):
    arms = {a: {"em": outcome.get(a, outcome["clean_a"]), "status": "answered", "continuation_sha256": sha + a,
                "final_answer": "Paris" if outcome.get(a, outcome["clean_a"]) else "Rome"} for a in outcome["_arms"]}
    return {"sample_id": "{}:{}".format(dataset, i), "dataset": dataset, "target_turn": 1, "gold_answer": "Paris",
            "known_fact_ids": [], "candidate_fact_ids": [], "target_visible_fact_ids": [], "source_search_calls": 2,
            "deletion_eligibility": {}, "arms": arms}


class AnalysisTests(unittest.TestCase):
    def test_readings(self):
        assay, natural = ARM_GROUPS["assay"], ARM_GROUPS["natural"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, n, h = [], [], []
            for i in range(20):
                ds = "hotpotqa" if i % 2 else "nq"
                # degraded: hint always repairs, its filler never; review repairs as often as its filler
                a.append(state(i, ds, {"_arms": assay, "clean_a": 1, "perm_1": 0, "perm1_review": i % 2, "perm1_neutral": i % 2,
                                        "perm1_alert": 0, "perm1_neutral_alert": 0, "perm1_hint": 1, "perm1_neutral_hint": 0}))
                h.append(state(i, ds, {"_arms": ARM_GROUPS["h1"], "clean_a": 1, "perm_1": 0}))
                # natural: wrong states rescued equally by method and filler; right states untouched
                n.append(state(i, ds, {"_arms": natural, "clean_a": int(i < 10), "perm_1": int(i < 10), "review": int(i < 12),
                                        "neutral": int(i < 12), "reflect": int(i < 11), "neutral_reflect": int(i < 11)}))
            write_run(root / "v03_7b_ppo" / "as1", a)
            write_run(root / "v03_7b_ppo" / "pcr", h)
            write_run(root / "v03_7b_ppo" / "u1n", n)
            r = as1u1n.analyze(root, ["v03_7b_ppo", "research_32b"], 50)
            c = r["assay"]["pooled"]["sem"]["contents"]
            self.assertAlmostEqual(c["hint"]["repair_minus_filler"]["estimate"], 1.0)
            self.assertAlmostEqual(c["review"]["repair_minus_filler"]["estimate"], 0.0)
            self.assertIsNone(r["assay"]["agents"]["research_32b"])
            self.assertEqual(r["assay"]["reproduction"]["v03_7b_ppo"], {k: 1.0 for k in as1u1n.REPRO_ARMS})
            self.assertTrue(r["readings"]["A3"])
            m = r["natural"]["pooled"]["sem"]["methods"]["review"]
            self.assertAlmostEqual(m["rescue"], 0.2)
            self.assertAlmostEqual(m["explained"], 1.0)
            self.assertAlmostEqual(m["all_minus_filler"]["estimate"], 0.0)
            g = r["readings"]
            self.assertTrue(g["A1_pooled"])
            self.assertFalse(g["A1"])          # one agent measured, 12 needed
            self.assertEqual(g["review"]["reading"], "same as H1")
            self.assertIn("A1_insensitive", g)
            self.assertIn("AS1 / U1N", as1u1n.markdown(r))


if __name__ == "__main__":
    unittest.main()

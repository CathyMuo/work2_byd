"""CPU tests for AS2 (research/AS2_PROTOCOL.md): filler variants, graded positive controls, sampled arms, configs, analysis."""
import collections
import json
import tempfile
import unittest
from pathlib import Path

import itertools

from work2 import as2, pcr, u2
from work2.common import ROOT, file_hash, write_json
from work2.placebo import (ARM_GROUPS, FILLER_VARIANT_ARMS, REVIEW_PROMPT, SAMPLED_ARMS, inside_observation, permutations_for,
                           run_state)
from work2.rollout import run_sample
from test_pcr import AGENT, RETRIEVAL, DocRetriever, OrderGenerator, sample, source


class AnswerRetriever(DocRetriever):
    """Like DocRetriever, but the second visible document and the seventh candidate mention the answer."""
    def search(self, query, top_k):
        docs = super().search(query, top_k)
        docs[1] = dict(docs[1], contents="Beta\nBeta was born in Paris.")
        if top_k >= 7:
            docs[6] = dict(docs[6], contents="Filler7\nThe capital is Paris.")
        return docs


def answer_source(i=0):
    return run_sample(sample(i), OrderGenerator(), AnswerRetriever(), AGENT, RETRIEVAL)


class SampleGenerator(OrderGenerator):
    """Greedy as OrderGenerator; a sampled answer step answers Paris on odd seeds and London on even ones."""
    def generate_sampled(self, tokens, limit, seed, temperature, top_p):
        out = super().generate(tokens, limit, seed)
        if out["text"].startswith("<answer>"):
            text = "<answer>{}</answer>".format("Paris" if seed % 2 else "London")
            out = {"text": text, "token_ids": self.encode(text), "finish_reason": "stop"}
        return out


class FillerTests(unittest.TestCase):
    def test_variants_match_review_tokens_and_carry_what_they_test(self):
        g = OrderGenerator()
        arms = run_state(source(), 1, ARM_GROUPS["as2"], g, DocRetriever(), AGENT, RETRIEVAL)["arms"]
        review = len(g.encode(REVIEW_PROMPT))
        texts = {}
        for arm in FILLER_VARIANT_ARMS:
            m = arms[arm]["metadata"]
            self.assertEqual(m["instruction_tokens"], review)
            self.assertEqual(m["matched_tokens"], review)
            self.assertFalse(m["leaks_answer"])
            self.assertEqual(m["base_observation"], "perm_1")
            texts[arm] = m["instruction"]
            self.assertTrue(texts[arm].startswith("\n\n") and texts[arm].endswith("\n\n"))
        self.assertIn("<search> or <answer> protocol.", texts["perm1_fill_protocol"])
        self.assertNotIn("<search>", texts["perm1_fill_prose"])
        self.assertFalse(any(c.isalpha() for c in texts["perm1_fill_punct"]))
        shuffled = texts["perm1_fill_shuffled"].split()
        self.assertNotEqual(shuffled[:5], REVIEW_PROMPT.split()[:5])
        have, ref = collections.Counter(shuffled[:-1]), collections.Counter(REVIEW_PROMPT.split())
        self.assertTrue(all(n <= ref[w] for w, n in have.items()))     # sub-multiset; Counter <= needs Python 3.10
        syntax = texts["perm1_fill_syntax"]
        self.assertTrue(syntax.startswith("\n\nBefore continuing, note:"))
        self.assertIn("<search> or <answer> protocol.", syntax)
        for word in ("review", "check", "reflect", "wrong"):
            self.assertNotIn(word, syntax.lower())

    def test_inside_observation_arms(self):
        self.assertEqual(inside_observation("\n\n<information>A</information>\n\n", "\n\nNote x\n\n", "information"),
                         "\n\n<information>A\nNote x\n</information>\n\n")
        with self.assertRaises(ValueError):
            inside_observation("<result>A</result>", "x", "information")
        arms = run_state(source(), 1, ARM_GROUPS["as2"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)["arms"]
        for inside, after in (("perm1_review_inobs", "perm1_review"), ("perm1_neutral_inobs", "perm1_neutral")):
            self.assertTrue(arms[inside]["observation_changed"])
            self.assertEqual(arms[inside]["em"], arms[after]["em"])        # the stub reads the text wherever it is
            self.assertNotEqual(arms[inside]["observation_sha256"], arms[after]["observation_sha256"])
        self.assertEqual(arms["perm1_review_inobs"]["metadata"]["inside_text"], REVIEW_PROMPT)
        self.assertEqual(arms["perm1_neutral_inobs"]["metadata"]["inside_text"], arms["perm1_neutral"]["metadata"]["instruction"])

    def test_shared_arms_identical_to_h1(self):
        a = run_state(source(), 1, ARM_GROUPS["h1"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)["arms"]
        b = run_state(source(), 1, ARM_GROUPS["as2"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)["arms"]
        for arm in ("clean_a", "perm_1", "perm1_review", "perm1_neutral"):
            self.assertEqual(a[arm]["continuation_sha256"], b[arm]["continuation_sha256"])
            self.assertEqual(a[arm]["metadata"], b[arm]["metadata"])


class GradedTests(unittest.TestCase):
    def test_ineligible_states_record_none(self):
        arms = run_state(source(), 1, ARM_GROUPS["as2"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)["arms"]
        for arm in ("perm1_pointer", "perm1_neutral_pointer", "perm1_evidence", "perm1_neutral_evidence"):
            self.assertIsNone(arms[arm])

    def test_pointer_and_evidence(self):
        src = answer_source()
        arms = run_state(src, 1, ARM_GROUPS["as2"], OrderGenerator(), AnswerRetriever(), AGENT, RETRIEVAL)["arms"]
        order = permutations_for(src["sample_id"], 1, 3)[0]
        position = order.index(1) + 1                      # where Beta (canonical index 1) is shown under perm_1
        p = arms["perm1_pointer"]["metadata"]
        self.assertEqual(p["eligibility"]["position"], position)
        self.assertIn("the {} of the documents".format(("first", "second", "third")[position - 1]), p["instruction"])
        self.assertNotIn("Paris", p["instruction"])
        e = arms["perm1_evidence"]["metadata"]
        self.assertEqual(e["eligibility"]["candidate_rank"], 7)
        self.assertIn("The capital is Paris.", e["instruction"])
        for method, filler in (("perm1_pointer", "perm1_neutral_pointer"), ("perm1_evidence", "perm1_neutral_evidence")):
            self.assertEqual(arms[filler]["metadata"]["instruction_tokens"], arms[method]["metadata"]["instruction_tokens"])
            self.assertNotIn("Paris", arms[filler]["metadata"]["instruction"])


class SampledTests(unittest.TestCase):
    SAMPLING = {"temperature": 1.0, "top_p": 1.0, "k": 3}

    def test_sampled_arms_k_samples_and_greedy_controls(self):
        row = run_state(source(), 1, ARM_GROUPS["as2s"], SampleGenerator(), DocRetriever(), AGENT, RETRIEVAL, sampling=self.SAMPLING)
        h1 = run_state(source(), 1, ARM_GROUPS["h1"], OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)["arms"]
        self.assertTrue(row["controls_exact"])
        for arm in ("clean_a", "perm_1"):
            self.assertEqual(row["arms"][arm]["continuation_sha256"], h1[arm]["continuation_sha256"])
        for arm in SAMPLED_ARMS:
            samples = row["arms"][arm]["samples"]
            self.assertEqual(len(samples), 3)
            self.assertEqual(len({s["sampling"]["seed"] for s in samples}), 3)
            self.assertTrue(all(s["sampling"]["temperature"] == 1.0 for s in samples))
        self.assertEqual(row["arms"]["s_perm1_review"]["samples"][0]["metadata"]["sampled_from"], "perm1_review")
        self.assertEqual(row["arms"]["s_perm2"]["samples"][0]["metadata"]["order"], h1["perm_2"]["metadata"]["order"])
        again = run_state(source(), 1, ARM_GROUPS["as2s"], SampleGenerator(), DocRetriever(), AGENT, RETRIEVAL, sampling=self.SAMPLING)
        self.assertEqual(json.dumps(again["arms"], sort_keys=True), json.dumps(row["arms"], sort_keys=True))

    def test_five_orders_enumerate_every_non_identity_permutation(self):
        row = run_state(source(), 1, ARM_GROUPS["as2s"], SampleGenerator(), DocRetriever(), AGENT, RETRIEVAL, sampling=self.SAMPLING)
        orders = [tuple(row["arms"]["perm_{}".format(k)]["metadata"]["order"]) for k in range(1, 6)]
        self.assertEqual(set(orders), set(itertools.permutations(range(3))) - {(0, 1, 2)})
        self.assertEqual([list(o) for o in orders[:4]], permutations_for(source()["sample_id"], 1, 3))

    def test_sampled_arm_needs_configuration(self):
        with self.assertRaises(ValueError):
            run_state(source(), 1, ARM_GROUPS["as2s"], SampleGenerator(), DocRetriever(), AGENT, RETRIEVAL)


class ConfigTests(unittest.TestCase):
    def test_configs_reuse_as1_selection_and_validate(self):
        for variant in ("bounded", "official", "research", "stepsearch"):
            as1 = json.loads((ROOT / "configs" / "as1_{}.json".format(variant)).read_text())
            new = json.loads((ROOT / "configs" / "as2_{}.json".format(variant)).read_text())
            self.assertEqual({k: v for k, v in new.items() if k not in ("arm_group", "scope")},
                             {k: v for k, v in as1.items() if k not in ("arm_group", "scope")})
            self.assertEqual(new["arm_group"], "as2")
            pcr.load_protocol(ROOT / "configs" / "as2_{}.json".format(variant))
        s = pcr.load_protocol(ROOT / "configs" / "as2s_bounded.json")
        self.assertEqual((s["sampling"]["temperature"], s["sampling"]["k"]), (1.0, 4))
        self.assertEqual(s["seed"], json.loads((ROOT / "configs" / "as1_bounded.json").read_text())["seed"])
        self.assertEqual(s["agent"]["temperature"], 0)

    def test_mrsearch_greedy_config(self):
        p = u2.load_protocol(ROOT / "configs" / "u2_mrsearch_greedy.json")
        self.assertEqual((p["sampling"]["temperature"], p["sampling"]["k"]), (0.0, 1))
        base = json.loads((ROOT / "configs" / "u2_mrsearch.json").read_text())
        text = ("sampling", "source", "scope", "reference")
        self.assertEqual({k: v for k, v in p.items() if k not in text}, {k: v for k, v in base.items() if k not in text})
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ({"temperature": 1.0, "top_p": 1.0, "k": 1}, {"temperature": 0.0, "top_p": 1.0, "k": 2}):
                path = Path(tmp) / "p.json"
                path.write_text(json.dumps(dict(base, sampling=bad)))
                with self.assertRaises(ValueError):
                    u2.load_protocol(path)

    def test_sampling_block_only_with_as2s(self):
        base = json.loads((ROOT / "configs" / "as2s_bounded.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (dict(base, arm_group="as2"), {k: v for k, v in base.items() if k != "sampling"},
                        dict(base, sampling=dict(base["sampling"], k=1))):
                path = Path(tmp) / "p.json"
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    pcr.load_protocol(path)


class LuckTests(unittest.TestCase):
    """Amendment 2: the luck model's |S| is the full stable set; sampled stable states are weighted up."""
    @staticmethod
    def row(i, perm1, y, raw_perm1=None):
        sem = dict({"perm_1": perm1}, **{"perm_{}".format(k): y for k in range(2, 6)})
        return {"dataset": "nq", "sample_id": "nq:{}".format(i), "y0": 1, "sem": sem,
                "raw": dict(sem, perm_1=perm1 if raw_perm1 is None else raw_perm1)}

    def test_weights_prediction_and_gap(self):
        deg = [self.row(i, 0, int(i < 5)) for i in range(10)]                       # observed repair 0.5
        stab = [self.row(10 + i, 1, int(i > 0)) for i in range(5)] + [self.row(20, 1, 1, raw_perm1=0)]
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "plan.json"
            write_json(plan, {"selected_from": {"available": {"stable": 10}, "selected": {"stable": 5}}})
            w = as2.stable_weights(deg + stab, plan)
        self.assertEqual(w["weight"], 2.0)
        self.assertEqual([r["w"] for r in stab], [2.0] * 5 + [1.0])                 # raw-degraded state was kept whole
        b = as2.luck_block(deg + stab, 200)
        o = b["orders"]["perm_3"]
        self.assertAlmostEqual(b["stable_represented"], 11.0)
        self.assertAlmostEqual(o["predicted_repair"], 0.2)                            # 1 break x weight 2 / |D| = 10
        self.assertAlmostEqual(o["gap"]["estimate"], 0.3)
        self.assertEqual(o["harm_events_in_sample"], 1)
        self.assertLessEqual(o["gap"]["ci95"][0], 0.3)

    def test_as2l_group_and_config(self):
        p = pcr.load_protocol(ROOT / "configs" / "as2l_bounded.json")
        self.assertEqual(p["arm_group"], "as2l")
        self.assertNotIn("sampling", p)
        self.assertEqual(ARM_GROUPS["as2l"], ("clean_a", "clean_b", "perm_1", "perm_2", "perm_3", "perm_4", "perm_5"))
        self.assertEqual(p["seed"], json.loads((ROOT / "configs" / "as2s_bounded.json").read_text())["seed"])


def arm(em, sha="x", **extra):
    return dict({"em": em, "status": "answered", "final_answer": "Paris" if em else "Rome", "continuation_sha256": sha,
                 "metadata": {}}, **extra)


def st(i, ds, arms):
    return {"sample_id": "{}:{}".format(ds, i), "dataset": ds, "target_turn": 1, "gold_answer": "Paris", "controls_exact": True,
            "arms": arms}


def write_run(d, states):
    d.mkdir(parents=True)
    (d / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    write_json(d / "run_manifest.json", {"status": "completed", "states_sha256": file_hash(d / "states.jsonl"),
                                         "control_mismatch_states": 0})


class AnalysisTests(unittest.TestCase):
    def test_readings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "u1"
            h1, a2, a1, a2s = [], [], [], []
            for i in range(20):
                ds = "hotpotqa" if i % 2 else "nq"
                base = {"clean_a": arm(1, "c"), "perm_1": arm(0, "p"), "perm1_review": arm(i % 2, "r"), "perm1_neutral": arm(i % 2, "n")}
                h1.append(st(i, ds, dict(base, perm_2=arm(int(i < 10), "q2"))))
                # degraded states: punct filler never repairs (so P2 - punct = 0.5), other fillers repair like P2;
                # pointer eligible on even states and always repairs, its filler never; evidence never eligible
                a2.append(st(i, ds, dict(base, perm1_fill_prose=arm(i % 2), perm1_fill_protocol=arm(i % 2),
                                          perm1_fill_shuffled=arm(i % 2), perm1_fill_punct=arm(0), perm1_fill_syntax=arm(i % 2),
                                          perm1_review_inobs=arm(1), perm1_neutral_inobs=arm(i % 2),
                                          perm1_pointer=arm(1) if i % 2 == 0 else None,
                                          perm1_neutral_pointer=arm(0) if i % 2 == 0 else None,
                                          perm1_evidence=None, perm1_neutral_evidence=None)))
                a1.append(st(i, ds, dict(base, perm1_alert=arm(0), perm1_neutral_alert=arm(0), perm1_hint=arm(1), perm1_neutral_hint=arm(0))))
                samples = lambda p: {"samples": [arm(int(j < p)) for j in range(4)]}
                a2s.append(st(i, ds, {"clean_a": arm(1, "c"), "clean_b": arm(1, "c"), "perm_1": arm(0, "p"),
                                      **{"perm_{}".format(k): arm(int(i < 10), "q{}".format(k)) for k in range(2, 6)},
                                      "s_perm1": samples(3), "s_perm2": samples(2), "s_perm1_review": samples(3),
                                      "s_perm1_neutral": samples(2), "s_perm1_reflect": samples(2), "s_perm1_neutral_reflect": samples(2)}))
            write_run(root / "v03_7b_ppo" / "pcr", h1)
            write_run(root / "v03_7b_ppo" / "as2", a2)
            write_run(root / "v03_7b_ppo" / "as1", a1)
            write_run(root / "v03_7b_ppo" / "as2s", a2s)
            (root / "v03_7b_ppo" / "as2s_plan").mkdir()
            write_json(root / "v03_7b_ppo" / "as2s_plan" / "plan.json", {"selected_from": {"available": {"stable": 0}, "selected": {"stable": 0}}})
            r = as2.analyze(root, ["v03_7b_ppo", "research_32b"], ["v03_7b_ppo"], 50)
            f = r["filler"]["pooled"]["sem"]
            self.assertAlmostEqual(f["variants"]["punct"]["p2_minus_v"]["estimate"], 0.5)
            self.assertAlmostEqual(f["variants"]["prose"]["p2_minus_v"]["estimate"], 0.0)
            g = r["graded"]["pooled"]["sem"]["contents"]
            self.assertEqual(g["pointer"]["eligible_degraded"], 10)
            self.assertAlmostEqual(g["pointer"]["repair_minus_filler"]["estimate"], 1.0)
            self.assertAlmostEqual(g["pointer"]["ladder"]["hint"]["estimate"], 1.0)
            self.assertEqual(g["evidence"]["eligible_degraded"], 0)
            s = r["sampled"]["pooled"]["sem"]["methods"]
            self.assertAlmostEqual(s["review"]["repair"], 0.75)
            self.assertAlmostEqual(s["review"]["repair_minus_filler"]["estimate"], 0.25)
            self.assertAlmostEqual(s["review"]["explained"], 1.0)              # the plain resample repairs as much
            self.assertAlmostEqual(s["review"]["repair_minus_resample"]["estimate"], 0.0)
            i = r["interface"]["pooled"]["sem"]
            self.assertAlmostEqual(i["repair_content_inside"]["estimate"], 0.5)
            self.assertAlmostEqual(i["repair_content_after"]["estimate"], 0.0)
            lk = r["luck"]["pooled"]["sem"]
            self.assertAlmostEqual(lk["orders"]["perm_3"]["repair"]["estimate"], 0.5)
            self.assertAlmostEqual(lk["never_repaired"], 0.5)
            self.assertEqual(lk["repair_spread"], 0.0)
            x = r["readings"]
            self.assertTrue(x["L1"])
            self.assertFalse(x["I1"])
            self.assertEqual(x["F1_variants"], ["punct"])
            self.assertTrue(x["G1"]); self.assertFalse(x["G2"])
            self.assertTrue(x["S1"]); self.assertTrue(x["S2"])
            self.assertTrue(x["repro_ok"])
            self.assertIsNone(r["filler"]["agents"]["research_32b"])
            self.assertIn("AS2", as2.markdown(r))


if __name__ == "__main__":
    unittest.main()

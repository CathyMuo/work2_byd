"""CPU tests for PCR v1 placebo arms, runner, analysis and Phase 0; no model/GPU."""
import argparse
import json
import re
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from work2 import pcr, pcr_analysis, phase0_luck
from work2.common import ROOT, file_hash, load_json, write_json
from work2.placebo import (ALL_ARMS, FORMAT_ARMS, PERMUTATION_ARMS, REVIEW_PROMPT, deletion_plan, multiset,
                           neutral_instruction, permutations_for, render, run_state, usable_states)
from work2.replay import observation, replay_branch, validate_prefix
from work2.rollout import run_sample

AGENT = {"max_search_turns": 4, "max_context_tokens": 20000, "max_generation_tokens_per_turn": 512, "seed": 42, "temperature": 0}
RETRIEVAL = {"visible_top_k": 3, "candidate_top_k": 20}


class DocRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        docs = ["Alpha\nAlpha fact.", "Beta\nBeta text here.", "Gamma\nGamma text here."]
        docs += ["Filler{}\nFiller text {}.".format(i, i) for i in range(4, top_k + 1)]
        return [{"document_id": "d{}".format(i), "title": c.split("\n")[0], "contents": c, "rank": i + 1, "score": 1.0}
                for i, c in enumerate(docs[:top_k])]


class OrderGenerator:
    """Answers correctly only when Alpha is shown first in canonical format, or after a review hint."""
    def __init__(self, searches=None, drift_after=None):
        self.searches, self.calls, self.drift_after = searches, 0, drift_after

    def encode(self, text):
        return list(text.encode())

    def initial_prompt(self, text):
        return text

    def generate(self, tokens, limit, seed):
        self.calls += 1
        text = bytes(tokens).decode()
        seen = text.count("\n\n<information>")
        wanted = self.searches or 1 + int(re.search(r"Where (\d+)\?", text).group(1)) % 2
        if seen < wanted:
            out = "<search>q{}</search>".format(seen)
        else:
            last = text.rsplit("\n\n<information>", 1)[1]
            good = last.startswith("Doc 1(Title: Alpha)") or "briefly review" in last
            out = "<answer>{}</answer>".format("Paris" if good else "London")
            if self.drift_after is not None and self.calls > self.drift_after:
                out = "<think>x</think>" + out
        return {"text": out, "token_ids": self.encode(out), "finish_reason": "stop"}


def sample(i=0, dataset="hotpotqa"):
    return {"sample_id": "{}:{}".format(dataset, i), "dataset": dataset, "question": "Where {}?".format(i),
            "gold_answer": "Paris", "gold_supporting_facts": [{"title": "Alpha", "sentence_id": 0, "sentence": "Alpha fact."}]}


def source(i=0, dataset="hotpotqa"):
    """Question i searches 1 + i % 2 times."""
    return run_sample(sample(i, dataset), OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)


class PlaceboTests(unittest.TestCase):
    def test_format_variants_keep_titles_bodies_and_order(self):
        docs = DocRetriever().search("q", 3)
        for variant in FORMAT_ARMS:
            text = render(docs, variant)
            self.assertNotEqual(text, observation(docs))
            positions = [text.index(d["contents"].split("\n", 1)[1]) for d in docs]
            self.assertEqual(positions, sorted(positions))
            for d in docs:
                self.assertIn(d["contents"].split("\n")[0], text)
        self.assertEqual(render(docs), observation(docs))

    def test_permutations_deterministic_distinct_non_identity(self):
        a = permutations_for("x", 1, 3)
        self.assertEqual(a, permutations_for("x", 1, 3))
        self.assertEqual(len({tuple(p) for p in a}), 4)
        self.assertNotIn([0, 1, 2], a)

    def test_neutral_matches_review_token_count(self):
        enc = lambda t: list(t.encode())
        text = neutral_instruction(enc, len(enc(REVIEW_PROMPT)))
        self.assertEqual(len(enc(text)), len(enc(REVIEW_PROMPT)))
        self.assertNotIn("review", text)

    def test_deletion_plan_uses_same_clean_candidate(self):
        src = source()
        plan = deletion_plan(validate_prefix(src, 1, AGENT, RETRIEVAL), src, 1)
        sup, non = plan["arms"]["drop_support"], plan["arms"]["drop_nonsupport"]
        self.assertTrue(sup["eligible"] and non["eligible"])
        self.assertEqual(sup["slot"], 0)
        self.assertEqual(sup["replacement_document_id"], non["replacement_document_id"])
        self.assertEqual(sup["lost_fact_ids"], [0])
        self.assertEqual(non["lost_fact_ids"], [])

    def test_state_runs_all_arms_with_exact_controls(self):
        src = source()
        row = run_state(src, 1, ALL_ARMS, OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)
        arms = row["arms"]
        self.assertEqual(set(arms), set(ALL_ARMS))
        self.assertEqual(arms["clean_a"]["em"], 1)
        self.assertEqual(arms["clean_a"]["continuation_sha256"], arms["clean_b"]["continuation_sha256"])
        for arm in PERMUTATION_ARMS + FORMAT_ARMS:
            self.assertTrue(arms[arm]["observation_changed"])
        self.assertEqual([arms[a]["em"] for a in FORMAT_ARMS], [0, 0])
        self.assertEqual(arms["drop_support"]["em"], 0)
        self.assertEqual(arms["drop_nonsupport"]["em"], 1)
        self.assertEqual(arms["perm1_review"]["em"], 1)
        self.assertEqual(arms["review"]["metadata"]["instruction_tokens"], arms["neutral"]["metadata"]["instruction_tokens"])

    def test_budget2_allows_two_extra_searches(self):
        class Searcher(OrderGenerator):
            def generate(self, tokens, limit, seed):
                out = "<search>again{}</search>".format(bytes(tokens).decode().count("\n\n<information>"))
                return {"text": out, "token_ids": self.encode(out), "finish_reason": "stop"}
        src = run_sample(sample(), Searcher(), DocRetriever(), AGENT, RETRIEVAL)
        self.assertEqual(src["search_calls"], 4)
        row = run_state(src, 1, ("clean_a", "clean_b", "budget2"), Searcher(), DocRetriever(), AGENT, RETRIEVAL)
        self.assertEqual(row["arms"]["budget2"]["search_calls"], 6)

    def test_control_drift_stops(self):
        src = source()
        with self.assertRaises(ValueError):
            run_state(src, 1, ("clean_a", "clean_b"), OrderGenerator(drift_after=1), DocRetriever(), AGENT, RETRIEVAL)

    def test_control_drift_recorded_and_excluded(self):
        src = source()
        row = run_state(src, 1, ("clean_a", "clean_b", "perm_1"), OrderGenerator(drift_after=1), DocRetriever(),
                        AGENT, RETRIEVAL, on_mismatch="record")
        self.assertFalse(row["controls_exact"])
        self.assertEqual(set(row["arms"]), {"clean_a", "clean_b", "perm_1"})
        receipt = row["control_mismatch"]["arms"]["clean_b"]
        self.assertEqual((receipt["kind"], receipt["step"], receipt["char"]), ("text", 1, 1))  # <answer> vs <think>
        self.assertEqual(list(row["control_mismatch"]["arms"]), ["clean_b"])
        self.assertFalse(row["control_mismatch"]["controls_agree"])
        self.assertEqual(usable_states([row]), [])
        with self.assertRaises(ValueError):
            usable_states([row], {"control_mismatch_states": 0})

    def test_second_turn_fork(self):
        src = source(1)
        row = run_state(src, 2, ALL_ARMS, OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)
        self.assertEqual(row["arms"]["clean_a"]["search_calls"], 2)
        self.assertEqual(row["deletion_plan"]["earlier_fact_ids"], [0])


class AnalysisTests(unittest.TestCase):
    def test_flip_kind(self):
        self.assertEqual(pcr_analysis.flip_kind("Paris", 1, "Paris, France", 0), "format")
        self.assertEqual(pcr_analysis.flip_kind("Paris", 1, "London", 0), "semantic")
        self.assertEqual(pcr_analysis.flip_kind("Paris", 1, "Paris", 1), "none")

    def merged_run(self, root, n=6):
        states = [run_state(source(i, d), 1, ALL_ARMS, OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)
                  for d in ("hotpotqa", "2wiki") for i in range(0, 2 * n, 2)]
        run = root / "run"; run.mkdir()
        with (run / "states.jsonl").open("w", encoding="utf-8") as f:
            for s in states:
                f.write(json.dumps(s) + "\n")
        write_json(run / "run_manifest.json", {"status": "completed", "states_sha256": file_hash(run / "states.jsonl")})
        return run

    def test_analysis_luck_recovery_deletion_gates(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pcr_analysis, "git_output", return_value="fixture"):
            report, cases = pcr_analysis.analyze(self.merged_run(Path(tmp)), resamples=50)
            luck = report["luck"]["semantic"]
            self.assertEqual(luck["clean_correct"], 12)
            self.assertEqual(luck["lucky_successes"], 12)
            self.assertEqual(luck["lucky_share_of_accuracy"]["estimate"], 1.0)
            self.assertTrue(report["gates"]["G1_semantic_lucky_share_ge_10pct_ci_lo_gt_5pct"])
            self.assertTrue(report["deletion"]["semantic"]["available"])
            self.assertLess(report["deletion"]["semantic"]["all_matched"]["support_minus_nonsupport"]["estimate"], 0)
            self.assertTrue(cases)
            self.assertIn("路径运气", pcr_analysis.markdown(report))

    def test_tampered_states_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.merged_run(Path(tmp), 2)
            with (run / "states.jsonl").open("a") as f:
                f.write("\n")
            with self.assertRaises(ValueError):
                pcr_analysis.analyze(run, resamples=0)


class RunnerTests(unittest.TestCase):
    def test_prepare_worker_merge(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(pcr, "git_output", return_value="fixture"):
            root = Path(tmp)
            sources = root / "sources.jsonl"
            with sources.open("w") as f:
                for d in ("hotpotqa", "2wiki"):
                    for i in range(4):
                        f.write(json.dumps(source(i, d)) + "\n")
            protocol = ROOT / "configs/pcr_v1.json"
            pcr.prepare(argparse.Namespace(protocol=protocol, sources=sources, output=root / "plan", per_dataset=3))
            plan = load_json(root / "plan" / "plan.json")
            self.assertEqual(len(plan["question_ids"]), 6)
            self.assertTrue(all(s["target_turn"] in (1, 2) for s in plan["states"]))
            local = root / "local.json"
            write_json(local, {"agent_checkpoint": str(root), "retrieval_url": "http://x"})
            run_dir = root / "run"; (run_dir / "logs").mkdir(parents=True)
            write_json(run_dir / "run_plan.json", {"plan_path": str(root / "plan" / "plan.json"),
                       "plan_sha256": file_hash(root / "plan" / "plan.json"), "question_ids": plan["question_ids"],
                       "gpus": ["0", "1"], "backend": "vllm", "gpu_memory_utilization": .5, "request_timeout": 5,
                       "local_file": str(local), "local_file_sha256": file_hash(local), "repository_commit": "fixture"})
            write_json(run_dir / "run_manifest.json", {"status": "running"})
            for index in range(2):
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(index)}), \
                     mock.patch("work2.backends.VLLMGenerator", side_effect=lambda *a: OrderGenerator()), \
                     mock.patch("work2.retrieval.SearchR1Retriever", side_effect=lambda *a: DocRetriever()):
                    pcr.worker(run_dir, index)
            m = pcr.merge(run_dir)
            self.assertEqual(m["recorded_states"], len(plan["states"]))
            self.assertEqual(m["retrieval_drift"], 0)
            self.assertEqual(m["control_mismatch_states"], 0)
            self.assertEqual(pcr.crosscheck(argparse.Namespace(a=run_dir, b=run_dir, output=None))["identical_states"],
                             len(plan["states"]))
            # one receipted mismatch: refused under the v1 stop rule, excluded under record but over the 2% bound here
            path = run_dir / "workers" / "worker_0" / "states.jsonl"
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            rows[0].update(controls_exact=False, control_mismatch={"arms": {"clean_a": {"kind": "text"}}, "controls_agree": True})
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                pcr.merge(run_dir)
            write_json(run_dir / "run_plan.json", dict(load_json(run_dir / "run_plan.json"), control_mismatch="record"))
            with self.assertRaisesRegex(ValueError, "not measurable"):
                pcr.merge(run_dir)
            self.assertEqual(load_json(run_dir / "run_manifest.json")["status"], "control_reproducibility_failed")
            with mock.patch.object(pcr, "CONTROL_MISMATCH_MAX", 0.5):
                m = pcr.merge(run_dir)
            self.assertEqual((m["status"], m["control_mismatch_states"], m["all_controls_exact"]), ("completed", 1, False))


class Phase0Tests(unittest.TestCase):
    def test_collect_from_a2_style_branches(self):
        src = source()
        branches = {k: replay_branch(src, 1, k, OrderGenerator(), DocRetriever(), AGENT, RETRIEVAL)
                    for k in ("clean_a", "clean_b", "permuted_a")}
        case = {"sample_id": src["sample_id"], "dataset": src["dataset"], "target_turn": 1,
                "status": "completed", "branches": branches}
        rows, counters = phase0_luck.collect([src], [case], {})
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["y0"], rows[0]["y1"], rows[0]["kind"]), (1, 0, "semantic"))
        summary = phase0_luck.summarize(rows, 0)
        self.assertEqual(summary["E1_permutation"]["all"]["degraded_semantic"], 1)


if __name__ == "__main__":
    unittest.main()

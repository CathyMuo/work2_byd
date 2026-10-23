"""CPU tests for U2 (ReSeek judge forcing, MR-Search reflection turn); fake policies, no GPU."""
import argparse
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_credit import BatchRetriever
from work2 import u2
from work2 import u2_analysis as ua
from work2.backends import TokenizerMixin
from work2.common import ROOT, file_hash, load_json, write_json
from work2.credit_testbed import LiveSearcher
from work2.rollout import RESEEK_SYSTEM, RESEEK_USER_PREFIX, question_prompt

RESEEK = load_json(ROOT / "configs/u2_reseek.json")
MR = load_json(ROOT / "configs/u2_mrsearch.json")
# sha256 of DEFAULT_USER_CONTENT_PREFIX in TencentBAC/ReSeek utils/preprocess_reseek_dataset.py (read with ast on the home server).
RESEEK_PREFIX_SHA256 = "65dded3ed2e6afe73a3de448138ea64a5de436ebd334127701cabf3d9842de67"
OBS = "\n\n<information>"  # observation blocks only; the prompts themselves mention <information>



class ByteGen:
    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()

    def sample_batch(self, prompts, max_tokens, seeds, temperature, top_p=1.0, stop=None):
        outs = []
        for p, seed in zip(prompts, seeds):
            text = self._one(self.decode(p), seed, temperature, stop or ["</search>", "</answer>"])
            outs.append({"text": text, "token_ids": self.encode(text), "finish_reason": "stop"})
        return outs


class ReSeekGen(ByteGen):
    """Judges Yes iff Alpha is the FIRST document; after No searches again; greedy answers London, sampling answers Paris."""
    def initial_prompt(self, text, system=None):
        return "[{}]\n{}".format(system, text) if system else text

    def _one(self, text, seed, temperature, stop):
        seen = text.count(OBS)
        if seen == 0:
            return "<search>q0</search>"
        last_obs = text.rsplit(OBS, 1)[1].split("</information>", 1)[0]
        tail = text.rsplit("</information>", 1)[1]
        if "<judge>" in tail:
            value, head = u2.judge_of(tail), ""
        else:
            value = "Yes" if last_obs.startswith("Doc 1(Title: Alpha)") else "No"
            head = "<think>x</think><judge>{}</judge>".format(value)
            if "</judge>" in stop:
                return head
        if value == "No" and seen < 3:
            return head + "<search>q{}</search>".format(seen)
        return head + "<answer>{}</answer>".format("Paris" if temperature > 0 and value == "Yes" else "London")


class MRGen(ByteGen):
    """Episode 1 (greedy) answers London; reflect -> search then Paris; reask -> Paris; neutral -> London; fresh -> seed parity."""
    def initial_prompt(self, text, system=None):
        return text

    def user_turn(self, content):
        return "<|im_end|>\n\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(content)

    def _one(self, text, seed, temperature, stop):
        episode = text.split("<|im_start|>user")[-1] if "<|im_start|>user" in text else text
        seen = episode.count(OBS)
        if "<|im_start|>user" not in text:
            if temperature == 0:
                return "<search>q0</search>" if seen == 0 else "<answer>London</answer>"
            return "<answer>{}</answer>".format("Paris" if seed % 2 == 0 else "London")
        if "Reflect on your current answer" in episode:
            return "<search>again</search>" if seen == 0 else "<answer>Paris</answer>"
        if "Answer the question again." in episode:
            return "<answer>Paris</answer>"
        return "<answer>London</answer>"


class StubTokenizer:
    chat_template = "qwen"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        system = messages[0]["content"] if messages[0]["role"] == "system" else "You are a helpful assistant."
        users = [m for m in messages if m["role"] == "user"]
        return ("<|im_start|>system\n{}<|im_end|>\n".format(system)
                + "".join("<|im_start|>user\n{}<|im_end|>\n".format(m["content"]) for m in users) + "<|im_start|>assistant\n")


class Mixin(TokenizerMixin):
    tokenizer = StubTokenizer()


def sample(i=0, d="hotpotqa"):
    return {"sample_id": "{}:{}".format(d, i), "dataset": d, "question": "Where {}?".format(i), "gold_answer": ["Paris", "paris city"],
            "gold_supporting_facts": []}


class PromptTests(unittest.TestCase):
    def test_reseek_prompt_is_verbatim(self):
        self.assertEqual(hashlib.sha256(RESEEK_USER_PREFIX.encode()).hexdigest(), RESEEK_PREFIX_SHA256)
        user = question_prompt("Who?", 4, "reseek")
        self.assertTrue(user.endswith("    Question: Who?"))
        text = Mixin().initial_prompt(user, system=RESEEK_SYSTEM)
        self.assertTrue(text.startswith("<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>"))

    def test_mr_user_turn_matches_retry_str(self):
        turn = Mixin().user_turn(u2.MR_RETRY_PROMPT)
        self.assertEqual(turn, "<|im_end|>\n\n<|im_start|>user\n" + u2.MR_RETRY_PROMPT + "<|im_end|>\n<|im_start|>assistant\n")
        self.assertIn("search engines. \nYou must", u2.MR_RETRY_PROMPT)
        self.assertTrue(u2.MR_REASK_PROMPT.startswith("Answer the question again. \nYou must"))

    def test_protocols_load(self):
        self.assertEqual(u2.load_protocol(ROOT / "configs/u2_reseek.json")["protocol"], u2.RESEEK)
        self.assertEqual(u2.load_protocol(ROOT / "configs/u2_mrsearch.json")["protocol"], u2.MRSEARCH)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            write_json(path, dict(RESEEK, arms=["nat", "perm"]))
            with self.assertRaises(ValueError):
                u2.load_protocol(path)


class ReSeekTests(unittest.TestCase):
    def test_cut_at_judge_forces_value(self):
        gen = ReSeekGen()
        text = "<think>x</think><judge>Yes</judge>"
        self.assertEqual(gen.decode(u2.cut_at_judge(gen, gen.encode(text), text, "No")), "<think>x</think><judge>No</judge>")

    def test_question_arms(self):
        gen = ReSeekGen()
        row = u2.reseek_question(sample(), gen, LiveSearcher(BatchRetriever(), RESEEK["retrieval"]), RESEEK)
        self.assertEqual((row["source"]["search_calls"], row["source"]["em"], row["source_judges"]), (1, 0, ["Yes"]))
        st = row["states"][0]
        self.assertEqual(st["stage_a_judge"], ["Yes"] * 4)
        self.assertEqual(st["arms"]["force_no"]["first_judge"], ["No"] * 4)
        self.assertEqual(st["arms"]["force_no"]["extra_searches"], [1] * 4)
        self.assertEqual(st["arms"]["nat"]["em"], [1] * 4)
        self.assertEqual(row["branch_rollouts"], 16)


class MRTests(unittest.TestCase):
    def test_question_arms(self):
        gen = MRGen()
        row = u2.mrsearch_question(sample(), gen, LiveSearcher(BatchRetriever(), MR["retrieval"]), MR, u2.mr_turns(gen))
        self.assertEqual((row["episode1"]["em"], row["episode1"]["status"]), (0, "answered"))
        a = row["arms"]
        self.assertEqual((a["reflect"]["final_em"], a["reask"]["final_em"], a["neutral"]["final_em"]), ([1] * 4, [1] * 4, [0] * 4))
        self.assertEqual(a["reflect"]["ep2_searches"], [1] * 4)
        self.assertEqual(a["reflect"]["changed"], [True] * 4)


def run_pipeline(root, protocol_path, gen_factory, label):
    protocol = load_json(protocol_path)
    fresh = root / "u1_fresh"
    if not fresh.exists():
        fresh.mkdir()
        rows = [sample(i, d) for d in ("hotpotqa", "nq", "bamboogle") for i in range(4)]
        (fresh / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(fresh / "selection_manifest.json", {"samples_sha256": file_hash(fresh / "samples.jsonl")})
        u2.prepare(argparse.Namespace(samples_dir=fresh, per_dataset=3, seed=1, output=root / "plan"))
    local = root / "local.json"; write_json(local, {"agent_checkpoint": str(root / "m"), "retrieval_url": "http://x"})
    plan = load_json(root / "plan" / "plan.json")
    run_dir = root / ("run_" + label); (run_dir / "logs").mkdir(parents=True)
    write_json(run_dir / "run_plan.json", {"protocol": protocol, "protocol_sha256": u2.sha(protocol), "agent_label": label,
               "plan_path": str(root / "plan" / "plan.json"), "plan_sha256": file_hash(root / "plan" / "plan.json"),
               "question_ids": [q["sample_id"] for q in plan["questions"]], "gpus": ["0", "1"], "gpu_memory_utilization": .5,
               "request_timeout": 5, "local_file": str(local), "local_file_sha256": file_hash(local), "repository_commit": "fixture"})
    write_json(run_dir / "run_manifest.json", {"status": "running"})
    for index in range(2):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(index)}), \
             mock.patch("work2.backends.VLLMGenerator", side_effect=lambda *a: gen_factory()), \
             mock.patch("work2.retrieval.SearchR1Retriever", side_effect=lambda *a: BatchRetriever()):
            u2.worker(run_dir, index)
    u2.merge(run_dir)
    return run_dir


class PipelineTests(unittest.TestCase):
    def test_prepare_run_merge_analyze_both_audits(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(u2, "git_output", return_value="fixture"), \
             mock.patch.object(ua, "git_output", return_value="fixture"):
            root = Path(tmp)
            run_r = run_pipeline(root, ROOT / "configs/u2_reseek.json", ReSeekGen, "fake_reseek")
            plan = load_json(root / "plan" / "plan.json")
            self.assertEqual(len(plan["questions"]), 9)
            self.assertEqual(len(plan["smoke_question_ids"]), 6)
            r = ua.analyze(run_r, resamples=50)
            self.assertTrue(r["fidelity"]["passes"])
            self.assertEqual(r["sets"]["S_fail"]["n_states"], 9)
            self.assertAlmostEqual(r["sets"]["S_fail"]["explained_by_nat"]["estimate"], 1.0)
            self.assertTrue(r["gates"]["H1_reseek_resample"])
            self.assertIn("U2", ua.markdown(r))
            run_m = run_pipeline(root, ROOT / "configs/u2_mrsearch.json", MRGen, "fake_mr")
            self.assertIn("reflect", load_json(run_m / "run_manifest.json")["user_turns"])
            m = ua.analyze(run_m, resamples=50)
            w = m["sets"]["S_wrong"]
            self.assertEqual(w["n_questions"], 9)
            self.assertAlmostEqual(w["value"]["reflect"]["estimate"], 1.0)
            self.assertAlmostEqual(w["explained_by_neutral"]["estimate"], 0.0)
            self.assertFalse(m["gates"]["H1_mr_neutral"])
            self.assertTrue(m["fidelity"]["passes"])
            self.assertIn("reflect_minus_neutral", ua.markdown(m))


if __name__ == "__main__":
    unittest.main()

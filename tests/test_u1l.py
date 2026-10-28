"""CPU tests for the U1L ReSearch format (system prompt, <result> blocks, boxed answers) and its PCR replay."""
import hashlib
import re
import tempfile
import unittest
from pathlib import Path

from test_pcr import DocRetriever, sample
from work2 import pcr
from work2 import credit_testbed as ct
from work2.common import ROOT, load_json, write_json
from work2.placebo import ARM_GROUPS, run_state
from work2.rollout import RESEARCH_SYSTEM, initial_text, last_boxed, observation_text, run_sample

PCR_RESEARCH = load_json(ROOT / "configs/pcr_u1_research.json")
AGENT = dict(PCR_RESEARCH["agent"], max_context_tokens=30000)
RETRIEVAL = PCR_RESEARCH["retrieval"]
# sha256 of re_search_template_sys (Agent-RL/ReCall at 7fcf4d3~1), read with ast on the home server.
RESEARCH_SYSTEM_SHA256 = "87e802b6736cb3165ea6a644d4fe2f2c7118b21828103c9eeb338ce0c7f8b23b"
OBS = re.compile(r" <result>\n")


class ResearchGen:
    """Searches 1 + (i % 2) times; answers Paris (boxed) iff Alpha is the first document of the last result block."""
    def encode(self, text):
        return list(text.encode())

    def initial_prompt(self, text, system=None):
        return "[SYS]{}[/SYS]\n{}".format(system, text) if system else text

    def generate(self, tokens, limit, seed):
        text = bytes(tokens).decode()
        seen = len(OBS.findall(text))
        wanted = 1 + int(re.search(r"Where (\d+)\?", text).group(1)) % 2
        if seen < wanted:
            out = "<think> need facts </think> <search>q{}</search>".format(seen)
        else:
            last = OBS.split(text)[-1]
            good = last.startswith("Alpha\nAlpha fact.") or "briefly review" in last
            out = "<think> done </think> <answer> The final answer is \\[ \\boxed{{{}}} \\] </answer>".format("Paris" if good else "London")
        return {"text": out, "token_ids": self.encode(out), "finish_reason": "stop"}

    def sample_batch(self, prompts, max_tokens, seeds, temperature, top_p=1.0, stop=None):
        return [self.generate(p, m, s) for p, m, s in zip(prompts, max_tokens, seeds)]


class FormatTests(unittest.TestCase):
    def test_system_prompt_is_verbatim(self):
        self.assertEqual(hashlib.sha256(RESEARCH_SYSTEM.encode()).hexdigest(), RESEARCH_SYSTEM_SHA256)
        text = initial_text(ResearchGen(), "Who?", AGENT)
        self.assertTrue(text.startswith("[SYS]You are a helpful assistant that can solve") and text.endswith("\nWho?"))

    def test_observation_and_boxed_answer(self):
        docs = DocRetriever().search("q", 5)
        obs = observation_text(docs, "research")
        self.assertTrue(obs.startswith(" <result>\nAlpha\nAlpha fact.\n\nBeta\n") and obs.endswith("Filler5\nFiller text 5.\n</result>"))
        self.assertEqual(last_boxed("The final answer is \\[ \\boxed{Paris} \\]"), "Paris")
        self.assertEqual(last_boxed("\\boxed{x} then \\boxed{a{b}c}"), "a{b}c")
        self.assertEqual(last_boxed("no box here"), "")

    def test_source_and_h1_replay(self):
        src = run_sample(sample(1), ResearchGen(), DocRetriever(), AGENT, RETRIEVAL)
        self.assertEqual((src["prompt_version"], src["status"], src["final_answer"], src["em"], src["search_calls"]),
                         ("research_official_sys_v1", "answered", "Paris", 1, 2))
        state = run_state(src, 1, ARM_GROUPS["h1"], ResearchGen(), DocRetriever(), AGENT, RETRIEVAL)
        self.assertTrue(state["controls_exact"])
        self.assertTrue(state["arms"]["perm_1"]["observation_changed"])

    def test_batched_engine_extracts_boxed(self):
        gen = ResearchGen()
        r = ct.new_rollout(sample(0), gen.encode(initial_text(gen, "Where 0?", AGENT)), 0, 1)
        ct.batched_continue([r], gen, ct.LiveSearcher(DocBatch(), RETRIEVAL), AGENT, RETRIEVAL, 1.0, 1.0)
        self.assertEqual((r["final_answer"], r["em"]), ("Paris", 1))

    def test_pcr_accepts_research_budget_only(self):
        self.assertEqual(pcr.load_protocol(ROOT / "configs/pcr_u1_research.json")["agent"]["max_search_turns"], 8)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "p.json"
            write_json(bad, dict(PCR_RESEARCH, agent=dict(PCR_RESEARCH["agent"], max_search_turns=5)))
            with self.assertRaises(ValueError):
                pcr.load_protocol(bad)
        roster = load_json(ROOT / "configs/u1l_agents.json")["agents"]
        self.assertEqual([a["label"] for a in roster], ["v03_32b_grpo", "research_32b", "v03_14b_ppo", "v03_14b_it_grpo",
                                                         "qwen25_32b_inst", "research_7b", "qwen25_7b_inst"])


class DocBatch(DocRetriever):
    def search_batch(self, queries, top_k):
        return [self.search(q, top_k) for q in queries]


if __name__ == "__main__":
    unittest.main()

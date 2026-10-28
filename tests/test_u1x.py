"""CPU tests for U1x agent formats (ZeroSearch / AutoRefine / StepSearch) and Llama BOS handling; no GPU."""
import hashlib
import re
import unittest

from test_pcr import DocRetriever, sample
from work2 import audit_confirm
from work2.backends import TokenizerMixin
from work2.common import ROOT, load_json
from work2.placebo import ARM_GROUPS, run_state
from work2.replay import PROMPT_VARIANT_OF, observation
from work2.rollout import (AUTOREFINE_TEMPLATE, FORMATS, STEPSEARCH_TEMPLATE, observation_text, question_prompt,
                           run_sample)

RETRIEVAL = {"visible_top_k": 3, "candidate_top_k": 20}
# sha256 of the effective make_prefix templates read with ast on the home server ("{question}" placeholder kept).
AUTOREFINE_SHA256 = "7e3f4930396a84c99021fe814dff93f09ebf8048f895af7367bc684c9008985e"
STEPSEARCH_SHA256 = "0c376357466e32746a47379aea8da6576f807c54ae4073dc72f01aea7bbff7de"
OBS = re.compile(r"\n\n<(?:information|documents)>")


def agent(variant):
    return {"max_search_turns": 4, "max_context_tokens": 20000, "max_generation_tokens_per_turn": 512, "seed": 42,
            "temperature": 0, "prompt_variant": variant}


class StyleGen:
    """Answers correctly only when Alpha is the first document of the last observation, in any agent style."""
    def encode(self, text):
        return list(text.encode())

    def initial_prompt(self, text):
        return text

    def generate(self, tokens, limit, seed):
        text = bytes(tokens).decode()
        seen = len(OBS.findall(text))
        wanted = 1 + int(re.search(r"Where (\d+)\?", text).group(1)) % 2
        if seen < wanted:
            out = "<search>q{}</search>".format(seen)
        else:
            last = OBS.split(text)[-1].lstrip()
            good = last.startswith("Doc 1(Title: Alpha)") or last.startswith("Doc 1<## Title: Alpha ##>") or "briefly review" in last
            out = "<answer>{}</answer>".format("Paris" if good else "London")
        return {"text": out, "token_ids": self.encode(out), "finish_reason": "stop"}


class Tok:
    def __init__(self, adds_bos):
        self.chat_template, self.bos_token, self.bos_token_id, self.adds_bos = None, "<|begin_of_text|>", 128000, adds_bos

    def encode(self, text, add_special_tokens=True):
        return ([self.bos_token_id] if add_special_tokens and self.adds_bos else []) + [ord(c) for c in text]


class Mixin(TokenizerMixin):
    def __init__(self, tok):
        self.tokenizer = tok


class FormatTests(unittest.TestCase):
    def test_templates_are_verbatim(self):
        self.assertEqual(hashlib.sha256(AUTOREFINE_TEMPLATE.encode()).hexdigest(), AUTOREFINE_SHA256)
        self.assertEqual(hashlib.sha256(STEPSEARCH_TEMPLATE.encode()).hexdigest(), STEPSEARCH_SHA256)
        self.assertTrue(question_prompt("Who {x}?", 4, "autorefine").endswith("Question: Who {x}?\n"))
        self.assertTrue(question_prompt("Who?", 4, "stepsearch").endswith("Question:Who?\n"))

    def test_observation_styles(self):
        docs = DocRetriever().search("q", 3)
        self.assertEqual(observation(docs), observation_text(docs, "bounded"))
        self.assertTrue(observation(docs).endswith("Gamma text here.\n</information>\n\n"))   # historical, unstripped
        a = observation_text(docs, "autorefine")
        self.assertTrue(a.startswith("\n\n<documents>Doc 1(Title: Alpha) Alpha fact.\n") and a.endswith("Gamma text here.</documents>\n\n"))
        s = observation_text(docs, "stepsearch")
        self.assertTrue(s.startswith("\n\n<information> Doc 1<## Title: Alpha ##> Alpha fact.\n") and s.endswith("Gamma text here. </information>\n\n"))
        self.assertEqual(set(PROMPT_VARIANT_OF.values()), {"bounded", "official", "autorefine", "stepsearch", "research"})

    def test_sources_and_h1_arms_replay_in_each_style(self):
        for variant in ("official", "autorefine", "stepsearch"):
            src = run_sample(sample(1), StyleGen(), DocRetriever(), agent(variant), RETRIEVAL)
            self.assertEqual((src["prompt_version"], src["status"], src["search_calls"]), (FORMATS[variant]["version"], "answered", 2))
            self.assertIn(observation_text(src["prefixes"][0]["visible_documents"], variant), "".join(
                [src["initial_prompt"]] + [s["text"] + p["observation_text"] for s, p in zip(src["steps"], src["prefixes"])]))
            state = run_state(src, 1, ARM_GROUPS["h1"], StyleGen(), DocRetriever(), agent(variant), RETRIEVAL)
            self.assertTrue(state["controls_exact"])
            self.assertTrue(state["arms"]["perm_1"]["observation_changed"])
            self.assertEqual(state["arms"]["clean_a"]["em"], 1)

    def test_fidelity_and_pcr_configs_load(self):
        for v in ("autorefine", "stepsearch"):
            self.assertEqual(audit_confirm.load_protocol(ROOT / "configs/fidelity_{}.json".format(v))["agent"]["prompt_variant"], v)
            self.assertEqual(load_json(ROOT / "configs/pcr_u1_{}.json".format(v))["arm_group"], "h1")
        roster = load_json(ROOT / "configs/u1x_agents.json")["agents"]
        self.assertEqual({a["prompt_variant"] for a in roster}, {"official", "autorefine", "stepsearch"})


class FidelityTests(unittest.TestCase):
    def test_gap_and_flags(self):
        import json, tempfile
        from pathlib import Path
        from work2 import u1x
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "stepsearch_7b" / "sources"; d.mkdir(parents=True)
            rows = [{"dataset": "hotpotqa", "em": int(i < 2), "status": "answered"} for i in range(10)]
            rows += [{"dataset": "2wiki", "em": 0, "status": "malformed_action"} for _ in range(2)]
            (d / "sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            f = u1x.agent_fidelity(tmp, "stepsearch_7b")
            self.assertAlmostEqual(f["gap_by_dataset"]["hotpotqa"], 0.2 - 0.380)
            self.assertTrue(f["possibly_misharnessed"])
            self.assertTrue(f["excluded_for_format_failure"])
            self.assertIsNone(u1x.agent_fidelity(tmp, "missing"))


class BosTests(unittest.TestCase):
    def test_bos_added_only_when_the_tokenizer_adds_it(self):
        self.assertEqual(Mixin(Tok(True)).initial_prompt("Q"), "<|begin_of_text|>Q")
        self.assertEqual(Mixin(Tok(False)).initial_prompt("Q"), "Q")


if __name__ == "__main__":
    unittest.main()

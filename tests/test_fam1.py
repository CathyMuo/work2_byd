"""CPU tests for FAM1: the R1-Searcher format, the inserted-text protocol tag, pool E selection and the F1-F8 analysis."""
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from test_pcr import DocRetriever, OrderGenerator, sample, source
from work2 import audit_confirm, fam1
from work2.backends import STOP_STRINGS, STOP_STRINGS_OF, TokenizerMixin
from work2.common import ROOT, file_hash, load_json, write_json
from work2.placebo import ARM_GROUPS, REVIEW_PROMPT, protocol_text, run_state
from work2.replay import PROMPT_VARIANT_OF
from work2.rollout import (FORMATS, R1SEARCHER_TEMPLATE, initial_text, observation_text, parse_action, question_prompt,
                           run_sample)

RETRIEVAL = {"visible_top_k": 3, "candidate_top_k": 20}
# sha256 of base_prompt_v0 read with ast from RUCAIBox/R1-Searcher ad86a6692be0d7fc4445c77938c090349815e997,
# evaluation/eval_search_loacl.py (file sha256 c313d59f...171b), function process_text; "{question}" placeholder kept.
R1SEARCHER_SHA256 = "10e11a97464126d50c5eb2a4b59a6a7534e3763d7e8f782ffc3a6c07b15aed93"
AGENT = {"max_search_turns": 4, "max_context_tokens": 20000, "max_generation_tokens_per_turn": 512, "seed": 42,
         "temperature": 0, "prompt_variant": "r1searcher"}


class R1Gen:
    """R1-Searcher style: searches 1 + (question number % 2) times, answers right only when Alpha is the first document of
    the last observation or a review prompt follows it. Fails if anything asks it for a chat-template prompt."""
    def encode(self, text):
        return list(text.encode())

    def raw_prompt(self, text):
        return text

    def initial_prompt(self, text, system=None):
        raise AssertionError("R1-Searcher must not use a chat template")

    def generate(self, tokens, limit, seed):
        text = bytes(tokens).decode()
        seen = text.count("<|begin_of_documents|>") - 1   # the prompt itself names the tag once
        wanted = 1 + int(re.search(r"Where (\d+)\?", text).group(1)) % 2
        if seen < wanted:
            out = ' I need "q{}"\tmore... <|begin_of_query|> "q{}"\tfact... <|end_of_query|>'.format(seen, seen)
        else:
            last = text.split("<|begin_of_documents|>\n")[-1]
            good = last.startswith("(1)Alpha") or "briefly review" in last or "Paris" in last.split("<|end_of_documents|>")[-1]
            out = " done </think>\n\n<answer> {} </answer>".format("Paris" if good else "London")
        return {"text": out, "token_ids": self.encode(out), "finish_reason": "stop"}


class Tok:
    chat_template, bos_token, bos_token_id = "{{ messages }}", "<|begin_of_text|>", 128000

    def encode(self, text, add_special_tokens=True):
        return ([self.bos_token_id] if add_special_tokens else []) + [ord(c) for c in text]


class Mixin(TokenizerMixin):
    def __init__(self):
        self.tokenizer = Tok()


class FormatTests(unittest.TestCase):
    def test_template_is_verbatim_and_raw(self):
        self.assertEqual(hashlib.sha256(R1SEARCHER_TEMPLATE.encode()).hexdigest(), R1SEARCHER_SHA256)
        p = question_prompt("Who {x}?", 4, "r1searcher")
        self.assertTrue(p.endswith("\n\nUser:Who {x}?\nAssistant: <think>"))
        # A tokenizer WITH a chat template still gets the raw completion prompt, with BOS as vLLM would add it.
        self.assertEqual(initial_text(Mixin(), "Q?", AGENT), "<|begin_of_text|>" + p.replace("Who {x}?", "Q?"))
        self.assertIn("r1searcher", PROMPT_VARIANT_OF.values())
        self.assertEqual(STOP_STRINGS_OF["r1searcher"], ["<|end_of_query|>", "</answer>"])
        self.assertEqual(STOP_STRINGS, ["</search>", "</answer>"])

    def test_actions(self):
        v = "r1searcher"
        self.assertEqual(parse_action('x <|begin_of_query|> "a"\tb... <|end_of_query|>', v), ("search", "a b"))
        self.assertEqual(parse_action("<|begin_of_query|> old <|end_of_query|> <|begin_of_query|> new <|end_of_query|>", v), ("search", "new"))
        self.assertEqual(parse_action("<think>x</think>\n\n<answer> Paris </answer>", v), ("answer", "Paris"))
        self.assertEqual(parse_action("<|begin_of_query|> '' <|end_of_query|>", v), (None, None))
        self.assertEqual(parse_action("no action at all", v), (None, None))
        self.assertEqual(parse_action("<search> q </search>", v), (None, None))
        self.assertEqual(parse_action("<search> q </search>"), ("search", "q"))   # every other variant unchanged

    def test_observation(self):
        docs = DocRetriever().search("q", 3)
        self.assertEqual(observation_text(docs, "r1searcher"),
                         "\n\n<|begin_of_documents|>\n(1)Alpha\nAlpha fact.\n(2)Beta\nBeta text here.\n(3)Gamma\nGamma text here.\n"
                         "<|end_of_documents|>\n\n")

    def test_protocol_sentence_names_the_agent_tag(self):
        self.assertEqual(protocol_text(REVIEW_PROMPT, "bounded"), REVIEW_PROMPT)
        self.assertEqual(protocol_text(REVIEW_PROMPT, "official"), REVIEW_PROMPT)
        r = protocol_text(REVIEW_PROMPT, "r1searcher")
        self.assertIn("<|begin_of_query|> or <answer>", r)
        self.assertNotIn("<search>", r)

    def test_sources_and_arms_replay(self):
        src = run_sample(sample(1), R1Gen(), DocRetriever(), AGENT, RETRIEVAL)
        self.assertEqual((src["prompt_version"], src["status"], src["search_calls"], src["em"]),
                         (FORMATS["r1searcher"]["version"], "answered", 2, 1))
        self.assertEqual(src["prefixes"][0]["search_query"], "q0 fact")
        for group in ("h1", "assay", "natural"):
            state = run_state(src, 1, ARM_GROUPS[group], R1Gen(), DocRetriever(), AGENT, RETRIEVAL)
            self.assertTrue(state["controls_exact"])
            self.assertEqual(state["arms"]["clean_a"]["em"], 1)
        state = run_state(src, 1, ARM_GROUPS["h1"], R1Gen(), DocRetriever(), AGENT, RETRIEVAL)
        meta = state["arms"]["perm1_review"]["metadata"]
        self.assertIn("<|begin_of_query|>", meta["instruction"])
        self.assertEqual(state["arms"]["perm1_neutral"]["metadata"]["instruction_tokens"], meta["instruction_tokens"])

    def test_configs_and_roster(self):
        self.assertEqual(audit_confirm.load_protocol(ROOT / "configs/fidelity_r1searcher.json")["agent"]["prompt_variant"], "r1searcher")
        for kind, group in (("pcr_u1", "h1"), ("as1", "assay"), ("u1n", "natural")):
            p = load_json(ROOT / "configs/{}_r1searcher.json".format(kind))
            self.assertEqual((p["arm_group"], p["agent"]["prompt_variant"]), (group, "r1searcher"))
        roster = load_json(ROOT / "configs/fam1_agents.json")["agents"]
        self.assertEqual({a["label"]: a["prompt_variant"] for a in roster if a["prompt_variant"] != "bounded"}, {"r1s_l8b": "r1searcher"})
        self.assertEqual(sum(a["role"] == "new_nonqwen" for a in roster), 5)
        self.assertEqual(sum(a["role"] == "matched_qwen" for a in roster), 4)
        self.assertEqual(len(fam1.QWEN_MAIN), 13)
        script = (ROOT / "scripts/run_fam1.sh").read_text()
        for a in roster:
            self.assertIn(a["label"], script)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class PoolETests(unittest.TestCase):
    def test_selection_excludes_history_and_dedups(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            hot = [{"_id": "h{}".format(i), "question": "Hot question {}?".format(i), "answer": "a"} for i in range(8)]
            wiki = [{"_id": "w{}".format(i), "question": "Wiki question {}?".format(i), "answer": "a"} for i in range(8)]
            wiki.append({"_id": "w-dup", "question": "hot QUESTION 5?", "answer": "a"})      # same text as a hotpot question
            nq = [{"id": "n{}".format(i), "question": "Nq question {}?".format(i), "golden_answers": ["a"]} for i in range(8)]
            mu = [{"id": "m{}".format(i), "question": "Mu question {}?".format(i), "golden_answers": ["a"]} for i in range(8)]
            (t / "hot.json").write_text(json.dumps(hot)); (t / "wiki.json").write_text(json.dumps(wiki))
            jsonl(t / "nq.jsonl", nq); jsonl(t / "mu.jsonl", mu)
            local = {"hotpotqa_file": str(t / "hot.json"), "two_wiki_file": str(t / "wiki.json"),
                     "nq_file": str(t / "nq.jsonl"), "musique_file": str(t / "mu.jsonl")}
            used = [{"sample_id": "hotpotqa:h0", "question": "Hot question 0?", "dataset": "hotpotqa"},
                    {"sample_id": "nq:other", "question": "nq   question 1?", "dataset": "nq"}]      # blocked by text
            jsonl(t / "outputs/u1_fresh/samples.jsonl", used)
            jsonl(t / "outputs/u2/x/samples.jsonl", [{"sample_id": "musique:m2", "question": "?", "dataset": "musique"}])
            ids, questions, receipts = fam1.history(t / "outputs", [], [], ())
            self.assertEqual(len(receipts), 2)
            chosen, counts, _ = fam1.select_e(local, ids, questions, per_dataset=8)
            got = {r["sample_id"] for r in chosen}
            self.assertNotIn("hotpotqa:h0", got); self.assertNotIn("nq:n1", got); self.assertNotIn("musique:m2", got)
            self.assertEqual(counts, {"hotpotqa": 7, "2wiki": 8, "nq": 7, "musique": 7})   # w-dup dropped as a duplicate
            self.assertEqual(len({fam1.question_key(r["question"]) for r in chosen}), len(chosen))
            again, _, _ = fam1.select_e(local, ids, questions, per_dataset=3)
            self.assertEqual([r["sample_id"] for r in again], [r["sample_id"] for r in fam1.select_e(local, ids, questions, per_dataset=3)[0]])
            jsonl(t / "outputs/bad/samples.jsonl", [{"x": 1}])
            with self.assertRaises(ValueError):
                fam1.history(t / "outputs", [], [], ())
            self.assertEqual(len(fam1.history(t / "outputs", [], [t / "outputs/bad/samples.jsonl"], ())[2]), 2)
            with self.assertRaises(ValueError):
                fam1.history(t / "elsewhere", [], [], ())


def write_run(root, label, n=6, groups=("hotpotqa", "2wiki")):
    srcs = [source(i, d) for d in groups for i in range(n)]
    jsonl(root / label / "sources" / "sources.jsonl", srcs)
    states = [run_state(s, 1, ARM_GROUPS["h1"], OrderGenerator(), DocRetriever(), dict(s_agent()), RETRIEVAL) for s in srcs]
    run = root / label / "pcr"
    jsonl(run / "states.jsonl", states)
    write_json(run / "run_manifest.json", {"status": "completed", "states_sha256": file_hash(run / "states.jsonl")})


def s_agent():
    return {"max_search_turns": 4, "max_context_tokens": 20000, "max_generation_tokens_per_turn": 512, "seed": 42,
            "temperature": 0, "prompt_variant": "bounded"}


class AnalysisTests(unittest.TestCase):
    def test_boot_sums_matches_plain_means(self):
        rows = [{"dataset": d, "sample_id": str(i), "v": i % 3} for d in ("a", "b") for i in range(10)]
        e = fam1.boot_sums(rows, lambda r: [r["v"], 1], lambda t: t[0] / t[1], 200)
        self.assertAlmostEqual(e["estimate"], sum(r["v"] for r in rows) / len(rows))
        self.assertLessEqual(e["ci95"][0], e["estimate"]); self.assertGreaterEqual(e["ci95"][1], e["estimate"])

    def test_luck_gap_formula(self):
        mk = lambda i, y0, p1, p2: {"dataset": "a" if i % 2 else "b", "sample_id": str(i), "y0": y0,
                                    "sem": dict({a: 0 for a in fam1.NEED}, perm_1=p1, perm_2=p2)}
        d = [mk(i, 1, 0, int(i < 3)) for i in range(6)]          # 3 of 6 degraded states repaired by P1
        s = [mk(10 + i, 1, 1, int(i >= 2)) for i in range(12)]   # 2 of 12 stable states harmed by P1
        e = fam1.luck_gap(d + s, "sem", 0)
        self.assertAlmostEqual(e["estimate"], 3 / 6 - 2 / 6)

    def test_readings_rules(self):
        est = lambda v, l, h: {"estimate": v, "ci95": [l, h]}
        method = lambda rep_lo, rep_hi: {"repair": 0.3, "P2_repair": 0.27, "repair_minus_P2": est(0.03, rep_lo, rep_hi),
                                         "repair_minus_P1": est(-0.1, -0.2, -0.01), "harm_minus_P2": est(0.03, 0.01, 0.05),
                                         "all_minus_P2": est(-0.005, -0.01, 0.0)}
        block = {"F1_luck_gap": est(0.01, -0.05, 0.06), "methods": {"review": method(-0.02, 0.08), "reflect": method(0.05, 0.2)}}
        vs = {"review": {"repair_minus_P2": est(0.0, -0.05, 0.05)}, "reflect": {"repair_minus_P2": est(0.1, 0.02, 0.2)}}
        r = fam1.readings(block, vs)
        self.assertTrue(r["F1"] and r["F2"] and r["F3"] and r["F5"] and r["title_kept"])
        self.assertEqual(r["F4"]["review"]["verdict"], "generalizes")
        self.assertEqual(r["F4"]["reflect"]["verdict"], "family_dependent")
        vs["reflect"]["repair_minus_P2"] = est(0.05, -0.01, 0.12)
        self.assertEqual(fam1.readings(block, vs)["F4"]["reflect"]["verdict"], "undetermined")

    def test_analyze_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, "v01_l3b_ppo")
            write_run(root, "v03_3b_ppo")
            r = fam1.analyze(root, ROOT / "configs/fam1_agents.json", 50)
            self.assertEqual(r["included_nonqwen"], ["v01_l3b_ppo"])
            self.assertEqual(r["agent_status"]["r1s_l8b"], "not_run")
            v = r["views"]["sem"]["versions"]["primary"]
            self.assertIn(v["readings"]["F4"]["review"]["verdict"], ("generalizes", "family_dependent", "undetermined"))
            self.assertIn("F4", fam1.markdown(r))
            with self.assertRaises(ValueError):
                fam1.analyze(Path(tmp) / "empty", ROOT / "configs/fam1_agents.json", 10)


if __name__ == "__main__":
    unittest.main()

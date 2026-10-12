"""CPU synthetic reviews test protocol wiring, not semantic entailment or GPU results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_e
from test_a1 import FakeGenerator
from test_a2 import DistinctRetriever
from work2.common import file_hash, load_json, write_json
from work2.prepare_e import reload_plan, validate_plan
from work2.prepare_e_probe import build_probe, reload_probe_plan, validate_probe
from work2.run_e_probe import main, metrics


class EProbeTests(unittest.TestCase):
    def fixture(self, root):
        helper = test_e.ETests(methodName="runTest")
        helper.setUp()
        expected, context = helper.fixture(root)
        parent = helper.finalized(expected)
        for group in parent["groups"]:
            group.update(decision="skip", reason="Synthetic stable insufficiency; no semantic certification")
            for case in group["stable_candidates"]:
                case["event"]["review"].update(decision="insufficient", earliest_sufficient_turn=None,
                                               supporting_quotes=[], prefix_only_verified=True)
        parent_path = root/"v1_reviewed.json"
        write_json(parent_path, parent)
        plan, context = build_probe(parent_path)
        return plan, context, parent_path, expected

    def finalized(self, plan):
        plan = copy.deepcopy(plan)
        for case in plan["cases"]:
            case.update(decision="run", reason="Synthetic frozen development case")
            review = case["event"]["review"]
            review["prefix_only_verified"] = True
            if case["role"] == "stable_insufficient":
                review.update(missing_question_constraints=["Synthetic missing original-question relation"],
                              alternative_derivations_considered="CPU fixture only: this attestation tests validation, not entailment")
        return plan

    def test_v1_all_skip_preserved_and_gate_cannot_be_reinterpreted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan,_,parent,expected = self.fixture(root)
            parent_hash = file_hash(parent)
            self.assertEqual(reload_plan(parent)[1],[])
            self.assertEqual([c["role"] for c in plan["cases"]],["failed_sufficient","stable_insufficient"])
            self.assertEqual(len(validate_probe(self.finalized(plan),plan)),2)
            original = load_json(parent)
            original["groups"][0]["decision"] = "run"
            with self.assertRaisesRegex(ValueError,"no independently reviewed sufficient stable"):
                validate_plan(original,expected)
            self.assertEqual(file_hash(parent),parent_hash)

    def test_secondary_needs_constraints_and_alternative_derivations(self):
        with tempfile.TemporaryDirectory() as folder:
            plan,_,_,_ = self.fixture(Path(folder))
            for field in ("missing_question_constraints","alternative_derivations_considered"):
                reviewed = self.finalized(plan)
                reviewed["cases"][1]["event"]["review"][field] = [] if field.startswith("missing") else ""
                with self.assertRaisesRegex(ValueError,"explicit missing constraints"):
                    validate_probe(reviewed,plan)
            reviewed = self.finalized(plan)
            reviewed["cases"][1]["event"]["review"]["decision"] = "uncertain"
            with self.assertRaisesRegex(ValueError,"uncertainty must be skipped"):
                validate_probe(reviewed,plan)
            reviewed["cases"][1].update(decision="skip", reason="Derivation unresolved; harm remains unmeasured")
            self.assertEqual([c["role"] for c in validate_probe(reviewed,plan)],["failed_sufficient"])

    def test_frozen_context_parent_hash_and_secondary_only_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan,_,parent,_ = self.fixture(root)
            reviewed = self.finalized(plan)
            reviewed["cases"][0]["event"]["question"] = "Different question"
            with self.assertRaisesRegex(ValueError,"context changed"):
                validate_probe(reviewed,plan)
            reviewed = self.finalized(plan)
            reviewed["cases"][0].update(decision="skip", reason="No primary")
            with self.assertRaisesRegex(ValueError,"secondary-only"):
                validate_probe(reviewed,plan)
            path = root/"v2_reviewed.json"
            write_json(path,self.finalized(plan))
            parent.write_text(parent.read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"v1 reviewed plan changed"):
                reload_probe_plan(path)

    def run_cli(self, root, plan, responses, check_only=False):
        path = root/"v2_reviewed.json"
        write_json(path,plan)
        local = root/"local.json"
        write_json(local,{"retrieval_url":"http://localhost/retrieve"})
        run = root/"run"
        argv = ["e_probe","--plan",str(path),"--local",str(local),"--run-dir",str(run),"--group-limit","1"]
        if check_only:
            argv.append("--check-only")
        generator = FakeGenerator(responses)
        with mock.patch("sys.argv",argv),mock.patch("work2.run_e.git_output",side_effect=lambda *a:"" if a[0]=="status" else "fixture"), \
            mock.patch("work2.run_e.check_runtime",return_value=(root,{"actual_index_type":"flat"})) as runtime, \
            mock.patch("work2.run_e.SearchR1Retriever",return_value=DistinctRetriever()), \
            mock.patch("work2.backends.VLLMGenerator",return_value=generator) as backend:
            result = main()
            if check_only:
                runtime.assert_not_called()
                backend.assert_not_called()
                self.assertFalse(run.exists())
        return result,run,generator

    def test_cli_controls_first_strata_cost_contrasts_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan,_,parent,_ = self.fixture(root)
            hashes = {str(p):file_hash(p) for p in [parent,root/"a3.jsonl",root/"a1/clean_rollouts.jsonl"]}
            # Original controls failed(2), stable(2); failed general wrong/constraint correct;
            # stable arm order is constraint wrong/general correct.
            responses = ["<answer>wrong</answer>"]*2+["<answer>Fact 1</answer>"]*2+[
                "<answer>wrong</answer>","<answer>Fact 1</answer>","<answer>wrong</answer>","<answer>Fact 1</answer>"]
            result,run,generator = self.run_cli(root,self.finalized(plan),responses)
            self.assertEqual(result,0)
            receipt = load_json(run/"mechanism_metrics.json")
            self.assertEqual(receipt["by_role"]["failed_sufficient"]["constraint_review"]["gains"],1)
            self.assertEqual(receipt["by_role"]["stable_insufficient"]["constraint_review"]["harms"],1)
            self.assertEqual(receipt["by_role"]["failed_sufficient"]["constraint_minus_general"]["mean_delta_em"],1)
            self.assertEqual(receipt["by_role"]["stable_insufficient"]["constraint_minus_general"]["mean_delta_em"],-1)
            self.assertTrue(all(receipt["engineering_gate"].values()))
            self.assertNotIn("net_em",receipt)
            manifest = load_json(run/"run_manifest.json")
            self.assertEqual(manifest["stage"],"E_stratified_probe")
            self.assertEqual(len(manifest["frozen_v2_case_decisions"]),2)
            for tokens in generator.inputs:
                self.assertNotIn("REVIEW_ONLY_SECRET",bytes(tokens).decode())
                self.assertNotIn("missing original-question relation",bytes(tokens).decode())
            self.assertEqual({name:file_hash(Path(name)) for name in hashes},hashes)

    def test_secondary_control_mismatch_stops_every_review_action(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan,_,_,_ = self.fixture(root)
            result,run,_ = self.run_cli(root,self.finalized(plan),["<answer>wrong</answer>"]*2+[
                "<answer>Fact 1</answer>","<think>changed</think><answer>Fact 1</answer>"])
            self.assertEqual(result,2)
            rows = [json.loads(line) for line in (run/"mechanism_rollouts.jsonl").read_text().splitlines()]
            self.assertTrue(all(set(row["branches"]) <= {"original_a","original_b"} for row in rows))

    def test_check_only_needs_no_assets_and_absent_secondary_is_unmeasured(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan,_,_,_ = self.fixture(root)
            finalized = self.finalized(plan)
            finalized["cases"][1].update(decision="skip",reason="Uncertain prefix sufficiency")
            self.assertEqual(self.run_cli(root,finalized,[],check_only=True)[0],0)
            cases = validate_probe(finalized,plan)
            receipt = metrics([],cases)
            self.assertEqual(receipt["stratum_coverage"]["stable_insufficient"]["planned_cases"],0)
            self.assertFalse(receipt["stratum_coverage"]["stable_insufficient"]["harm_measured"])


if __name__ == "__main__":
    unittest.main()

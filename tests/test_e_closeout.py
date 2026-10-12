"""Synthetic completed E receipts: audit-only checks without model/service execution."""
import json
import tempfile
import unittest
from pathlib import Path

import test_e_probe
from work2.audit_e_probe import audit_run
from work2.common import file_hash


class ECloseoutTests(unittest.TestCase):
    def run_fixture(self, root):
        helper = test_e_probe.EProbeTests(methodName="runTest")
        plan,_,_,_ = helper.fixture(root)
        replies = ["<answer>wrong</answer>"]*2+["<answer>Fact 1</answer>"]*2+[
            "<answer>Fact 1</answer>","<answer>wrong</answer>",
            "<search>third</search>","<search>fourth</search>","<search>fifth refused</search>",
            "<answer>Fact 1</answer>"]
        result,run,_ = helper.run_cli(root,helper.finalized(plan),replies)
        self.assertEqual(result,0)
        return run

    def mutate(self, run, operation):
        path = run/"mechanism_rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        operation(rows)
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))

    def test_recomputed_receipt_refused_fifth_search_and_immutable_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = self.run_fixture(root)
            paths = list(run.glob("*.json*"))+[root/"v1_reviewed.json",root/"a3.jsonl"]
            hashes = {p:file_hash(p) for p in paths}
            receipt = audit_run(run)
            self.assertTrue(receipt["mechanical_checks_passed"])
            self.assertEqual(receipt["primary_paired_outcomes"],{"general_only":1})
            self.assertEqual(receipt["harm_categories"]["stable_insufficient"]["constraint_review"],{"budget_search_refusal":1})
            arm = receipt["cases"][1]["arms"]["constraint_review"]
            self.assertEqual(arm["new_successful_searches"],2)
            self.assertEqual(arm["refused_over_budget_query"],"fifth refused")
            self.assertEqual(len(arm["after_anchor_steps"]),3)
            self.assertTrue(all(case["manual_review"]["decision"]=="pending" for case in receipt["cases"]))
            self.assertEqual({p:file_hash(p) for p in paths},hashes)

    def test_budget_receipt_requires_actual_final_refused_search(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.run_fixture(Path(folder))
            self.mutate(run,lambda rows:rows[1]["branches"]["constraint_review"]["steps"][-1].update(text="<think>no search</think>"))
            with self.assertRaisesRegex(ValueError,"final over-budget refused search"):
                audit_run(run)

    def test_raw_cost_and_metrics_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.run_fixture(Path(folder))
            self.mutate(run,lambda rows:rows[0]["branches"]["general_review"].update(model_input_tokens_after_anchor=0))
            with self.assertRaisesRegex(ValueError,"incremental cost"):
                audit_run(run)
        with tempfile.TemporaryDirectory() as folder:
            run = self.run_fixture(Path(folder))
            path = run/"mechanism_metrics.json"
            metrics = json.loads(path.read_text())
            metrics["completed_cases"] = 900
            path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError,"raw recomputation"):
                audit_run(run)

    def test_control_or_frozen_plan_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.run_fixture(Path(folder))
            self.mutate(run,lambda rows:rows[0]["branches"]["original_b"]["steps"][-1].update(text="<answer>corrupted</answer>"))
            with self.assertRaisesRegex(ValueError,"controls/history not exact"):
                audit_run(run)
        with tempfile.TemporaryDirectory() as folder:
            run = self.run_fixture(Path(folder))
            path = run/"frozen_plan.json"
            path.write_text(path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"Frozen plan hash"):
                audit_run(run)


if __name__ == "__main__":
    unittest.main()

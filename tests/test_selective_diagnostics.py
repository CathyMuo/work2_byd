"""CPU-only post-report diagnostics; no policy changes or real factory evidence."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import test_selective as fixtures
from work2 import run_selective as cli
from work2.common import file_hash
from work2.diagnose_selective import diagnose, selection_details
from work2.selective import ARMS


class DiagnosticTests(unittest.TestCase):
    def test_triggered_correct_denominator_differs_from_population_correct(self):
        cases = []
        for i in range(4):
            base = int(i < 2)
            arms = {arm:{"em":base} for arm in ARMS}
            arms["review"]["em"] = int(i == 2)
            cases.append({"sample_id":str(i),"arms":arms})
        receipt = selection_details(cases,{"0":"review","1":"keep","2":"review","3":"review"})
        self.assertEqual(receipt["triggered"],3)
        self.assertEqual(receipt["triggered_baseline_correct"],1)
        self.assertEqual(receipt["triggered_baseline_wrong"],2)
        self.assertEqual(receipt["harm_given_trigger_and_baseline_correct"],1)
        self.assertEqual(receipt["gains"],1)
        self.assertEqual(receipt["harms"],1)
        self.assertEqual(receipt["EM_unchanged_triggered"],1)
        self.assertIsNone(selection_details(cases,{str(i):"keep" for i in range(4)})["harm_per_trigger"])

    def evaluated(self, root):
        helper = fixtures.SelectiveTests()
        args = helper.collect_fixture(root)
        folder = args.run_dir
        model, decisions, report = (folder/name for name in ("selector.json","decisions.json","validation_report.json"))
        cli.train(argparse.Namespace(run_dir=folder,model=model))
        cli.select(argparse.Namespace(model=model,states=folder/"states.jsonl",decisions=decisions))
        cli.evaluate(argparse.Namespace(run_dir=folder,model=model,decisions=decisions,output=report))
        return folder

    def test_diagnostics_reproduce_predeclared_controls_without_mutating_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.evaluated(Path(tmp))
            paths = list(folder.glob("*.json"))+list(folder.glob("*.jsonl"))
            hashes = {p:file_hash(p) for p in paths}
            receipt = diagnose(folder)
            self.assertTrue(receipt["mechanical_checks_passed"])
            self.assertEqual(receipt["train"]["n"],4)
            self.assertEqual(receipt["validation"]["n"],4)
            self.assertEqual(receipt["train"]["feature_variation"]["spent_call_values"],[1])
            self.assertEqual(receipt["train"]["feature_variation"]["remaining_call_values"],[3])
            self.assertEqual(set(receipt["predeclared_policies"]),{"selector","random_matched_rate","simple_fixed_rule","risk_neutral_ablation"})
            self.assertEqual({p:file_hash(p) for p in paths},hashes)

    def test_predeclared_report_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.evaluated(Path(tmp))
            path = folder/"validation_report.json"
            report = json.loads(path.read_text())
            report["random_matched_rate"]["gain"] = 900
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,"controls disagree"):
                diagnose(folder)


if __name__ == "__main__":
    unittest.main()

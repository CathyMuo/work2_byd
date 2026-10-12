"""Independent CPU regression fixtures; no factory results are fabricated."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_a2
from test_a1 import FakeGenerator
from test_a2 import DistinctRetriever
from work2.audit_pairs import audit_rows, main, paired_interval, summarize_pairs
from work2.audit_recovery import audit_recovery
from work2.common import ROOT, load_json
from work2.replay import replay_branch, replay_intervention, replay_observation


class FactoryAuditTests(unittest.TestCase):
    def setUp(self):
        fixture=test_a2.ReplayTests()
        fixture.setUp()
        self.source,self.agent,self.retrieval=fixture.source,fixture.agent,fixture.retrieval
        self.branches={name:replay_branch(self.source,2,name,FakeGenerator(["<answer>SECRET_GOLD</answer>"]),DistinctRetriever(),self.agent,self.retrieval) for name in ("clean_a","clean_b")}
        self.branches["permuted_a"]=replay_branch(self.source,2,"permuted_a",FakeGenerator(["<answer>wrong</answer>"]),DistinctRetriever(),self.agent,self.retrieval)
        docs=copy.deepcopy(self.source["prefixes"][1]["visible_documents"])
        docs[2]=copy.deepcopy(self.source["prefixes"][1]["candidate_documents"][3])
        self.branches["E2_distractor"]=replay_observation(self.source,2,"E2_distractor",docs,"E2_distractor",{},FakeGenerator(["<answer>SECRET_GOLD</answer>"]),DistinctRetriever(),self.agent,self.retrieval)
        self.case={"sample_id":self.source["sample_id"],"dataset":self.source["dataset"],"target_turn":2,"status":"completed","branches":self.branches}

    def test_factory_api_matches_existing_observation_runner(self):
        docs=copy.deepcopy(self.source["prefixes"][1]["visible_documents"])
        docs[2]=copy.deepcopy(self.source["prefixes"][1]["candidate_documents"][3])
        expected=replay_observation(self.source,2,"factory",docs,"E2_distractor",{"type":"replacement"},FakeGenerator(["<answer>wrong</answer>"]),DistinctRetriever(),self.agent,self.retrieval)
        actual=replay_intervention(self.source,2,"factory",docs,{"type":"replacement"},FakeGenerator(["<answer>wrong</answer>"]),DistinctRetriever(),self.agent,self.retrieval,condition="E2_distractor")
        actual.pop("elapsed_seconds")
        expected.pop("elapsed_seconds")
        self.assertEqual(actual,expected)
        self.assertEqual(docs[0],self.source["prefixes"][1]["visible_documents"][0])

    def test_paired_effects_use_condition_matched_e1_subset(self):
        pairs,events,counts=audit_rows([self.source],[self.case],[2],{})
        summary=summarize_pairs(pairs,resamples=0)
        self.assertEqual(summary["by_condition"]["E1_permutation"]["degradations"],1)
        self.assertEqual(summary["by_condition"]["E2_distractor"]["delta_em"]["estimate"],0)
        self.assertEqual(summary["by_condition"]["E2_distractor"]["excess_over_e1_same_states"]["estimate"],1)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]["review"]["decision"],"pending")
        self.assertNotIn("final_answer",events[0])
        self.assertEqual(counts["completed"],1)

    def test_duplicate_or_missing_planned_states_rejected(self):
        for cases,turns in [([self.case,self.case],[2]),([self.case],[1,2]),([],[2])]:
            with self.assertRaisesRegex(ValueError,"coverage"):
                audit_rows([self.source],cases,turns,{})

    def test_false_unreachable_and_drifted_controls_rejected(self):
        missing={**self.case,"status":"unreachable","branches":{}}
        with self.assertRaisesRegex(ValueError,"Reachable"):
            audit_rows([self.source],[missing],[2],{})
        changed=copy.deepcopy(self.case)
        changed["branches"]["clean_b"]["steps"][-1]["token_ids"].append(999)
        with self.assertRaisesRegex(ValueError,"E0/E0"):
            audit_rows([self.source],[changed],[2],{})

    def test_corrupted_scores_or_pre_target_history_rejected(self):
        changed=copy.deepcopy(self.case)
        changed["branches"]["permuted_a"]["em"]=1
        with self.assertRaisesRegex(ValueError,"scores"):
            audit_rows([self.source],[changed],[2],{})
        changed=copy.deepcopy(self.case)
        changed["branches"]["permuted_a"]["prefixes"][0]["observation_text"]="mutated earlier history"
        with self.assertRaisesRegex(ValueError,"Pre-target"):
            audit_rows([self.source],[changed],[2],{})

    def test_bootstrap_keeps_all_turns_in_question_clusters(self):
        rows=[{"sample_id":sid,"dataset":"hotpotqa","delta_em":delta} for sid,delta in [("a",-1),("b",1)] for _ in (1,2)]
        result=paired_interval(rows,"delta_em",1000,42)
        self.assertEqual(result["states"],4)
        self.assertEqual(result["questions"],2)
        self.assertEqual(result["estimate"],0)
        self.assertEqual(result["ci95"],[-1,1])
        self.assertEqual(paired_interval(rows,"delta_em",1000,42),result)

    def test_sparse_dataset_stratum_does_not_claim_uncertainty_interval(self):
        rows=[{"sample_id":"a","dataset":"hotpotqa","delta_em":-1},{"sample_id":"b","dataset":"2wiki","delta_em":1}]
        self.assertIsNone(paired_interval(rows,"delta_em",100,42)["ci95"])

    def test_recovery_reports_harms_and_does_not_relabel_missing_cases(self):
        pairs=[{"sample_id":"a","dataset":"hotpotqa","target_turn":1,"condition":"E1_permutation","clean_em":1,"perturbed_em":0},
               {"sample_id":"b","dataset":"hotpotqa","target_turn":1,"condition":"E1_permutation","clean_em":1,"perturbed_em":1}]
        rows=[{**r,"baseline":"B3","recovered_em":1-r["perturbed_em"],"source_locator":"fixture only"} for r in pairs]
        result=audit_recovery(pairs,rows,resamples=0)
        b3=result["by_baseline"]["B3"]
        self.assertEqual((b3["gains"],b3["harms"],b3["net_correct_events"]),(1,1,0))
        partial=audit_recovery(pairs,rows+[dict(rows[0],baseline="B1")],resamples=0)
        self.assertFalse(partial["baseline_coverage"]["B1"]["all_selected_events_recorded"])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            audit_recovery(pairs,rows+[rows[0]],resamples=0)

    def test_cli_writes_only_new_artifacts_and_binds_source_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            a1,a3,output=root/"a1.jsonl",root/"a3.jsonl",root/"audit"
            a1.write_text(json.dumps(self.source)+"\n")
            a3.write_text(json.dumps(self.case)+"\n")
            original=(a1.read_bytes(),a3.read_bytes())
            argv=["audit","--a1-rollouts",str(a1),"--a3-rollouts",str(a3),"--target-turns","2","--resamples","0","--output-dir",str(output)]
            with mock.patch("sys.argv",argv):
                self.assertEqual(main(),0)
            self.assertEqual(original,(a1.read_bytes(),a3.read_bytes()))
            self.assertEqual(load_json(output/"sufficiency_review.json")["distinct_states"],1)
            self.assertEqual(len(load_json(output/"paired_events.json")),2)
            self.assertIn("a3_rollouts_sha256",load_json(output/"paired_audit.json")["provenance"])

    def test_schema_probe_needs_no_a1_and_prints_no_question_or_document_text(self):
        import io
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"a3.jsonl"
            path.write_text(json.dumps(self.case)+"\n")
            output=io.StringIO()
            with mock.patch("sys.argv",["audit","--a3-rollouts",str(path),"--schema-only"]),mock.patch("sys.stdout",output):
                self.assertEqual(main(),0)
            value=json.loads(output.getvalue())
            self.assertIn("branches",value["top_level_keys"])
            self.assertEqual(value["branches"]["permuted_a"]["condition"],"E1_permutation")
            self.assertNotIn("SECRET_GOLD",output.getvalue())

    def test_uncommitted_factory_files_stop_alignment_before_branch_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def git(*args):
                return subprocess.check_output(["git",*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            git("init","-b","main")
            git("config","user.name","CPU fixture")
            git("config","user.email","fixture@example.invalid")
            (root/"work2").mkdir()
            (root/"research").mkdir()
            (root/"work2/replay.py").write_text('# fixture')
            (root/"research/STATE.json").write_text('{}')
            git("add",".")
            git("commit","-m","fixture")
            prior=git("rev-parse","HEAD")
            (root/"work2/replay.py").write_text('# uncommitted factory edit')
            result=subprocess.run(["bash",str(ROOT/"scripts/align_public.sh"),"--use-fetched-ref"],cwd=root,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(git("rev-parse","HEAD"),prior)
            self.assertEqual((root/"work2/replay.py").read_text(),'# uncommitted factory edit')

    def test_public_alignment_preserves_factory_commits_and_ignored_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def git(*args):
                return subprocess.check_output(["git",*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            git("init","-b","main")
            git("config","user.name","CPU fixture")
            git("config","user.email","fixture@example.invalid")
            git("remote","add","origin","https://github.com/CathyMuo/work2_byd.git")
            (root/"work2").mkdir()
            (root/"research").mkdir()
            (root/"work2/replay.py").write_text('# fixture')
            (root/"research/STATE.json").write_text('{}')
            (root/".gitignore").write_text('outputs/\nconfigs/\n')
            git("add",".")
            git("commit","-m","public fixture")
            public=git("rev-parse","HEAD")
            git("update-ref","refs/remotes/origin/main",public)
            (root/"factory.txt").write_text('factory committed file')
            git("add","factory.txt")
            git("commit","-m","unpublished factory fixture")
            factory=git("rev-parse","HEAD")
            (root/"outputs").mkdir()
            (root/"outputs/result.json").write_text('preserved')
            result=subprocess.run(["bash",str(ROOT/"scripts/align_public.sh"),"--use-fetched-ref"],cwd=root,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(git("rev-parse","HEAD"),public)
            backups=git("for-each-ref","--format=%(objectname)","refs/heads/factory-backup/").splitlines()
            self.assertEqual(backups,[factory])
            self.assertEqual((root/"outputs/result.json").read_text(),'preserved')


if __name__=="__main__":
    unittest.main()

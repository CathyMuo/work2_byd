"""CPU-only E replay/matching fixtures, never experimental semantic judgments."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_a1 import FakeGenerator
from test_a2 import DistinctRetriever
from work2.audit_pairs import audit_rows
from work2.common import file_hash, load_json, write_json
from work2.metrics import summarize
from work2.post_evidence import PROTOCOL, PROMPTS, continue_from_observation, restore_after_observation
from work2.prepare_e import build_plan, reload_plan, validate_plan
from work2.replay import replay_branch, replay_observation, token_hash
from work2.rollout import run_sample
from work2.run_e import main


class ETests(unittest.TestCase):
    def setUp(self):
        self.agent = {"max_search_turns":4,"max_context_tokens":10000,"max_generation_tokens_per_turn":512,"temperature":0,"seed":42}
        self.retrieval = {"visible_top_k":3,"candidate_top_k":20}
        self.sample = {"sample_id":"hotpotqa:failed","dataset":"hotpotqa","question":"Where failed?",
                       "gold_answer":"Fact 1","gold_supporting_facts":[{"title":"Title1","sentence_id":0,"sentence":"Fact 1"}]}
        self.clean = run_sample(self.sample,FakeGenerator(["<search>first</search>","<search>second</search>","<search>third</search>","<answer>Fact 1</answer>"]),DistinctRetriever(),self.agent,self.retrieval)
        self.perturbed = replay_branch(self.clean,1,"permuted_a",FakeGenerator(["<search>second</search>","<search>third</search>","<answer>wrong</answer>"]),DistinctRetriever(),self.agent,self.retrieval)

    def test_real_perturbed_history_restored_after_later_observation(self):
        original = copy.deepcopy(self.perturbed)
        generator = FakeGenerator(["<answer>wrong</answer>"])
        text,tokens,anchor = restore_after_observation(self.perturbed,self.clean,1,3,self.agent,self.retrieval,generator)
        prefix = self.perturbed["prefixes"][2]
        self.assertEqual(tokens,prefix["prefix_token_ids"]+generator.encode(prefix["observation_text"]))
        self.assertEqual(text,prefix["prefix_text"]+prefix["observation_text"])
        self.assertEqual(anchor["remaining_searches"],1)
        row = continue_from_observation(self.perturbed,self.clean,1,3,"original_a",generator,DistinctRetriever(),self.agent,self.retrieval)
        self.assertEqual(generator.inputs[0],tokens)
        self.assertEqual(row["prefixes"][:3],self.perturbed["prefixes"][:3])
        self.assertEqual(row["condition"],"E1_permutation")
        self.assertEqual(self.perturbed,original)

    def test_budget_not_reset_and_final_answer_allowed_after_fourth_search(self):
        live = DistinctRetriever()
        row = continue_from_observation(self.perturbed,self.clean,1,3,"general_review",FakeGenerator(["<search>fourth</search>","<answer>Fact 1</answer>"]),live,self.agent,self.retrieval)
        self.assertEqual(row["search_calls"],4)
        self.assertEqual(row["search_calls_after_anchor"],1)
        self.assertEqual(row["status"],"answered")
        self.assertEqual(live.queries,["fourth"])
        row = continue_from_observation(self.perturbed,self.clean,1,3,"constraint_review",FakeGenerator(["<search>fourth</search>","<search>refused</search>"]),DistinctRetriever(),self.agent,self.retrieval)
        self.assertEqual(row["status"],"search_budget_exhausted")
        self.assertEqual(row["search_calls"],4)

    def test_fourth_observation_anchor_has_one_generation_and_no_search_credit(self):
        clean = run_sample(self.sample,FakeGenerator(["<search>first</search>","<search>second</search>","<search>third</search>","<search>fourth</search>","<answer>Fact 1</answer>"]),DistinctRetriever(),self.agent,self.retrieval)
        parent = replay_branch(clean,1,"permuted_a",FakeGenerator(["<search>second</search>","<search>third</search>","<search>fourth</search>","<answer>wrong</answer>"]),DistinctRetriever(),self.agent,self.retrieval)
        live = DistinctRetriever()
        row = continue_from_observation(parent,clean,1,4,"constraint_review",FakeGenerator(["<answer>Fact 1</answer>"]),live,self.agent,self.retrieval)
        self.assertEqual(row["status"],"answered")
        self.assertEqual(row["e_metadata"]["anchor"]["remaining_searches"],0)
        self.assertEqual(row["e_metadata"]["maximum_future_generation_segments"],1)
        self.assertEqual(live.queries,[])
        row = continue_from_observation(parent,clean,1,4,"general_review",FakeGenerator(["<search>fifth</search>"]),live,self.agent,self.retrieval)
        self.assertEqual(row["status"],"search_budget_exhausted")
        self.assertEqual(live.queries,[])
        self.assertEqual(row["search_calls_after_anchor"],0)

    def test_relabel_corruption_tokenizer_and_second_intervention_rejected(self):
        for kind in ("relabel","checksum","second"):
            row = copy.deepcopy(self.perturbed)
            if kind=="relabel":
                row["condition"]="E0_clean"
            elif kind=="checksum":
                row["prefixes"][1]["prefix_token_sha256"]="bad"
            else:
                row["prefixes"][1]["visible_documents"].reverse()
            with self.assertRaises(ValueError):
                restore_after_observation(row,self.clean,1,3,self.agent,self.retrieval,FakeGenerator([]))
        generator = FakeGenerator([])
        generator.encode = lambda text:list(text.encode())+[999]
        with self.assertRaises(ValueError):
            restore_after_observation(self.perturbed,self.clean,1,3,self.agent,self.retrieval,generator)

    def test_replacement_conditions_preserve_actual_later_history(self):
        for condition in ("E2_distractor","E3_bridge_removal","E4_redundancy"):
            with self.subTest(condition=condition):
                documents = copy.deepcopy(self.clean["prefixes"][1]["visible_documents"])
                documents[2] = copy.deepcopy(self.clean["prefixes"][1]["candidate_documents"][3])
                parent = replay_observation(self.clean,2,"replacement",documents,condition,{"type":"fixture"},
                    FakeGenerator(["<search>third</search>","<answer>wrong</answer>"]),DistinctRetriever(),self.agent,self.retrieval)
                generator = FakeGenerator(["<answer>wrong</answer>"])
                row = continue_from_observation(parent,self.clean,2,3,"original_a",generator,DistinctRetriever(),self.agent,self.retrieval)
                expected = parent["prefixes"][2]["prefix_token_ids"]+generator.encode(parent["prefixes"][2]["observation_text"])
                self.assertEqual(generator.inputs[0],expected)
                self.assertEqual(row["condition"],condition)
                self.assertEqual(row["prefixes"][1]["visible_documents"],documents)
                corrupt = copy.deepcopy(parent)
                corrupt["prefixes"][1]["visible_documents"][0] = documents[2]
                with self.assertRaisesRegex(ValueError,"exactly one slot"):
                    restore_after_observation(corrupt,self.clean,2,3,self.agent,self.retrieval,generator)

    def test_context_failure_retained_without_compressing_or_resetting_history(self):
        generator = FakeGenerator([])
        _,tokens,_ = restore_after_observation(self.perturbed,self.clean,1,3,self.agent,self.retrieval,generator)
        agent = {**self.agent,"max_context_tokens":len(tokens)+1}
        row = continue_from_observation(self.perturbed,self.clean,1,3,"constraint_review",generator,DistinctRetriever(),agent,self.retrieval)
        self.assertEqual(row["status"],"context_limit")
        self.assertEqual(row["em"],0)
        self.assertEqual(row["search_calls"],3)
        self.assertEqual(row["generated_tokens_after_anchor"],0)
        self.assertEqual(row["prefixes"],self.perturbed["prefixes"][:3])
        self.assertEqual(generator.inputs,[])

    def test_resume_before_parent_intervention_rejected(self):
        row = replay_branch(self.clean,2,"permuted_a",FakeGenerator(["<search>third</search>","<answer>wrong</answer>"]),DistinctRetriever(),self.agent,self.retrieval)
        with self.assertRaisesRegex(ValueError,"at/after"):
            restore_after_observation(row,self.clean,2,1,self.agent,self.retrieval,FakeGenerator([]))

    def test_generated_token_ids_are_not_retokenized(self):
        row = copy.deepcopy(self.perturbed)
        row["steps"][1]["token_ids"]=[999]
        generator = FakeGenerator(["<answer>wrong</answer>"])
        first,second,third = row["prefixes"]
        second["prefix_token_ids"] = first["prefix_token_ids"]+generator.encode(first["observation_text"])+[999]
        second["prefix_token_sha256"] = token_hash(second["prefix_token_ids"])
        second["generated_tokens_before_observation"] = len(row["steps"][0]["token_ids"])+1
        row["steps"][2]["input_tokens"] = len(second["prefix_token_ids"])+len(generator.encode(second["observation_text"]))
        third["prefix_token_ids"] = second["prefix_token_ids"]+generator.encode(second["observation_text"])+row["steps"][2]["token_ids"]
        third["prefix_token_sha256"] = token_hash(third["prefix_token_ids"])
        third["generated_tokens_before_observation"] = second["generated_tokens_before_observation"]+len(row["steps"][2]["token_ids"])
        row["steps"][3]["input_tokens"] = len(third["prefix_token_ids"])+len(generator.encode(third["observation_text"]))
        result = continue_from_observation(row,self.clean,1,3,"original_a",generator,DistinctRetriever(),self.agent,self.retrieval)
        self.assertIn(999,generator.inputs[0])
        self.assertEqual(result["steps"][1]["token_ids"],[999])

    def test_instruction_cost_and_no_offline_review_labels_in_prompts(self):
        generator = FakeGenerator(["<answer>Fact 1</answer>"])
        row = continue_from_observation(self.perturbed,self.clean,1,3,"constraint_review",generator,DistinctRetriever(),self.agent,self.retrieval)
        anchor = self.perturbed["prefixes"][2]
        before = anchor["prefix_token_ids"]+generator.encode(anchor["observation_text"])
        self.assertEqual(generator.inputs[0],before+generator.encode(PROMPTS["constraint_review"]))
        self.assertEqual(row["e_metadata"]["instruction_tokens"],len(generator.encode(PROMPTS["constraint_review"])))
        self.assertEqual(row["model_input_tokens_after_anchor"],len(generator.inputs[0]))
        for prompt in PROMPTS.values():
            self.assertNotIn("gold",prompt)
            self.assertNotIn("failed",prompt)

    def fixture(self,root):
        source = root/"a1"
        source.mkdir()
        samples,clean_rows,cases = [],[],[]
        for name in ("failed","stable"):
            sample = {**self.sample,"sample_id":"hotpotqa:"+name,"question":"Where "+name+"?"}
            clean = run_sample(sample,FakeGenerator(["<search>first</search>","<search>second</search>","<answer>Fact 1</answer>"]),DistinctRetriever(),self.agent,self.retrieval)
            samples.append(sample)
            clean_rows.append(clean)
            for turn in (1,2):
                replies = (["<search>second</search>"] if turn==1 else [])+["<answer>Fact 1</answer>"]
                branches = {n:replay_branch(clean,turn,n,FakeGenerator(replies),DistinctRetriever(),self.agent,self.retrieval) for n in ("clean_a","clean_b")}
                perturb = (["<search>second</search>"] if turn==1 else [])+["<answer>wrong</answer>" if name=="failed" else "<answer>Fact 1</answer>"]
                branches["permuted_a"] = replay_branch(clean,turn,"permuted_a",FakeGenerator(perturb),DistinctRetriever(),self.agent,self.retrieval)
                cases.append({"sample_id":sample["sample_id"],"dataset":"hotpotqa","target_turn":turn,"status":"completed","branches":branches})
        pilot,a3 = root/"pilot.jsonl",root/"a3.jsonl"
        pilot.write_text("".join(json.dumps(r)+"\n" for r in samples))
        (source/"clean_rollouts.jsonl").write_text("".join(json.dumps(r)+"\n" for r in clean_rows))
        a3.write_text("".join(json.dumps(r)+"\n" for r in cases))
        manifest = {"stage":"A1","condition":"E0_clean","status":"completed","working_tree_dirty":False,
            "selected_ids":[r["sample_id"] for r in clean_rows],"samples_path":str(pilot),"samples_sha256":file_hash(pilot),
            "design":{"agent":self.agent,"retrieval":self.retrieval},"repository_commit":"fixture","checkpoint":{},
            "checkpoint_key":"agent_checkpoint","backend":"vllm","tensor_parallel_size":1,"gpu_memory_utilization":0.75}
        write_json(source/"run_manifest.json",manifest)
        write_json(source/"baseline_metrics.json",summarize(clean_rows,clean_rows))
        _,events,_ = audit_rows(clean_rows,cases,[1,2],{})
        packet = {"schema_version":1,"provenance":{"a3_rollouts_sha256":file_hash(a3),"a1_rollouts_sha256":file_hash(source/"clean_rollouts.jsonl"),"input_branch_mapping":{}},"events":events}
        reviewed = copy.deepcopy(packet)
        for event in reviewed["events"]:
            if event["target_turn"]==2:
                event["review"] = self.sufficient()
        packet_path,reviewed_path = root/"packet.json",root/"reviews.json"
        write_json(packet_path,packet)
        write_json(reviewed_path,reviewed)
        plan,context = build_plan(source,a3,packet_path,reviewed_path)
        return plan,context

    def sufficient(self):
        return {"decision":"sufficient","reviewer":"Independent CPU fixture","independent_of_studied_agent":True,
                "earliest_sufficient_turn":1,"supporting_quotes":[{"search_turn":1,"document_id":"1","quote":"Fact 1"}],
                "question_constraints_verified":True,"answer_derivation":"REVIEW_ONLY_SECRET fixture","reason":"CPU fixture, not research evidence"}

    def finalized(self,plan):
        plan = copy.deepcopy(plan)
        for group in plan["groups"]:
            group.update(decision="run",reason="CPU fixture pairing")
            group["failed"]["event"]["review"]["prefix_only_verified"] = True
            group["stable_candidates"][0]["event"]["review"] = {**self.sufficient(),"prefix_only_verified":True}
        return plan

    def test_plan_anchors_after_t2_even_if_sufficient_t1_and_reviews_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            plan,_ = self.fixture(Path(folder))
            failed = plan["groups"][0]["failed"]
            self.assertEqual(failed["event"]["target_turn"],2)
            self.assertEqual(failed["event"]["review"]["earliest_sufficient_turn"],1)
            self.assertEqual(failed["resume_after_turn"],2)
            self.assertTrue(plan["groups"][0]["stable_candidates"])
            with self.assertRaisesRegex(ValueError,"Finalize"):
                validate_plan(plan,plan)
            selected = validate_plan(self.finalized(plan),plan)
            self.assertEqual([c["role"] for c in selected],["failed","stable"])
            corrupt = self.finalized(plan)
            corrupt["protocol"]["prompts"]["general_review"]+="gold oracle"
            with self.assertRaisesRegex(ValueError,"context/protocol"):
                validate_plan(corrupt,plan)

    def test_cli_success_records_hashes_control_first_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan,context = self.fixture(root)
            reviewed = root/"e_reviewed.json"
            write_json(reviewed,self.finalized(plan))
            local = root/"local.json"
            write_json(local,{"retrieval_url":"http://localhost/retrieve"})
            hashes = {str(path):file_hash(path) for path in [root/"a3.jsonl",root/"packet.json",root/"reviews.json",root/"a1/run_manifest.json",root/"a1/clean_rollouts.jsonl"]}
            # Both cases' two originals first, then general/constraint in rotated order.
            replies = ["<answer>wrong</answer>"]*2+["<answer>Fact 1</answer>"]*2+["<answer>Fact 1</answer>"]*4
            generator = FakeGenerator(replies)
            run = root/"run"
            argv = ["e","--plan",str(reviewed),"--local",str(local),"--run-dir",str(run)]
            with mock.patch("sys.argv",argv),mock.patch("work2.run_e.git_output",side_effect=lambda *a:"" if a[0]=="status" else "fixture"), \
                mock.patch("work2.run_e.check_runtime",return_value=(root,{"actual_index_type":"flat"})), \
                mock.patch("work2.run_e.SearchR1Retriever",return_value=DistinctRetriever()), \
                mock.patch("work2.backends.VLLMGenerator",return_value=generator) as backend:
                self.assertEqual(main(),0)
                self.assertEqual(backend.call_args.args[3],0.75)
            self.assertEqual({name:file_hash(Path(name)) for name in hashes},hashes)
            m = load_json(run/"mechanism_metrics.json")
            self.assertTrue(all(m["engineering_gate"].values()))
            self.assertEqual(m["by_role"]["failed"]["general_review"]["gains"],1)
            records = [json.loads(line) for line in (run/"mechanism_rollouts.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["anchor"]["post_observation_token_sha256"] for r in records))
            self.assertEqual(len(generator.inputs),8)
            for tokens in generator.inputs:
                self.assertNotIn("REVIEW_ONLY_SECRET",bytes(tokens).decode())
            self.assertEqual(load_json(run/"run_manifest.json")["status"],"completed")

    def test_stable_control_mismatch_prevents_every_review_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan,_ = self.fixture(root)
            path = root/"e_reviewed.json"
            write_json(path,self.finalized(plan))
            local = root/"local.json"
            write_json(local,{"retrieval_url":"http://localhost/retrieve"})
            run = root/"run"
            generator = FakeGenerator(["<answer>wrong</answer>"]*2+["<answer>Fact 1</answer>","<think>extra</think><answer>Fact 1</answer>"])
            with mock.patch("sys.argv",["e","--plan",str(path),"--local",str(local),"--run-dir",str(run)]), \
                mock.patch("work2.run_e.git_output",side_effect=lambda *a:"" if a[0]=="status" else "fixture"), \
                mock.patch("work2.run_e.check_runtime",return_value=(root,{})), \
                mock.patch("work2.run_e.SearchR1Retriever",return_value=DistinctRetriever()), \
                mock.patch("work2.backends.VLLMGenerator",return_value=generator):
                self.assertEqual(main(),2)
            records = [json.loads(line) for line in (run/"mechanism_rollouts.jsonl").read_text().splitlines()]
            self.assertTrue(all(set(r["branches"]) <= {"original_a","original_b"} for r in records))
            self.assertEqual(load_json(run/"run_manifest.json")["status"],"control_failed")

    def test_failed_review_arm_is_saved_before_infrastructure_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan,_ = self.fixture(root)
            path = root/"e_reviewed.json"
            write_json(path,self.finalized(plan))
            local = root/"local.json"
            write_json(local,{"retrieval_url":"http://localhost/retrieve"})
            run = root/"run"
            generator = FakeGenerator(["<answer>wrong</answer>"]*2+["<answer>Fact 1</answer>"]*2)
            with mock.patch("sys.argv",["e","--plan",str(path),"--local",str(local),"--run-dir",str(run)]), \
                mock.patch("work2.run_e.git_output",side_effect=lambda *a:"" if a[0]=="status" else "fixture"), \
                mock.patch("work2.run_e.check_runtime",return_value=(root,{})), \
                mock.patch("work2.run_e.SearchR1Retriever",return_value=DistinctRetriever()), \
                mock.patch("work2.backends.VLLMGenerator",return_value=generator):
                with self.assertRaises(RuntimeError):
                    main()
            records = [json.loads(line) for line in (run/"mechanism_rollouts.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["branches"]["general_review"]["status"],"infrastructure_error")
            self.assertEqual(records[1]["status"],"controls_passed")
            self.assertEqual(load_json(run/"run_manifest.json")["status"],"failed")


if __name__=="__main__":
    unittest.main()

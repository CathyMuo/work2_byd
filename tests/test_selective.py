"""CPU protocol tests; synthetic fixtures are never offered as experiment backends."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_a1 import FakeGenerator, FakeRetriever
from work2.common import load_json, write_json
from work2.rollout import run_sample
from work2.selective import (ARMS, VERSION, choose, continue_arm, exact, features, fit,
                            observable, prepare, sha, summarize)
from work2 import run_selective as cli


class RoutingGenerator(FakeGenerator):
    def __init__(self, fail_controls=False):
        super().__init__([])
        self.answer_calls=0
        self.fail_controls=fail_controls

    def generate(self,tokens,limit,seed):
        text=bytes(tokens).decode()
        self.inputs.append(list(tokens))
        if '<information>Doc 1(' not in text:
            result='<search>bridge</search>'
        else:
            self.answer_calls+=1
            result='<answer>gold</answer>' if 'briefly review' in text or 'Search budget:' in text else '<answer>wrong</answer>'
            if self.fail_controls and self.answer_calls>8:
                result='<think>changed</think>'+result
        return {'text':result,'token_ids':self.encode(result),'finish_reason':'stop'}


class SelectiveTests(unittest.TestCase):
    def setUp(self):
        self.sample={'sample_id':'hotpotqa:one','dataset':'hotpotqa','split':'train',
                     'question':'Where did it occur?','gold_answer':'SECRET_GOLD','gold_supporting_facts':[]}
        self.agent={'max_search_turns':4,'max_context_tokens':10000,'max_generation_tokens_per_turn':512,'seed':42,'temperature':0}
        self.retrieval={'visible_top_k':3,'candidate_top_k':20}
        self.source=run_sample(self.sample,FakeGenerator(['<search>bridge</search>','<answer>wrong</answer>']),FakeRetriever(),self.agent,self.retrieval)

    def test_original_replay_uses_raw_ids_and_spent_budget(self):
        gen=FakeGenerator(['<search>a</search>','<search>b</search>','<search>c</search>','<search>refused</search>'])
        retr=FakeRetriever();row=continue_arm(self.source,'keep',gen,retr,self.agent,self.retrieval)
        p=self.source['prefixes'][0]
        self.assertEqual(gen.inputs[0],p['prefix_token_ids']+gen.encode(p['observation_text']))
        self.assertEqual(row['search_calls'],4)
        self.assertEqual(row['status'],'search_budget_exhausted')
        self.assertEqual(len(retr.queries),3)
        self.assertEqual(row['si1']['search_calls_after_anchor'],3)

    def test_control_matches_original_full_suffix(self):
        row=continue_arm(self.source,'keep',FakeGenerator(['<answer>wrong</answer>']),FakeRetriever(),self.agent,self.retrieval)
        self.assertTrue(exact(self.source,row))
        row['steps'][1]['token_ids']+=[1]
        self.assertFalse(exact(self.source,row))

    def test_answer_only_search_request_is_failed_without_retrieval(self):
        retr=FakeRetriever()
        row=continue_arm(self.source,'answer',FakeGenerator(['<search>not allowed</search>']),retr,self.agent,self.retrieval)
        self.assertEqual(row['status'],'answer_only_search_refused')
        self.assertEqual(row['search_calls'],1)
        self.assertEqual(row['em'],0)
        self.assertEqual(retr.queries,[])
        self.assertGreater(row['si1']['incremental_instruction_tokens'],0)

    def test_features_ignore_gold_hidden_candidates_and_future(self):
        original=observable(self.source,self.agent,self.retrieval,FakeGenerator([]))
        changed=copy.deepcopy(self.source)
        changed['gold_answer']='LEAK'
        changed['prefixes'][0]['candidate_documents'][10]['contents']='LEAK_FUTURE'
        changed['steps'][1]['text']='<answer>future</answer>'
        changed['final_answer']='future';changed['em']=1
        later=observable(changed,self.agent,self.retrieval,FakeGenerator([]))
        self.assertEqual(original,later)
        self.assertEqual(features(original),features(later))
        self.assertNotIn('SECRET_GOLD',json.dumps(original))
        self.assertNotIn('LEAK_FUTURE',json.dumps(later))

    def test_raw_anchor_corruption_rejected_before_generation(self):
        bad=copy.deepcopy(self.source);bad['prefixes'][0]['prefix_token_sha256']='bad'
        gen=FakeGenerator(['<answer>wrong</answer>'])
        with self.assertRaises(ValueError):continue_arm(bad,'review',gen,FakeRetriever(),self.agent,self.retrieval)
        self.assertEqual(gen.inputs,[])

    def test_nonsearch_questions_keep_all_actions_unchanged(self):
        source=run_sample(self.sample,FakeGenerator(['<answer>SECRET_GOLD</answer>']),FakeRetriever(),self.agent,self.retrieval)
        for arm in ARMS:
            gen=FakeGenerator([])
            row=continue_arm(source,arm,gen,FakeRetriever(),self.agent,self.retrieval)
            self.assertTrue(exact(row,source));self.assertEqual(gen.inputs,[])

    def test_context_limited_anchor_keeps_denominator_without_extra_generation(self):
        retr=FakeRetriever();docs=retr.search('bridge',20)
        for d in docs:d['contents']='Title\n'+'long '*2500
        with mock.patch.object(retr,'search',return_value=docs):
            source=run_sample(self.sample,FakeGenerator(['<search>bridge</search>']),retr,self.agent,self.retrieval)
        self.assertEqual(source['status'],'context_limit')
        state=observable(source,self.agent,self.retrieval,FakeGenerator([]))
        self.assertFalse(state['reachable'])
        self.assertGreater(state['context_tokens'],self.agent['max_context_tokens'])
        for arm in ARMS:
            gen=FakeGenerator([])
            self.assertTrue(exact(source,continue_arm(source,arm,gen,retr,self.agent,self.retrieval)))
            self.assertEqual(gen.inputs,[])

    def fixture(self,root):
        model=root/'checkpoint';model.mkdir();write_json(model/'config.json',{})
        local={'agent_checkpoint':str(model),'retriever_model':str(model),'retrieval_url':'http://local/retrieve'}
        for key in ('corpus_file','index_file'):
            path=root/(key+'.txt');path.write_text('asset');local[key]=str(path)
        for dataset,key in (('hotpotqa','hotpotqa_file'),('2wiki','two_wiki_file')):
            rows=[{'_id':str(i),'question':dataset+' Q '+str(i),'answer':'gold','context':[],'supporting_facts':[]} for i in range(10)]
            path=root/(dataset+'.json');write_json(path,rows);local[key]=str(path)
        old=root/'old.jsonl';old.write_text(json.dumps({'sample_id':'hotpotqa:0','dataset':'hotpotqa','question':'hotpotqa Q 0'})+'\n')
        loc=root/'local.json';write_json(loc,local)
        frozen=root/'frozen';prepare(local,frozen,[old],2,2)
        return loc,frozen

    def collect_fixture(self,root,fail=False,after_smoke=None):
        loc,frozen=self.fixture(root)
        args=argparse.Namespace(local=loc,protocol=Path('configs/selective_v1.json'),samples_dir=frozen,smoke=False,
                                request_timeout=120,check_only=False,run_dir=root/'run',backend='vllm',tensor_parallel_size=1,gpu_memory_utilization=.75,smoke_run_dir=root/'smoke')
        smoke_args=copy.copy(args);smoke_args.smoke=True;smoke_args.run_dir=root/'smoke'
        with mock.patch('work2.backends.VLLMGenerator',return_value=RoutingGenerator()),mock.patch.object(cli,'SearchR1Retriever',return_value=FakeRetriever()),mock.patch.object(cli,'git_output',return_value='fixture'):
            cli.collect(smoke_args)
        if after_smoke:after_smoke()
        with mock.patch('work2.backends.VLLMGenerator',return_value=RoutingGenerator(fail)),mock.patch.object(cli,'SearchR1Retriever',return_value=FakeRetriever()),mock.patch.object(cli,'git_output',return_value='fixture'):
            cli.collect(args)
        return args

    def test_prepare_excludes_id_and_normalized_question_before_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            loc,frozen=self.fixture(Path(tmp));m=load_json(frozen/'selection_manifest.json')
            self.assertNotIn('hotpotqa:0',m['sample_ids'])
            self.assertEqual(sum(v=='train' for v in m['splits'].values()),4)
            self.assertEqual(sum(v=='validation' for v in m['splits'].values()),4)
            with self.assertRaises(ValueError):prepare(load_json(loc),frozen,[Path(tmp)/'old.jsonl'],2,2)
            with self.assertRaises(ValueError):prepare(load_json(loc),Path(tmp)/'other',[],2,2)

    def test_full_cpu_pipeline_train_select_evaluate_and_raw_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.collect_fixture(root)
            cases,m=cli.validated_cases(args.run_dir)
            self.assertEqual(len(cases),8)
            self.assertTrue(m['all_original_controls_exact'])
            model=root/'model.json';decisions=root/'decisions.json';report=root/'evaluation.json'
            cli.train(argparse.Namespace(run_dir=args.run_dir,model=model))
            self.assertEqual(set(load_json(model)['all_train_ids']),{c['sample_id'] for c in cases if c['split']=='train'})
            # select reads only states/model even if cases (containing labels) are absent.
            casefile=args.run_dir/'cases.jsonl';hidden=args.run_dir/'hidden.jsonl';casefile.rename(hidden)
            cli.select(argparse.Namespace(model=model,states=args.run_dir/'states.jsonl',decisions=decisions))
            hidden.rename(casefile)
            cli.evaluate(argparse.Namespace(run_dir=args.run_dir,model=model,decisions=decisions,output=report))
            self.assertEqual(load_json(report)['n'],4)
            self.assertEqual(set(r['sample_id'] for r in load_json(decisions)['decisions']),{c['sample_id'] for c in cases if c['split']=='validation'})
            corrupt=load_json(decisions);corrupt['decisions'][0]['arm']='answer';write_json(decisions,corrupt)
            with self.assertRaises(ValueError):cli.evaluate(argparse.Namespace(run_dir=args.run_dir,model=model,decisions=decisions,output=root/'bad.json'))

    def test_global_control_failure_runs_no_treatment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with mock.patch.object(cli,'continue_arm',wraps=continue_arm) as wrapped:
                with self.assertRaises(ValueError):self.collect_fixture(root,fail=True,after_smoke=wrapped.reset_mock)
                self.assertTrue(all(call.args[1]=='keep' for call in wrapped.call_args_list))
            m=load_json(root/'run/run_manifest.json')
            self.assertEqual(m['status'],'failed')
            self.assertFalse((root/'run/cases.jsonl').exists())

    def test_changed_artifact_and_smoke_cannot_be_fitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.collect_fixture(root)
            m=load_json(args.run_dir/'run_manifest.json');m['smoke']=True;write_json(args.run_dir/'run_manifest.json',m)
            with self.assertRaises(ValueError):cli.validated_cases(args.run_dir)
            self.assertEqual(len(cli.validated_cases(args.run_dir,True)[0]),8)
            with (args.run_dir/'cases.jsonl').open('a') as f:f.write('\n')
            with self.assertRaises(ValueError):cli.validated_cases(args.run_dir,True)

    def test_fit_rejects_validation_and_keep_on_unreachable(self):
        with self.assertRaises(ValueError):fit([{'split':'validation'}])
        state=observable(self.source,self.agent,self.retrieval,FakeGenerator([]))
        arms={a:{'em':0,'search_calls':1} for a in ARMS};arms['review']['em']=1
        case={'split':'train','sample_id':'train:one','state':state,'arms':arms}
        model=fit([case]);state['reachable']=False
        self.assertEqual(choose(model,state)[0],'keep')

    def test_changed_model_cannot_reuse_old_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.collect_fixture(root)
            write_json(root/'checkpoint/config.json',{'changed':True})
            args.run_dir=root/'other_run'
            with self.assertRaisesRegex(ValueError,'identity changed'):cli.collect(args)
            self.assertFalse(args.run_dir.exists())

    def test_net_metric_counts_harm_and_preserves_denominator(self):
        cases=[]
        for i in range(2):
            arms={a:{'em':i,'f1':float(i),'search_calls':1,'generated_tokens':10,'model_input_tokens':100,'status':'answered'} for a in ARMS}
            arms['review']=dict(arms['keep'],em=1-i,f1=float(1-i))
            cases.append({'sample_id':str(i),'state':{'reachable':True},'arms':arms})
        m=summarize(cases)
        self.assertEqual(m['arms']['review']['gain'],1)
        self.assertEqual(m['arms']['review']['harm'],1)
        self.assertEqual(m['arms']['review']['net_em'],0)
        self.assertEqual(m['finite_action_oracle_em'],1)


if __name__=='__main__':unittest.main()

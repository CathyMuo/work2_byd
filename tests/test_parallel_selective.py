"""CPU-only orchestration tests; no real model/GPU backend is exposed."""
import argparse
import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_a1 import FakeRetriever
from test_selective import RoutingGenerator
import test_selective as fixtures
from work2.common import load_json, write_json
from work2.replay import AuditedRetriever
from work2 import parallel_selective as dp
from work2.run_selective import validated_cases


class ParallelTests(unittest.TestCase):
    def frozen(self, root):
        # Reuse data-only fixture, not the serial inference driver.
        fixture = fixtures.SelectiveTests()
        loc, samples = fixture.fixture(root)
        args = argparse.Namespace(local=loc, protocol=Path("configs/selective_v1.json"), samples_dir=samples,
                                  run_dir=root/"dp4", smoke=True, smoke_run_dir=None, gpus="0,1,2,3",
                                  backend="vllm", gpu_memory_utilization=.75, request_timeout=120, check_only=False)
        with mock.patch.object(dp,"git_output",return_value="fixture"):
            plan=dp.freeze_plan(args)
        args.run_dir.mkdir();(args.run_dir/"logs").mkdir()
        path=args.run_dir/"parallel_plan.json";write_json(path,plan)
        return path,plan,args

    def worker(self,path,index,phase,fail=False):
        factory=lambda plan,local:(RoutingGenerator(fail),AuditedRetriever(FakeRetriever(),3))
        with mock.patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":str(index)}),mock.patch.object(dp,"git_output",return_value="fixture"),mock.patch.object(dp,"runtime",side_effect=factory):
            (dp.worker_controls if phase=="controls" else dp.worker_actions)(path,index)

    def test_exactly_four_distinct_devices_required(self):
        self.assertEqual(dp.devices("0,1,2,3"),["0","1","2","3"])
        for text in ("0", "0,1", "0,1,2,2", "a,b,c,d"):
            with self.assertRaises(ValueError):dp.devices(text)

    def test_preflight_rejects_insufficient_physical_gpus(self):
        with mock.patch.object(dp.subprocess,'check_output',return_value='0\n1\n'):
            with self.assertRaises(ValueError):dp.check_gpus(['0','1','2','3'])
        with mock.patch.object(dp.subprocess,'check_output',return_value='0\n1\n2\n3\n'):
            dp.check_gpus(['0','1','2','3'])

    def test_four_card_smoke_merge_passes_serial_raw_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,args=self.frozen(Path(tmp))
            for i in range(4):self.worker(path,i,"controls")
            self.assertFalse(any((Path(w["run_dir"])/"cases.jsonl").exists() for w in plan["workers"]))
            dp.publish_barrier(path)
            for i in range(4):self.worker(path,i,"actions")
            merged=dp.merge(path)
            cases,m=validated_cases(args.run_dir,True)
            self.assertEqual(len(cases),4)
            self.assertEqual([c["sample_id"] for c in cases],plan["sample_ids"])
            self.assertEqual(m["execution"],dp.DP_VERSION)
            self.assertEqual(m["gpu_devices"],["0","1","2","3"])
            self.assertTrue(load_json(args.run_dir/"audit_receipt.json")["mechanical_checks_passed"])
            self.assertEqual(len(merged["worker_manifest_sha256"]),4)

    def test_four_card_development_and_selector_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,args=self.frozen(Path(tmp))
            for i in range(4):self.worker(path,i,"controls")
            dp.publish_barrier(path)
            for i in range(4):self.worker(path,i,"actions")
            dp.merge(path)
            args.smoke=False;args.smoke_run_dir=args.run_dir;args.run_dir=Path(tmp)/"development"
            with mock.patch.object(dp,"git_output",return_value="fixture"):development=dp.freeze_plan(args)
            args.run_dir.mkdir();path=args.run_dir/"parallel_plan.json";write_json(path,development)
            for i in range(4):self.worker(path,i,"controls")
            dp.publish_barrier(path)
            for i in range(4):self.worker(path,i,"actions")
            dp.merge(path)
            cases,_=validated_cases(args.run_dir)
            self.assertEqual(len(cases),8)
            self.assertEqual(sum(c['split']=='train' for c in cases),4)
            self.assertEqual(sum(c['split']=='validation' for c in cases),4)
            from work2 import run_selective as cli
            model=args.run_dir/'selector.json';decisions=args.run_dir/'decisions.json';report=args.run_dir/'evaluation.json'
            cli.train(argparse.Namespace(run_dir=args.run_dir,model=model))
            cli.select(argparse.Namespace(model=model,states=args.run_dir/'states.jsonl',decisions=decisions))
            cli.evaluate(argparse.Namespace(run_dir=args.run_dir,model=model,decisions=decisions,output=report))
            self.assertEqual(load_json(report)['n'],4)

    def test_actions_require_all_workers_controls_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,_=self.frozen(Path(tmp))
            self.worker(path,0,"controls")
            with self.assertRaises(FileNotFoundError):dp.publish_barrier(path)
            with mock.patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}),mock.patch.object(dp,"git_output",return_value="fixture"),mock.patch.object(dp,"runtime") as model:
                with self.assertRaises(FileNotFoundError):dp.worker_actions(path,0)
                model.assert_not_called()
            self.assertFalse((Path(plan["workers"][0]["run_dir"])/"cases.jsonl").exists())

    def test_failure_in_one_worker_blocks_global_treatments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,_=self.frozen(Path(tmp))
            for i in range(3):self.worker(path,i,"controls")
            # Force a different suffix on the first replay; same final answer is insufficient.
            with mock.patch.object(dp,"exact",return_value=False):
                with self.assertRaises(ValueError):self.worker(path,3,"controls")
            with self.assertRaises(FileNotFoundError):dp.publish_barrier(path)
            self.assertFalse(any((Path(w["run_dir"])/"cases.jsonl").exists() for w in plan["workers"]))

    def test_immutable_controls_cannot_change_after_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,_=self.frozen(Path(tmp))
            for i in range(4):self.worker(path,i,"controls")
            dp.publish_barrier(path)
            with (Path(plan["workers"][2]["run_dir"])/"controls.jsonl").open("a") as f:f.write("\n")
            with self.assertRaises(ValueError):dp.verify_barrier(path)

    def test_cross_worker_same_query_drift_rejected(self):
        audit=lambda signature:{"drift":0,"calls":[{"query":"q","visible_sha256":signature,"matched_reference":True}]}
        with self.assertRaises(ValueError):dp.query_signatures([audit("a"),audit("b")])

    def test_gpu_binding_and_code_change_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,_=self.frozen(Path(tmp))
            with mock.patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"3"}),mock.patch.object(dp,"git_output",return_value="fixture"):
                with self.assertRaises(ValueError):dp.worker_context(path,0)
            with mock.patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}),mock.patch.object(dp,"git_output",return_value="new_commit"):
                with self.assertRaises(ValueError):dp.worker_context(path,0)

    def test_all_four_workers_are_started_before_polling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,_=self.frozen(Path(tmp));started=[]
            class Done:
                def poll(self):
                    if len(started)!=4:raise AssertionError("Waited before dispatching all GPUs")
                    return 0
            def spawn(argv,**kwargs):
                started.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
                self.assertTrue(kwargs["start_new_session"])
                return Done()
            with mock.patch.object(dp.subprocess,"Popen",side_effect=spawn):dp.run_phase(path,"controls")
            self.assertEqual(started,["0","1","2","3"])

    def test_nonzero_exit_stops_others_and_prevents_partial_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,_=self.frozen(Path(tmp));procs=[]
            class Process:
                def __init__(self,index):self.index=index
                def poll(self):return 1 if self.index==1 else None
            def spawn(*args,**kwargs):
                p=Process(len(procs));procs.append(p);return p
            with mock.patch.object(dp.subprocess,"Popen",side_effect=spawn),mock.patch.object(dp,"stop_worker") as stop:
                with self.assertRaises(RuntimeError):dp.run_phase(path,"controls")
                self.assertEqual(stop.call_count,4)

    def test_missing_worker_cannot_be_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_,_=self.frozen(Path(tmp))
            for i in range(4):self.worker(path,i,"controls")
            dp.publish_barrier(path)
            for i in range(3):self.worker(path,i,"actions")
            with self.assertRaises(ValueError):dp.merge(path)

    def test_extra_shard_rows_are_not_silently_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,plan,_=self.frozen(Path(tmp))
            for i in range(4):self.worker(path,i,"controls")
            shard=Path(plan["workers"][1]["run_dir"])/"states.jsonl"
            with shard.open("a") as f:f.write(shard.read_text())
            with self.assertRaises(ValueError):dp.merge_ordered(plan,"states.jsonl")


if __name__=="__main__":unittest.main()

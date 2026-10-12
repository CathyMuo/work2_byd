"""Factory SI1 CLI: freeze samples, collect controlled actions, fit, select, evaluate."""
import argparse
import copy
import datetime
import importlib.metadata
import json
import os
import time
from pathlib import Path

from .common import file_hash, git_output, load_json, local_path, write_json
from .datasets import read_rows
from .metrics import answer_scores
from .replay import AuditedRetriever
from .retrieval import SearchR1Retriever
from .rollout import parse_action, run_sample
from .run_baseline import asset_identity, validate_design
from .selective import (ARMS, PROMPTS, VERSION, arm_summary, choose, continue_arm, exact,
                        features, fit, jsonl, load_samples, observable, question_key, sha, summarize)


def load_protocol(path):
    p=load_json(path)
    validate_design(p)
    if p.get("protocol")!=VERSION or p.get("target_turn")!=1 or p.get("arms")!=list(ARMS):
        raise ValueError("Unsupported SI1 protocol/action/anchor")
    if p["agent"]["max_search_turns"]!=4 or p["agent"]["temperature"]!=0:
        raise ValueError("SI1 v1 freezes four searches and greedy decoding")
    s=p["selector"]
    if s["ridge"]<=0 or s["risk_weight"]<1 or s["margin"]<0 or s["call_penalty"]<0:
        raise ValueError("Invalid frozen selector settings")
    return p


def new_file(path,value):
    p=Path(path)
    if p.exists():raise ValueError("Refuse overwriting artifact: "+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def collect(args):
    local=load_json(args.local);protocol=load_protocol(args.protocol)
    samples,selection=load_samples(args.samples_dir,args.smoke)
    checkpoint=local_path(local["agent_checkpoint"])
    if not (checkpoint/"config.json").is_file():raise ValueError("Missing local checkpoint config.json")
    assets={k:asset_identity(local_path(local[k]),k in ("agent_checkpoint","retriever_model"))
            for k in ("agent_checkpoint","retriever_model","corpus_file","index_file")}
    smoke_sources=[]
    if not args.smoke and not args.check_only:
        previous=getattr(args,"smoke_run_dir",None)
        if previous is None:raise ValueError("Full development needs --smoke-run-dir with completed same-cohort smoke")
        _,sm=validated_cases(previous,allow_smoke=True)
        if not sm["smoke"] or sm["protocol_sha256"]!=sha(protocol) or sm["selection_manifest_sha256"]!=file_hash(Path(args.samples_dir)/"selection_manifest.json"):
            raise ValueError("Smoke belongs to different samples/protocol")
        if sm["assets"]!=assets or sm["backend"]!=args.backend or sm["tensor_parallel_size"]!=args.tensor_parallel_size or sm["gpu_memory_utilization"]!=args.gpu_memory_utilization or sm["retrieval_url"]!=local["retrieval_url"]:
            raise ValueError("Smoke model/resource/backend/service identity changed")
        smoke_sources=read_rows(Path(previous)/"sources.jsonl")
    retriever=SearchR1Retriever(local["retrieval_url"],args.request_timeout)
    probe=retriever.search(samples[0]["question"],protocol["retrieval"]["candidate_top_k"])
    if len(probe)<protocol["retrieval"]["visible_top_k"]:raise ValueError("Insufficient retrieval results")
    print("Selected {} questions; retrieval check passed".format(len(samples)),flush=True)
    if args.check_only:
        print("No model loaded, no experiment executed.");return
    folder=Path(args.run_dir)
    if folder.exists():raise ValueError("Run directory exists; preserve old outputs")
    folder.mkdir(parents=True)
    versions={}
    for name in ("torch","transformers","vllm"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    manifest={"version":VERSION,"protocol":protocol,"protocol_sha256":sha(protocol),
              "prompts_sha256":sha(PROMPTS),"selection_manifest_sha256":file_hash(Path(args.samples_dir)/"selection_manifest.json"),
              "sample_file_sha256":file_hash(Path(args.samples_dir)/("smoke.jsonl" if args.smoke else "samples.jsonl")),
              "sample_ids":[s["sample_id"] for s in samples],"smoke":args.smoke,
              "repository_commit":git_output("rev-parse","HEAD"),"working_tree_dirty":bool(git_output("status","--porcelain")),
              "submodules":git_output("submodule","status","--recursive"),
              "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "packages":versions,"backend":args.backend,"tensor_parallel_size":args.tensor_parallel_size,
              "gpu_memory_utilization":args.gpu_memory_utilization,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "assets":assets,
              "declared_asset_versions":{k:local.get(k) for k in ("checkpoint_version","corpus_version","index_version","actual_index_type")},
              "retrieval_url":local["retrieval_url"],"service_identity_note":"Configured local assets do not prove the service actually loaded them; inherit and verify existing factory service.",
              "status":"initializing","expected_questions":len(samples),"recorded_questions":0,
              "historical_exclusions":selection["exclusions"],"research_GO":False,"RL_GO":False}
    if smoke_sources:
        manifest["smoke_run_manifest_sha256"]=file_hash(Path(args.smoke_run_dir)/"run_manifest.json")
    write_json(folder/"run_manifest.json",manifest)
    audit=AuditedRetriever(retriever,protocol["retrieval"]["visible_top_k"])
    for row in smoke_sources:
        for prefix in row["prefixes"]:audit.seed(prefix["search_query"],prefix["visible_documents"])
    completed=[];agent=protocol["agent"];retrieval=protocol["retrieval"]
    try:
        os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
        from .backends import HFGenerator,VLLMGenerator
        if args.backend=="hf" and args.tensor_parallel_size!=1:raise ValueError("HF requires TP=1")
        generator=VLLMGenerator(checkpoint,agent,args.tensor_parallel_size,args.gpu_memory_utilization) if args.backend=="vllm" else HFGenerator(checkpoint)
        sources=[]
        manifest["status"]="baseline";write_json(folder/"run_manifest.json",manifest)
        with (folder/"sources.jsonl").open("x",encoding="utf-8") as f:
            for i,sample in enumerate(samples):
                row=run_sample(sample,generator,audit,agent,retrieval)
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");f.flush()
                if row["status"]=="infrastructure_error":raise RuntimeError(str(row.get("error")))
                sources.append(row)
                print("baseline {}/{} {} {}".format(i+1,len(samples),sample["sample_id"],row["status"]),flush=True)
        states=[dict(observable(row,agent,retrieval,generator),split=row["split"]) for row in sources]
        jsonl(folder/"states.jsonl",states)  # observable states contain NO gold or suffix
        controls=[]
        manifest["status"]="controls";write_json(folder/"run_manifest.json",manifest)
        with (folder/"controls.jsonl").open("x",encoding="utf-8") as f:
            for i,row in enumerate(sources):
                if states[i]["reachable"]:
                    a=continue_arm(row,"keep",generator,audit,agent,retrieval)
                    b=continue_arm(row,"keep",generator,audit,agent,retrieval)
                else:
                    a=run_sample(samples[i],generator,audit,agent,retrieval)
                    b=run_sample(samples[i],generator,audit,agent,retrieval)
                ok=exact(row,a) and exact(row,b) and exact(a,b)
                f.write(json.dumps({"sample_id":row["sample_id"],"exact":ok,"keep_a":a,"keep_b":b},ensure_ascii=False,allow_nan=False)+"\n");f.flush()
                if not ok:raise ValueError("Original controls/history mismatch: "+row["sample_id"])
                controls.append(a)
                print("controls {}/{} exact".format(i+1,len(samples)),flush=True)
        # Global control barrier: no treatment is run until EVERY selected original passed.
        manifest.update(status="actions",all_original_controls_exact=True)
        write_json(folder/"run_manifest.json",manifest)
        with (folder/"cases.jsonl").open("x",encoding="utf-8") as f:
            for i,(row,state) in enumerate(zip(sources,states)):
                arms={"keep":controls[i]}
                for arm in ARMS[1:]:
                    arms[arm]=continue_arm(row,arm,generator,audit,agent,retrieval)
                if not state["reachable"]:
                    # All policies leave no-search questions unchanged, preserving the denominator.
                    arms={arm:continue_arm(row,arm,generator,audit,agent,retrieval) for arm in ARMS}
                case={"version":VERSION,"sample_id":row["sample_id"],"dataset":row["dataset"],
                      "split":row["split"],"state":state,"state_sha256":sha(state),"arms":arms}
                f.write(json.dumps(case,ensure_ascii=False,allow_nan=False)+"\n");f.flush()
                completed.append(case);manifest["recorded_questions"]=len(completed)
                write_json(folder/"metrics.json",summarize(completed))
                write_json(folder/"run_manifest.json",manifest)
                print("actions {}/{} {} {}".format(i+1,len(samples),row["sample_id"],{a:r["em"] for a,r in arms.items()}),flush=True)
        manifest.update(status="completed",all_planned_questions_completed=True,
                        artifact_sha256={n:file_hash(folder/n) for n in ("sources.jsonl","states.jsonl","controls.jsonl","cases.jsonl")})
    except BaseException as exc:
        manifest.update(status="failed",error={"type":type(exc).__name__,"message":str(exc)})
        raise
    finally:
        write_json(folder/"retrieval_audit.json",{"calls":audit.calls,"drift":sum(not c["matched_reference"] for c in audit.calls)})
        if manifest["status"]=="completed":
            manifest["artifact_sha256"]["retrieval_audit.json"]=file_hash(folder/"retrieval_audit.json")
        write_json(folder/"run_manifest.json",manifest)
    print("SI1 collection complete; no scientific/RL GO inferred.",flush=True)


def validated_cases(folder, allow_smoke=False):
    folder=Path(folder);m=load_json(folder/"run_manifest.json")
    if m["version"]!=VERSION or m["status"]!="completed" or not m.get("all_original_controls_exact"):
        raise ValueError("Need completed SI1 run with exact global controls")
    if m["smoke"] and not allow_smoke:raise ValueError("Smoke is engineering only; fit/evaluate requires full development run")
    if sha(m["protocol"])!=m["protocol_sha256"] or sha(PROMPTS)!=m["prompts_sha256"]:
        raise ValueError("Frozen protocol/prompts mismatch")
    if m.get("execution") == "SI1_four_worker_global_barrier_v1":
        from .parallel_selective import verify_merged_provenance
        verify_merged_provenance(folder, m)
    for name,h in m["artifact_sha256"].items():
        if file_hash(folder/name)!=h:raise ValueError("Artifact changed: "+name)
    rows=read_rows(folder/"cases.jsonl");states=read_rows(folder/"states.jsonl")
    sources=read_rows(folder/"sources.jsonl");controls=read_rows(folder/"controls.jsonl")
    if any([r["sample_id"] for r in group]!=m["sample_ids"] for group in (rows,states,sources,controls)):
        raise ValueError("Actual question coverage/order mismatch")
    if len(rows)!=m["expected_questions"] or len(rows)!=m["recorded_questions"]:raise ValueError("Incomplete actual coverage")
    seen=set()
    for c,state,source,ctrl in zip(rows,states,sources,controls):
        if c["sample_id"] in seen:raise ValueError("Duplicate question")
        seen.add(c["sample_id"])
        if c["state"]!=state or sha(state)!=c["state_sha256"] or c["split"]!=source["split"] or state["split"]!=c["split"]:
            raise ValueError("State or split changed")
        if set(c["arms"])!=set(ARMS) or not ctrl["exact"] or not all(exact(source,ctrl[k]) for k in ("keep_a","keep_b")):
            raise ValueError("Control receipt failed")
        if not exact(source,c["arms"]["keep"]):raise ValueError("Keep/source mismatch")
        # Independently reconstruct the allowlisted state from the source; tokenization is runtime-verified.
        expected=observable(source,m["protocol"]["agent"],m["protocol"]["retrieval"])
        for key in expected:
            if key!="context_tokens" and state[key]!=expected[key]:raise ValueError("Observable state not bound to source: "+key)
        expected_length=(source["steps"][1]["input_tokens"] if state["reachable"] else
                         (source["final_prompt_token_count"] if source["prefixes"] else 0))
        if state["context_tokens"]!=expected_length:
            raise ValueError("Observable context length differs from actual anchor")
        for arm,row in c["arms"].items():
            if row["sample_id"]!=c["sample_id"] or row["status"]=="infrastructure_error":raise ValueError("Invalid arm identity/status")
            score=answer_scores(row["final_answer"],source["gold_answer"]) if row["status"]=="answered" else {"em":0,"f1":0.0}
            if row["em"]!=score["em"] or row["f1"]!=score["f1"]:raise ValueError("Raw score mismatch")
            if row["generated_tokens"]!=sum(len(s["token_ids"]) for s in row["steps"]) or row["model_input_tokens"]!=sum(s["input_tokens"] for s in row["steps"]):raise ValueError("Raw token cost mismatch")
            if row["search_calls"]!=len(row["prefixes"]) or row["search_calls"]>m["protocol"]["agent"]["max_search_turns"]:raise ValueError("Global search cap/count mismatch")
            if state["reachable"]:
                if row["steps"][:1]!=source["steps"][:1] or row["prefixes"][:1]!=source["prefixes"][:1]:raise ValueError("Anchor history mutated")
                meta=row["si1"]
                prompt=PROMPTS[arm].format(used=1,remaining=m["protocol"]["agent"]["max_search_turns"]-1,total=m["protocol"]["agent"]["max_search_turns"])
                if meta["arm"]!=arm or meta["instruction"]!=prompt or meta["spent_at_anchor"]!=1:
                    raise ValueError("Arm prompt/budget identity mismatch")
                if meta["post_observation_token_sha256"]!=c["arms"]["keep"]["si1"]["post_observation_token_sha256"]:
                    raise ValueError("Different action anchor")
                if len(row["steps"])>1 and row["steps"][1]["input_tokens"]-state["context_tokens"]!=meta["incremental_instruction_tokens"]:
                    raise ValueError("Instruction cost mismatch")
                for field,raw in (("search_calls_after_anchor",row["search_calls"]-1),
                                  ("generated_tokens_after_anchor",row["generated_tokens"]-len(source["steps"][0]["token_ids"])),
                                  ("model_input_tokens_after_anchor",row["model_input_tokens"]-source["steps"][0]["input_tokens"])):
                    if meta[field]!=raw:raise ValueError("Incremental cost mismatch")
                if arm=="answer" and row["search_calls"]!=1:raise ValueError("Answer-only executed retrieval")
            if row["status"] in ("search_budget_exhausted","answer_only_search_refused") and parse_action(row["steps"][-1]["text"])[0]!="search":
                raise ValueError("Refused-search status lacks actual final search request")
    audit=load_json(folder/"retrieval_audit.json")
    if audit["drift"]!=0 or any(not c["matched_reference"] for c in audit["calls"]):raise ValueError("Retrieval drift")
    expected_calls=sum(s["search_calls"] for s in sources)
    expected_calls+=sum(ctrl[k]["search_calls"]-(1 if state["reachable"] else 0)
                        for ctrl,state in zip(controls,states) for k in ("keep_a","keep_b"))
    expected_calls+=sum(c["arms"][a]["si1"]["search_calls_after_anchor"] for c in rows for a in ARMS[1:])
    if len(audit["calls"])!=expected_calls:raise ValueError("Actual retrieval execution count mismatch")
    return rows,m


def train(args):
    cases,m=validated_cases(args.run_dir)
    if args.model.exists():raise ValueError("Model exists; preserve frozen selector")
    training=[c for c in cases if c["split"]=="train"]
    validation=[c for c in cases if c["split"]=="validation"]
    if not validation:raise ValueError("No held-out validation questions")
    ids={c["sample_id"] for c in training};q={question_key(c["state"]["question"]) for c in training}
    if any(c["sample_id"] in ids or question_key(c["state"]["question"]) in q for c in validation):raise ValueError("Train/validation overlap")
    model=fit(training,m["protocol"]["selector"]["ridge"])
    train_stats=summarize(training)
    model.update(selector=m["protocol"]["selector"],run_manifest_sha256=file_hash(args.run_dir/"run_manifest.json"),
                 training_cases_sha256=sha(training),all_train_ids=sorted(ids),
                 all_train_question_keys=sorted(q),
                 frozen_best_fixed_arm=train_stats["best_fixed_em_arm"],protocol_sha256=m["protocol_sha256"])
    new_file(args.model,model)
    print("Fitted on {} training states; validation outcomes did not fit the model.".format(model["trained_rows"]))


def select(args):
    """This command reads only observable states and a frozen model, never cases/gold."""
    model=load_json(args.model);states=read_rows(args.states)
    rows=[s for s in states if s["split"]=="validation"]
    if not rows:raise ValueError("No validation states")
    settings=model["selector"]
    seen=set();out=[]
    for s in rows:
        q=question_key(s["question"])
        if s["sample_id"] in model["all_train_ids"] or q in model["all_train_question_keys"] or s["sample_id"] in seen:raise ValueError("Seen/duplicate validation question")
        seen.add(s["sample_id"]);started=time.monotonic()
        arm,scores=choose(model,s,settings["risk_weight"],settings["margin"],settings["call_penalty"])
        # Ablation is predeclared and shares the same trained heads/visible state.
        neutral,_=choose(model,s,1.0,settings["margin"],settings["call_penalty"])
        out.append({"sample_id":s["sample_id"],"state_sha256":sha(s),"arm":arm,"scores":scores,
                    "risk_neutral_arm":neutral,"decision_cpu_seconds":time.monotonic()-started})
    # Random assignment permutes the learned actions among reachable states, matching rates exactly.
    import random
    actions=[r["arm"] for r,s in zip(out,rows) if s["reachable"]]
    random.Random(20261012).shuffle(actions);actions=iter(actions)
    for r,s in zip(out,rows):
        r["random_matched_arm"]=next(actions) if s["reachable"] else "keep"
        r["simple_rule_arm"]=("budget" if s["remaining_calls"]<=1 else
                              ("review" if features(s)[10]<.25 else "keep")) if s["reachable"] else "keep"
    new_file(args.decisions,{"version":VERSION,"model_sha256":file_hash(args.model),
                            "states_sha256":file_hash(args.states),"frozen_best_fixed_arm":model["frozen_best_fixed_arm"],
                            "decisions":out,"note":"Selected from prefix-only states BEFORE reading any validation arm outcomes."})
    print("Saved {} prefix-only decisions.".format(len(out)))


def evaluate(args):
    cases,m=validated_cases(args.run_dir)
    validation=[c for c in cases if c["split"]=="validation"]
    decision=load_json(args.decisions)
    model=load_json(args.model)
    if decision["model_sha256"]!=file_hash(args.model) or model["run_manifest_sha256"]!=file_hash(args.run_dir/"run_manifest.json"):
        raise ValueError("Model/run/decision provenance mismatch")
    if decision["version"]!=VERSION or decision["states_sha256"]!=file_hash(args.run_dir/"states.jsonl"):raise ValueError("Decision/state provenance mismatch")
    rows=decision["decisions"]
    if len({r["sample_id"] for r in rows})!=len(rows):raise ValueError("Duplicate decision")
    by_id={r["sample_id"]:r for r in rows}
    if set(by_id)!={c["sample_id"] for c in validation}:raise ValueError("Decision coverage mismatch")
    if any(by_id[c["sample_id"]]["state_sha256"]!=c["state_sha256"] for c in validation):raise ValueError("Decision anchor changed")
    settings=model["selector"]
    for c in validation:
        predicted,scores=choose(model,c["state"],settings["risk_weight"],settings["margin"],settings["call_penalty"])
        if by_id[c["sample_id"]]["arm"]!=predicted or by_id[c["sample_id"]]["scores"]!=scores:
            raise ValueError("Decision is not reproducible from frozen prefix-only model")
    result=summarize(validation,{r["sample_id"]:r["arm"] for r in rows})
    fixed=decision["frozen_best_fixed_arm"]
    selected=[c["arms"][by_id[c["sample_id"]]["arm"]] for c in validation]
    result["selector"]["net_em_vs_train_frozen_fixed"]=sum(r["em"]-c["arms"][fixed]["em"] for r,c in zip(selected,validation))/len(validation)
    result["frozen_fixed_arm"]=fixed
    result["risk_neutral_ablation"]=summarize(validation,{r["sample_id"]:r["risk_neutral_arm"] for r in rows})["selector"]
    result["random_matched_rate"]=summarize(validation,{r["sample_id"]:r["random_matched_arm"] for r in rows})["selector"]
    result["simple_fixed_rule"]=summarize(validation,{r["sample_id"]:r["simple_rule_arm"] for r in rows})["selector"]
    result["by_dataset"]={d:summarize([c for c in validation if c["dataset"]==d],
                                   {c["sample_id"]:by_id[c["sample_id"]]["arm"] for c in validation if c["dataset"]==d})
                          for d in sorted({c["dataset"] for c in validation})}
    result["decision_cpu_seconds"]=sum(r["decision_cpu_seconds"] for r in rows)
    result["all_development_diagnostic"]=summarize(cases)
    result["provenance"]={"run_manifest_sha256":file_hash(args.run_dir/"run_manifest.json"),"decision_sha256":file_hash(args.decisions),"model_sha256":decision["model_sha256"]}
    result["decision"]="Exploratory validation only. No automated RL/confirmation GO; judge quality, harm, costs and strong-baseline gap together."
    new_file(args.output,result)
    print(json.dumps({"validation_n":result["n"],"selector":result["selector"],"oracle_headroom":result["oracle_headroom_vs_best_fixed_em"]},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare");q.add_argument("--local",type=Path,default=Path("configs/local.json"));q.add_argument("--output",type=Path,required=True)
    q.add_argument("--exclude",type=Path,nargs="+",required=True);q.add_argument("--train-per-dataset",type=int,default=600);q.add_argument("--validation-per-dataset",type=int,default=400)
    q=sub.add_parser("collect");q.add_argument("--local",type=Path,default=Path("configs/local.json"));q.add_argument("--protocol",type=Path,default=Path("configs/selective_v1.json"))
    q.add_argument("--samples-dir",type=Path,required=True);q.add_argument("--run-dir",type=Path,required=True);q.add_argument("--smoke",action="store_true");q.add_argument("--smoke-run-dir",type=Path);q.add_argument("--check-only",action="store_true")
    q.add_argument("--backend",choices=("vllm","hf"),default="vllm");q.add_argument("--tensor-parallel-size",type=int,default=1);q.add_argument("--gpu-memory-utilization",type=float,default=.75);q.add_argument("--request-timeout",type=float,default=120)
    q=sub.add_parser("audit");q.add_argument("--run-dir",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    q=sub.add_parser("fit");q.add_argument("--run-dir",type=Path,required=True);q.add_argument("--model",type=Path,required=True)
    q=sub.add_parser("select");q.add_argument("--model",type=Path,required=True);q.add_argument("--states",type=Path,required=True);q.add_argument("--decisions",type=Path,required=True)
    q=sub.add_parser("evaluate");q.add_argument("--run-dir",type=Path,required=True);q.add_argument("--model",type=Path,required=True);q.add_argument("--decisions",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        if a.command=="prepare":
            from .selective import prepare
            m=prepare(load_json(a.local),a.output,a.exclude,a.train_per_dataset,a.validation_per_dataset)
            print("Frozen {} questions; excluded {} historical IDs.".format(len(m["sample_ids"]),m["historical_unique_ids"]))
        elif a.command=="collect":
            if a.tensor_parallel_size<1 or not 0<a.gpu_memory_utilization<1 or a.request_timeout<=0:raise ValueError("Invalid runtime settings")
            collect(a)
        elif a.command=="audit":
            cases,m=validated_cases(a.run_dir,allow_smoke=True)
            new_file(a.output,{"version":VERSION,"mechanical_checks_passed":True,"questions":len(cases),"smoke":m["smoke"],"metrics":summarize(cases),"note":"CPU artifact/raw-cost audit; not runtime tokenizer or semantic validation."})
        else:{"fit":train,"select":select,"evaluate":evaluate}[a.command](a)
    except (ValueError,KeyError,FileNotFoundError) as exc:p.error(str(exc))
    return 0


if __name__=="__main__":raise SystemExit(main())

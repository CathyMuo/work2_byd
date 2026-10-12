"""CPU-only paired audit and semantic-review packet from existing raw A1/A3.

Input contract: A3 JSONL cases with sample_id/dataset/target_turn/status/branches;
branch records retain the shared rollout schema. Unknown schemas fail explicitly.
No model, retrieval service, or automatic semantic judge is invoked.
"""
import argparse
import collections
import json
import random
from pathlib import Path

from .common import file_hash, git_output, write_json
from .datasets import read_rows
from .metrics import answer_scores
from .replay import continuation_signature, observation, token_hash

CONDITIONS = ("E1_permutation", "E2_distractor", "E3_bridge_removal", "E4_redundancy")


def parse_mapping(values):
    mapping = {}
    for item in values:
        condition, separator, name = item.partition("=")
        if not separator or condition not in CONDITIONS or not name or condition in mapping:
            raise ValueError("Use unique --branch CONDITION=KEY entries for E1..E4")
        mapping[condition] = name
    return mapping


def condition_branches(case, mapping):
    branches = case["branches"]
    result = {}
    for condition in CONDITIONS:
        if condition in mapping:
            if mapping[condition] in branches:
                result[condition] = branches[mapping[condition]]
        else:
            matches = [r for r in branches.values() if r.get("condition") == condition]
            if len(matches) > 1:
                raise ValueError("Ambiguous branch for {}; provide --branch".format(condition))
            if matches:
                result[condition] = matches[0]
    unknown = [k for k, r in branches.items() if k not in ("clean_a", "clean_b")
               and r.get("condition") not in CONDITIONS and k not in mapping.values()]
    if unknown:
        raise ValueError("Unrecognized intervention branches {}; export documented schema or specify --branch".format(unknown))
    return result


def validate_trajectory(row, source, turn):
    for key in ("sample_id", "dataset", "question", "gold_answer", "gold_supporting_facts"):
        if row[key] != source[key]:
            raise ValueError("Branch/source labels disagree: " + key)
    if row["target_turn"] != turn or row["status"] == "infrastructure_error":
        raise ValueError("Invalid target turn or infrastructure failure")
    if any(row[k] != answer_scores(row["final_answer"], row["gold_answer"])[k] for k in ("em", "f1")):
        raise ValueError("Branch scores disagree with its raw final answer")
    if row["prefixes"][:turn-1] != source["prefixes"][:turn-1] or row["steps"][:turn] != source["steps"][:turn]:
        raise ValueError("Pre-target history changed")
    a, b = row["prefixes"][turn-1], source["prefixes"][turn-1]
    if a["prefix_token_ids"] != b["prefix_token_ids"] or a["prefix_token_sha256"] != token_hash(a["prefix_token_ids"]):
        raise ValueError("Target raw prefix identity/checksum failed")
    if a["prefix_text"] != b["prefix_text"] or a["search_query"] != b["search_query"]:
        raise ValueError("Target prefix text/query changed")
    for p in row["prefixes"]:
        if p["observation_text"] != observation(p["visible_documents"]):
            raise ValueError("Observed text disagrees with visible documents")
        if p["observed_document_ids"] != [d["document_id"] for d in p["visible_documents"]]:
            raise ValueError("Visible document IDs disagree")
    if len(a["visible_documents"]) != len(b["visible_documents"]):
        raise ValueError("Visible top-k changed")
    if any(p["visible_documents"] != p["candidate_documents"][:len(p["visible_documents"])] for p in row["prefixes"][turn:]):
        raise ValueError("Later retrieval was modified")


def audit_rows(sources, cases, target_turns, mapping):
    by_id = {r["sample_id"]: r for r in sources}
    if not sources or len(by_id) != len(sources):
        raise ValueError("A1 source IDs empty or duplicated")
    expected = {(sid,t) for sid in by_id for t in target_turns}
    actual = [(r["sample_id"],r["target_turn"]) for r in cases]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("A3 coverage incomplete/overlapping: expected={}, recorded={}, unique={}".format(len(expected),len(actual),len(set(actual))))
    pairs, events, counters = [], [], collections.Counter()
    for case in cases:
        source, turn = by_id[case["sample_id"]], case["target_turn"]
        if case["dataset"] != source["dataset"]:
            raise ValueError("Dataset changed")
        counters[case["status"]] += 1
        reachable = len(source["prefixes"]) >= turn
        if case["status"] == "unreachable":
            if reachable:
                raise ValueError("Reachable source marked unreachable")
            continue
        if not reachable:
            if case["status"] == "all_conditions_skipped" and not case["branches"]:
                continue
            raise ValueError("Unreachable source has replay branches")
        if case["status"] == "all_conditions_skipped":
            if case["branches"]:
                raise ValueError("Skipped state contains unaccounted branches")
            continue
        if case["status"] not in ("completed", "permutation_unchanged"):
            raise ValueError("Failed/unfinished A3 case: " + case["status"])
        branches = case["branches"]
        clean, control = branches["clean_a"], branches["clean_b"]
        for row in (clean, control):
            validate_trajectory(row, source, turn)
            if row["prefixes"][turn-1] != source["prefixes"][turn-1]:
                raise ValueError("Clean target observation differs from A1")
        if continuation_signature(clean) != continuation_signature(control):
            raise ValueError("Raw E0/E0 control mismatch")
        if continuation_signature(clean) != continuation_signature({**source,"target_turn":turn}):
            raise ValueError("Fresh clean continuation differs from A1")
        interventions = condition_branches(case,mapping)
        if not interventions:
            if case["status"] == "permutation_unchanged":
                continue
            raise ValueError("Completed case contains no recognized intervention")
        for condition, row in interventions.items():
            validate_trajectory(row, source, turn)
            original, changed = source["prefixes"][turn-1], row["prefixes"][turn-1]
            if condition == "E1_permutation":
                signature = lambda docs: collections.Counter(json.dumps(d,sort_keys=True,ensure_ascii=False) for d in docs)
                if signature(original["visible_documents"]) != signature(changed["visible_documents"]):
                    raise ValueError("E1 document multiset changed")
            elif sum(a != b for a,b in zip(original["visible_documents"],changed["visible_documents"])) != 1:
                raise ValueError("E2/E3/E4 must replace exactly one slot")
            if changed["observation_text"] == original["observation_text"]:
                if continuation_signature(row) != continuation_signature(clean):
                    raise ValueError("Unchanged observation produced a different continuation")
                counters[condition+"_unchanged_observation"] += 1
                continue
            pair = {"sample_id":source["sample_id"],"dataset":source["dataset"],"target_turn":turn,
                    "condition":condition,"clean_em":clean["em"],"perturbed_em":row["em"],
                    "delta_em":row["em"]-clean["em"],"delta_f1":row["f1"]-clean["f1"],
                    "same_subset_e1_delta_em":row["em"]-interventions["E1_permutation"]["em"] if "E1_permutation" in interventions else None}
            pairs.append(pair)
            if clean["em"] == 1 and row["em"] == 0:
                # Outcomes/continuation deliberately omitted from the first review view.
                events.append({"sample_id":source["sample_id"],"dataset":source["dataset"],"target_turn":turn,
                               "condition":condition,"question":source["question"],"gold_answer":source["gold_answer"],
                               "gold_supporting_facts":source["gold_supporting_facts"],
                               "visible_observations":[{"search_turn":p["search_turn"],"documents":p["visible_documents"]} for p in row["prefixes"]],
                               "review":{"decision":"pending","reviewer":"","earliest_sufficient_turn":None,
                                         "supporting_quotes":[],"reason":""}})
    return pairs,events,dict(counters)


def paired_interval(rows, field, resamples, seed):
    """Stratified question-cluster bootstrap of state-weighted paired means."""
    valid = [r for r in rows if r[field] is not None]
    if not valid:
        return {"states":0,"questions":0,"estimate":None,"ci95":None}
    strata = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in valid:
        strata[r["dataset"]][r["sample_id"]].append(r[field])
    clusters = [[(sum(values),len(values)) for values in questions.values()] for questions in strata.values()]
    question_count = sum(len(s) for s in clusters)
    result = {"states":len(valid),"questions":question_count,"estimate":sum(r[field] for r in valid)/len(valid),"ci95":None}
    if question_count < 2 or any(len(s) < 2 for s in clusters) or resamples == 0:
        result["interval_note"] = "Not estimated: need >=2 questions in each dataset stratum and nonzero resamples"
        return result
    rng, draws = random.Random(seed),[]
    for _ in range(resamples):
        total,denominator = 0,0
        for stratum in clusters:
            for value,count in rng.choices(stratum,k=len(stratum)):
                total += value
                denominator += count
        draws.append(total/denominator)
    draws.sort()
    result["ci95"] = [draws[int((resamples-1)*0.025)],draws[int((resamples-1)*0.975)]]
    return result


def summarize_pairs(pairs,resamples=2000,seed=42):
    def group(rows):
        return {"states":len(rows),"questions":len({(r["dataset"],r["sample_id"]) for r in rows}),
                "degradations":sum(r["clean_em"]==1 and r["perturbed_em"]==0 for r in rows),
                "improvements":sum(r["clean_em"]==0 and r["perturbed_em"]==1 for r in rows),
                "delta_em":paired_interval(rows,"delta_em",resamples,seed),
                "delta_f1":paired_interval(rows,"delta_f1",resamples,seed),
                "excess_over_e1_same_states":paired_interval(rows,"same_subset_e1_delta_em",resamples,seed)}
    return {"by_condition":{c:group([r for r in pairs if r["condition"]==c]) for c in CONDITIONS},
            "by_condition_and_turn":{c:{str(t):group([r for r in pairs if r["condition"]==c and r["target_turn"]==t]) for t in sorted({r["target_turn"] for r in pairs})} for c in CONDITIONS},
            "by_condition_and_dataset":{c:{d:group([r for r in pairs if r["condition"]==c and r["dataset"]==d]) for d in sorted({r["dataset"] for r in pairs})} for c in CONDITIONS},
            "bootstrap":{"unit":"whole question, keeping correlated target turns together", "strata":"dataset", "estimand":"eligible-state-weighted paired mean", "resamples":resamples,"seed":seed},
            "interpretation":"Exploratory paired intervals, not independent confirmation or a research/RL GO. Missing E1 matches excluded only from excess contrast, with explicit denominators. Semantic eligibility is not certified by this script."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-rollouts",type=Path)
    parser.add_argument("--a3-rollouts",type=Path,required=True)
    parser.add_argument("--schema-only",action="store_true",help="print only keys/branch names from the first record with branches, without reading all trajectories")
    parser.add_argument("--target-turns",nargs="+",type=int,default=[1,2])
    parser.add_argument("--branch",action="append",default=[],help="Explicit CONDITION=branch_key mapping; condition fields used otherwise")
    parser.add_argument("--resamples",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/semantic_audit"))
    args=parser.parse_args()
    if args.schema_only:
        try:
            with args.a3_rollouts.open(encoding="utf-8") as stream:
                first = None
                for index,line in enumerate(stream):
                    if not line.strip():
                        continue
                    row=json.loads(line)
                    first=row if first is None else first
                    if row.get("branches") or index>=99:
                        break
                row=row if first is not None else {}
                print(json.dumps({"top_level_keys":sorted(row),"status":row.get("status"),
                                  "branches":{k:{"condition":v.get("condition"),"rollout_keys":sorted(v)} for k,v in row.get("branches",{}).items()}},ensure_ascii=False))
        except (ValueError,OSError,AttributeError,TypeError) as exc:
            parser.error(str(exc))
        return 0
    if args.a1_rollouts is None:
        parser.error("--a1-rollouts is required for an actual paired audit")
    try:
        if args.resamples < 0 or not args.target_turns or any(t not in (1,2) for t in args.target_turns) or len(set(args.target_turns)) != len(args.target_turns):
            raise ValueError("Invalid resamples or target turns")
        if args.output_dir.exists():
            raise ValueError("Output exists; preserve previous artifacts and use a new directory")
        pairs,events,counters=audit_rows(read_rows(args.a1_rollouts),read_rows(args.a3_rollouts),args.target_turns,parse_mapping(args.branch))
        summary=summarize_pairs(pairs,args.resamples,args.seed)
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        parser.error("Raw schema/provenance audit failed: {}. See docs/FACTORY_ALIGNMENT_AND_NEXT_STEPS.md; no guessed-schema fallback.".format(exc))
    args.output_dir.mkdir(parents=True)
    provenance={"a1_rollouts_sha256":file_hash(args.a1_rollouts),"a3_rollouts_sha256":file_hash(args.a3_rollouts),
                "analysis_commit":git_output("rev-parse","HEAD"),"input_branch_mapping":parse_mapping(args.branch),
                "source_paths":{"a1":str(args.a1_rollouts.resolve()),"a3":str(args.a3_rollouts.resolve())},
                "scope":"Raw source, score, coverage and paired-history checks; no model/index/service identity revalidation or semantic certification"}
    write_json(args.output_dir/"paired_audit.json",{"provenance":provenance,"statuses":counters,**summary})
    write_json(args.output_dir/"paired_events.json",pairs)
    write_json(args.output_dir/"sufficiency_review.json",{"schema_version":1,"provenance":provenance,
               "selection":"ALL clean-correct to perturbed-wrong condition events; multiple events may share a state/question. Outcome-selected development audit, not prevalence estimate.",
               "degradation_events":len(events),"distinct_states":len({(r["sample_id"],r["target_turn"]) for r in events}),
               "allowed_decisions":["sufficient","insufficient","uncertain"],"events":events})
    print("CPU audit completed: {} paired events; {} degraded condition events across {} states. Semantic decisions remain pending.".format(len(pairs),len(events),len({(r["sample_id"],r["target_turn"]) for r in events})))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

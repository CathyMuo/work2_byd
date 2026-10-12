"""Audit compact recovery transitions exported from factory B artifacts, CPU only.

The explicit compact contract avoids guessing the unavailable factory B schema.
Requires E0/perturbed scores matching audit_pairs' validated paired_events.json.
"""
import argparse
import collections
from pathlib import Path

from .common import file_hash, git_output, write_json
from .datasets import read_rows
from .audit_pairs import paired_interval


def audit_recovery(pairs, rows, resamples=2000, seed=42):
    sources = {(r["sample_id"],r["target_turn"],r["condition"]):r for r in pairs}
    if len(sources)!=len(pairs):
        raise ValueError("Duplicate source intervention events")
    seen, by_baseline, assignments = set(),collections.defaultdict(list),collections.defaultdict(set)
    for row in rows:
        key=(row["sample_id"],row["target_turn"],row["condition"])
        baseline=row["baseline"]
        if not isinstance(baseline,str) or not baseline:
            raise ValueError("Baseline name missing")
        if key not in sources or (*key,baseline) in seen:
            raise ValueError("Unknown source event or duplicate recovery transition")
        seen.add((*key,baseline))
        source=sources[key]
        if row["dataset"]!=source["dataset"] or row["clean_em"]!=source["clean_em"] or row["perturbed_em"]!=source["perturbed_em"]:
            raise ValueError("Recovery transition disagrees with validated A3 source")
        if any(type(row[k]) is not int or row[k] not in (0,1) for k in ("clean_em","perturbed_em","recovered_em")):
            raise ValueError("EM fields must be integer 0/1; export the actual scores")
        if not isinstance(row.get("source_locator"),str) or not row["source_locator"].strip():
            raise ValueError("Transition needs a locator in original B artifacts")
        r={**row,"delta_em":row["recovered_em"]-row["perturbed_em"]}
        r["stratum"]="degraded" if row["clean_em"]==1 and row["perturbed_em"]==0 else "stable_clean_correct" if row["clean_em"]==1 else "clean_wrong"
        by_baseline[baseline].append(r)
        assignments[(row["sample_id"],row["target_turn"])].add(row["condition"])
    if not rows:
        raise ValueError("No recovery transitions")
    result={}
    for name,items in by_baseline.items():
        result[name]={"events":len(items),"distinct_states":len({(r["sample_id"],r["target_turn"]) for r in items}),
                      "gains":sum(r["delta_em"]==1 for r in items),"harms":sum(r["delta_em"]==-1 for r in items),
                      "net_correct_events":sum(r["delta_em"] for r in items),
                      "delta_em":paired_interval(items,"delta_em",resamples,seed),
                      "by_stratum":{s:{"events":len([r for r in items if r["stratum"]==s]),
                                         "gains":sum(r["delta_em"]==1 for r in items if r["stratum"]==s),
                                         "harms":sum(r["delta_em"]==-1 for r in items if r["stratum"]==s)} for s in ("degraded","stable_clean_correct","clean_wrong")}}
    transition_keys={tuple(k[:3]) for k in seen}
    return {"by_baseline":result,"validated_source_events":len(pairs),"selected_b_source_events":len(transition_keys),
            "condition_assignment_counts":dict(collections.Counter(c for conditions in assignments.values() for c in conditions)),
            "baseline_coverage":{name:{"selected_events":len(transition_keys),"recorded_events":len(items),
                                       "all_selected_events_recorded":len(items)==len(transition_keys)} for name,items in by_baseline.items()},
            "interpretation":"Input recovery scores are supplied by the export, not regraded from original B texts. Conditional gains/harms and net delta only on recorded events; missing events never scored as zero. Corpus reachability is not recovery. No full-cohort or deployable-trigger claim without complete coverage and pre-outcome condition assignment."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-events",type=Path,required=True)
    parser.add_argument("--transitions",type=Path,required=True)
    parser.add_argument("--assignment-policy",choices=("pre_outcome","outcome_selected","unknown"),default="unknown")
    parser.add_argument("--resamples",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",type=Path,default=Path("outputs/semantic_audit/recovery_audit.json"))
    args=parser.parse_args()
    try:
        if args.output.exists() or args.resamples<0:
            raise ValueError("Output exists or invalid resamples")
        result=audit_recovery(read_rows(args.paired_events),read_rows(args.transitions),args.resamples,args.seed)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,{"analysis_commit":git_output("rev-parse","HEAD"),
                            "paired_events_sha256":file_hash(args.paired_events),"transitions_sha256":file_hash(args.transitions),
                            "assignment_policy":args.assignment_policy,"bootstrap_resamples":args.resamples,"bootstrap_seed":args.seed,**result})
    print("Recovery transition audit written to {}. Original B artifacts unchanged.".format(args.output))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

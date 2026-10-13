"""Read-only CPU SI1 diagnostics; preserve frozen actions, selector and validation outcomes."""
import argparse
import collections
from pathlib import Path

from .common import file_hash, load_json
from .run_selective import new_file, validated_cases
from .selective import ARMS, choose, features, sha, summarize


def selection_details(cases, mapping):
    selected = [c for c in cases if mapping[c["sample_id"]] != "keep"]
    correct = sum(c["arms"]["keep"]["em"] == 1 for c in selected)
    wrong = len(selected)-correct
    gains = sum(c["arms"]["keep"]["em"] == 0 and c["arms"][mapping[c["sample_id"]]]["em"] == 1 for c in selected)
    harms = sum(c["arms"]["keep"]["em"] == 1 and c["arms"][mapping[c["sample_id"]]]["em"] == 0 for c in selected)
    return {"n":len(cases), "triggered":len(selected), "trigger_rate":len(selected)/len(cases),
            "triggered_baseline_correct":correct, "triggered_baseline_wrong":wrong,
            "gains":gains, "harms":harms, "EM_unchanged_triggered":len(selected)-gains-harms,
            "gain_per_trigger":gains/len(selected) if selected else None,
            "harm_per_trigger":harms/len(selected) if selected else None,
            "harm_given_trigger_and_baseline_correct":harms/correct if correct else None,
            "gain_given_trigger_and_baseline_wrong":gains/wrong if wrong else None}


def split_diagnostics(cases):
    gains = {arm:{c["sample_id"] for c in cases if c["arms"]["keep"]["em"] == 0 and c["arms"][arm]["em"] == 1} for arm in ARMS[1:]}
    union = set().union(*gains.values())
    patterns = collections.Counter("+".join(arm for arm in ARMS[1:] if c["sample_id"] in gains[arm]) for c in cases if c["sample_id"] in union)
    reachable = [c for c in cases if c["state"]["reachable"]]
    varying = []
    if reachable:
        vectors = [features(c["state"]) for c in reachable]
        varying = [i for i in range(len(vectors[0])) if len({x[i] for x in vectors}) > 1]
    return {"n":len(cases), "reachable":len(reachable), "arm_statistics":summarize(cases)["arms"],
            "oracle_gain_states":len(union), "gain_overlap_patterns":dict(patterns),
            "gain_states_exclusive_to_action":{arm:len(ids-set().union(*(gains[other] for other in ARMS[1:] if other != arm))) for arm,ids in gains.items()},
            "feature_variation":{"spent_call_values":sorted({c["state"]["spent_calls"] for c in reachable}),
                                 "remaining_call_values":sorted({c["state"]["remaining_calls"] for c in reachable}),
                                 "query_count_values":sorted({len(c["state"]["queries"]) for c in reachable}),
                                 "varying_feature_indices":varying},
            "endogenous_failure_note":"Malformed or search-refused outputs remain policy outcomes, not infrastructure errors; raw final text is needed to classify format failures."}


def diagnose(folder):
    cases, manifest = validated_cases(folder)
    model_path, decision_path, report_path = (folder/name for name in ("selector.json","decisions.json","validation_report.json"))
    model, decision, stored = load_json(model_path), load_json(decision_path), load_json(report_path)
    if decision["model_sha256"] != file_hash(model_path) or model["run_manifest_sha256"] != file_hash(folder/"run_manifest.json") or decision["states_sha256"] != file_hash(folder/"states.jsonl"):
        raise ValueError("Frozen selector/state/decision/run hashes mismatch")
    validation = [c for c in cases if c["split"] == "validation"]
    training = [c for c in cases if c["split"] == "train"]
    provenance = {"run_manifest_sha256":file_hash(folder/"run_manifest.json"),
                  "decision_sha256":file_hash(decision_path),"model_sha256":file_hash(model_path)}
    if stored["provenance"] != provenance or model["training_cases_sha256"] != sha(training):
        raise ValueError("Frozen report/training provenance mismatch")
    rows = decision["decisions"]
    by_id = {r["sample_id"]:r for r in rows}
    if not validation or len(rows) != len(by_id) or set(by_id) != {c["sample_id"] for c in validation}:
        raise ValueError("Validation decision coverage mismatch")
    settings = model["selector"]
    for c in validation:
        row = by_id[c["sample_id"]]
        predicted,scores = choose(model,c["state"],settings["risk_weight"],settings["margin"],settings["call_penalty"])
        if row["state_sha256"] != c["state_sha256"] or row["state_sha256"] != sha(c["state"]) or row["arm"] != predicted or row["scores"] != scores:
            raise ValueError("Decision is not reproducible at its frozen observable state")
    policies = {"selector":{sid:r["arm"] for sid,r in by_id.items()},
                "random_matched_rate":{sid:r["random_matched_arm"] for sid,r in by_id.items()},
                "simple_fixed_rule":{sid:r["simple_rule_arm"] for sid,r in by_id.items()},
                "risk_neutral_ablation":{sid:r["risk_neutral_arm"] for sid,r in by_id.items()}}
    stats = {name:summarize(validation,mapping)["selector"] for name,mapping in policies.items()}
    fixed = decision["frozen_best_fixed_arm"]
    if fixed != model["frozen_best_fixed_arm"] or fixed != stored["frozen_fixed_arm"]:
        raise ValueError("Training-frozen fixed baseline changed")
    stats["selector"]["net_em_vs_train_frozen_fixed"] = sum(c["arms"][policies["selector"][c["sample_id"]]]["em"]-c["arms"][fixed]["em"] for c in validation)/len(validation)
    if any(stored[name] != stats[name] for name in stats):
        raise ValueError("Predeclared stored controls disagree with raw SI1 rows")
    details = {name:selection_details(validation,mapping) for name,mapping in policies.items()}
    diag = split_diagnostics(validation)
    missed = diag["oracle_gain_states"]-details["selector"]["gains"]
    baseline = [c["arms"]["keep"] for c in validation]
    selected = [c["arms"][policies["selector"][c["sample_id"]]] for c in validation]
    mean_delta_calls = sum(s["search_calls"]-b["search_calls"] for s,b in zip(selected,baseline))/len(validation)
    observed_utility = (details["selector"]["gains"]-settings["risk_weight"]*details["selector"]["harms"])/len(validation)-settings["call_penalty"]*mean_delta_calls
    cost_only = sum(r["arm"] != "keep" and r["scores"][r["arm"]]["gain_score"]-settings["risk_weight"]*r["scores"][r["arm"]]["harm_score"] <= settings["margin"] for r in rows)
    return {"stage":"SI1_CPU_diagnostics", "mechanical_checks_passed":True,
            "source_hashes":{name:file_hash(folder/name) for name in ("run_manifest.json","cases.jsonl","states.jsonl","selector.json","decisions.json","validation_report.json")},
            "train":split_diagnostics(training), "validation":diag, "predeclared_policies":stats,
            "trigger_denominators":details, "frozen_fixed_arm":stored["frozen_fixed_arm"],
            "by_dataset":stored["by_dataset"],
            "selector_oracle_regret":{"missed_gain_states":missed,"avoidable_harm_states":details["selector"]["harms"],"total_EM_regret_states":missed+details["selector"]["harms"]},
            "selector_scoring":{"settings":settings,"observed_gain_harm_call_utility":observed_utility,
                                "triggers_crossing_margin_only_after_call_savings_term":cost_only,
                                "note":"Frozen ridge scores are clipped regression outputs, not calibrated probabilities. Cost-only crossings need not predict accuracy gains."},
            "interpretation":"CPU post-report diagnostics, no refitting or altered actions/evaluation. Same-rate random is the predeclared frozen realization, not a significance test. Train diagnostics are in-sample; they do not demonstrate held-out prediction. Keep validation observed; no tuning, fresh confirmation, new GPU or RL GO."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=Path("outputs/si1_dp4_dev"))
    parser.add_argument("--output",type=Path,default=Path("outputs/si1_dp4_diagnostics/report.json"))
    args = parser.parse_args()
    try:
        if args.output.exists():raise ValueError("Output exists; preserve previous diagnostics")
        result = diagnose(args.run_dir)
        new_file(args.output,result)
        print("SI1 CPU diagnostics saved; train {}, validation {}, triggered {}. No new inference.".format(result["train"]["n"],result["validation"]["n"],result["trigger_denominators"]["selector"]["triggered"]))
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

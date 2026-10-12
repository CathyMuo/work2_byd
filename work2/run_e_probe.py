"""E v2 uses the SAME control-first replay engine, with separate descriptive strata."""
from . import run_e
from .prepare_e_probe import reload_probe_plan


def metrics(records, planned):
    result = run_e.metrics(records, planned)
    for role, arms in result["by_role"].items():
        complete = [r for r in records if r["role"] == role and r["status"] == "completed"]
        contrasts = [(r["branches"]["general_review"], r["branches"]["constraint_review"]) for r in complete]
        keys = ("em", "f1", "generated_tokens_after_anchor", "model_input_tokens_after_anchor", "search_calls_after_anchor")
        arms["constraint_minus_general"] = {"completed_cases": len(contrasts), **{
            "mean_delta_"+key: sum(b[key]-a[key] for a,b in contrasts)/len(contrasts) if contrasts else None for key in keys}}
    result["v1_sufficient_pair_gate_passed"] = False
    result["stratum_coverage"] = {role: {
        "planned_cases": sum(c["role"] == role for c in planned),
        "completed_cases": sum(r["role"] == role and r["status"] == "completed" for r in records),
        "harm_measured": role == "stable_insufficient" and any(r["role"] == role and r["status"] == "completed" for r in records)
    } for role in ("failed_sufficient", "stable_insufficient")}
    result["interpretation"] = "V2 sufficient-failed primary repair and prefix-insufficient stable secondary harm are SEPARATE within-state development contrasts. No cross-state causal contrast, pooled population NetEM, parametric-knowledge attribution, B3 mechanism, F or RL GO. Missing secondary coverage is unmeasured harm, not zero harm."
    return result


def main():
    return run_e.main(plan_loader=reload_probe_plan, metric_fn=metrics, stage="E_stratified_probe",
                      default_plan="outputs/e_probe_review/reviewed.json", default_run_dir="outputs/e_probe_smoke")


if __name__ == "__main__":
    raise SystemExit(main())

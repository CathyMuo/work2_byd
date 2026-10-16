"""PCR Phase 0: CPU re-analysis of existing factory A1 + A3 raw artifacts (no GPU).

Same validated input contract as audit_pairs (A3 JSONL with clean_a/clean_b and
condition branches). Adds what the path-luck reading needs: both flip directions,
format-vs-semantic flip separation, and turn/dataset/answer-type/evidence strata.
"""
import argparse
import collections
import json
from pathlib import Path

from .audit_pairs import condition_branches, parse_mapping, validate_trajectory
from .common import file_hash, git_output, write_json
from .datasets import read_rows
from .pcr_analysis import answer_type, estimate, flip_kind, mean, ratio
from .replay import continuation_signature


def collect(sources, cases, mapping):
    by_id = {r["sample_id"]: r for r in sources}
    if len(by_id) != len(sources):
        raise ValueError("Duplicate A1 source IDs")
    rows, counters = [], collections.Counter()
    for case in cases:
        counters[case["status"]] += 1
        if case["status"] not in ("completed", "permutation_unchanged") or not case.get("branches"):
            continue
        source, turn = by_id[case["sample_id"]], case["target_turn"]
        clean, control = case["branches"]["clean_a"], case["branches"]["clean_b"]
        for row in (clean, control):
            validate_trajectory(row, source, turn)
        if continuation_signature(clean) != continuation_signature(control):
            raise ValueError("E0/E0 control mismatch: " + case["sample_id"])
        facts = {i for i, f in enumerate(source["gold_supporting_facts"]) if f.get("sentence")}
        candidates = {i for p in source["prefixes"] for i in p.get("candidate_lexical_fact_ids", [])}
        for condition, row in condition_branches(case, mapping).items():
            validate_trajectory(row, source, turn)
            if row["prefixes"][turn - 1]["observation_text"] == clean["prefixes"][turn - 1]["observation_text"]:
                counters[condition + "_unchanged_observation"] += 1
                continue
            rows.append({"sample_id": source["sample_id"], "dataset": source["dataset"], "turn": turn,
                         "condition": condition, "y0": clean["em"], "y1": row["em"],
                         "f1_0": clean["f1"], "f1_1": row["f1"],
                         "clean_answer": clean["final_answer"], "branch_answer": row["final_answer"],
                         "gold": source["gold_answer"], "question": source["question"],
                         "kind": flip_kind(clean["final_answer"], clean["em"], row["final_answer"], row["em"]),
                         "answer_type": answer_type(source["gold_answer"]),
                         "evidence_retrieved_proxy": bool(facts) and facts <= candidates,
                         "branch_status": row["status"], "clean_status": clean["status"]})
    return rows, dict(counters)


def block(rows, resamples):
    correct = [r for r in rows if r["y0"] == 1]
    wrong = [r for r in rows if r["y0"] == 0]
    deg_sem = lambda r: int(r["y0"] == 1 and r["y1"] == 0 and r["kind"] == "semantic")
    imp_sem = lambda r: int(r["y0"] == 0 and r["y1"] == 1 and r["kind"] == "semantic")
    return {"states": len(rows), "clean_correct": len(correct), "clean_wrong": len(wrong),
            "degraded_raw": sum(r["y1"] == 0 for r in correct), "improved_raw": sum(r["y1"] == 1 for r in wrong),
            "degraded_semantic": sum(map(deg_sem, rows)), "improved_semantic": sum(map(imp_sem, rows)),
            "format_flips": sum(r["kind"] == "format" for r in rows),
            "degradation_rate_semantic": estimate(rows, ratio(deg_sem, lambda r: r["y0"]), resamples),
            "improvement_rate_semantic": estimate(rows, ratio(imp_sem, lambda r: 1 - r["y0"]), resamples),
            "net_delta_em_raw": estimate(rows, mean(lambda r: r["y1"] - r["y0"]), resamples),
            "mean_abs_f1_change": sum(abs(r["f1_1"] - r["f1_0"]) for r in rows) / len(rows) if rows else None}


def summarize(rows, resamples):
    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == condition]
        out[condition] = {"all": block(sub, resamples)}
        for name, key in (("turn", lambda r: str(r["turn"])), ("dataset", lambda r: r["dataset"]),
                          ("answer_type", lambda r: r["answer_type"]),
                          ("evidence_retrieved_proxy", lambda r: str(r["evidence_retrieved_proxy"]))):
            out[condition][name] = {v: block([r for r in sub if key(r) == v], resamples // 4)
                                    for v in sorted({key(r) for r in sub})}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a1-rollouts", type=Path, required=True)
    p.add_argument("--a3-rollouts", type=Path, required=True)
    p.add_argument("--branch", action="append", default=[], help="CONDITION=branch_key, as in audit_pairs")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/pcr_phase0"))
    a = p.parse_args()
    try:
        if a.output_dir.exists():
            raise ValueError("Output exists; use a new directory")
        rows, counters = collect(read_rows(a.a1_rollouts), read_rows(a.a3_rollouts), parse_mapping(a.branch))
        if not rows:
            raise ValueError("No changed-observation pairs found; check schema with audit_pairs --schema-only")
        summary = summarize(rows, a.resamples)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        p.error("Phase 0 failed: {}. Run `python -m work2.audit_pairs --a3-rollouts ... --schema-only` and report the schema.".format(exc))
    a.output_dir.mkdir(parents=True)
    write_json(a.output_dir / "phase0_report.json", {
        "analysis_commit": git_output("rev-parse", "HEAD"),
        "inputs": {"a1_sha256": file_hash(a.a1_rollouts), "a3_sha256": file_hash(a.a3_rollouts)},
        "statuses": counters, "by_condition": summary,
        "format_flip_rule": "EM flip whose two answers are equal after normalization, contain each other, or token F1>=0.8",
        "note": "Single-perturbation flip rates; a single E1 degradation rate is a lower-bound indicator of lucky share, not p(s)."})
    with (a.output_dir / "format_flips_for_review.jsonl").open("x", encoding="utf-8") as f:
        for r in rows:
            if r["kind"] == "format":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (a.output_dir / "semantic_flips.jsonl").open("x", encoding="utf-8") as f:
        for r in rows:
            if r["kind"] == "semantic":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for condition, s in summary.items():
        b = s["all"]
        print("{}: states={} degraded raw/sem={}/{} of {} improved raw/sem={}/{} of {} format_flips={}".format(
            condition, b["states"], b["degraded_raw"], b["degraded_semantic"], b["clean_correct"],
            b["improved_raw"], b["improved_semantic"], b["clean_wrong"], b["format_flips"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

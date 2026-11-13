"""FAM1 (research/FAM1_PROTOCOL.md): do the main results carry over to non-Qwen (Llama) search agents?

freeze-e  -> question pool E (section 4.4): 500 new questions each from HotpotQA, 2Wiki, NQ and MuSiQue, in sha256(seed:id)
             order, disjoint by id AND normalized text from every canonical sample file under outputs/ (all earlier
             cohorts, the frozen 2,125 included) and the historical design files; cross-dataset dedup; outcome-blind.
fidelity  -> per roster agent: EM on the frozen questions vs its paper (section 6), format/length failures, flags.
analyze   -> readings F1-F8 (section 6), mechanical values only; the decision is written by hand into DECISION_LOG.md.

Runs live where the main agents' runs live: outputs/u1/<label>/{sources,pcr,as1,u1n} on the frozen questions and
outputs/u1/<label>/{sources_e,pcr_e} on pool E. Question-cluster bootstrap stratified by dataset, all agents' states of a
question together (the main table's rule), 2,000 resamples. Semantic view primary, raw EM view reported. CPU only.
"""
import argparse
import json
from pathlib import Path

from .arm_diagnose import diagnose
from .as1u1n import AGENTS as MAIN_AGENTS, EXPLAINED, RESCUE_PP, assay_block, natural_block, sensitive
from .audit_confirm import DEFAULT_EXCLUDE, write_selection
from .common import file_hash, git_output, load_json, local_path, write_json
from .datasets import canonicalize, canonicalize_flashrag, read_rows
from .paper_h1 import agent_rows
from .paper_robust import boot, effects
from .pcr import rank_key
from .selective import question_key
from .u1x import FAIL

VERSION = "FAM1_family_generalization_v1"
ROSTER = Path("configs/fam1_agents.json")
OLD_NONQWEN = ("zs_llama32_3b",)                                       # known before registration (section 0)
QWEN_MAIN = tuple(a for a in MAIN_AGENTS if a not in OLD_NONQWEN)      # the 13 Qwen agents of the main table
METHODS = {"review": "perm1_neutral", "reflect": "perm1_neutral_reflect"}
NEED = ("perm_1", "perm_2", "perm1_review", "perm1_neutral", "perm1_reflect", "perm1_neutral_reflect")
F4_UPPER = 0.10          # section 6: non-Qwen content effect upper bound below 10 pp
F5_UPPER = 0.01          # all-state method - P2 upper bound at most +1 pp
F2_SHARE = 2 / 3
E_SEED = 20261113
E_PER_DATASET = 500
E_DATASETS = (("hotpotqa", "hotpotqa_file", canonicalize), ("2wiki", "two_wiki_file", canonicalize),
              ("nq", "nq_file", canonicalize_flashrag), ("musique", "musique_file", canonicalize_flashrag))
# Section 6, read before any FAM1 run: Search-R1 arXiv 2503.09516v2 Table 3 (EM; base/instruct x PPO/GRPO, Llama-3.2-3B)
# and R1-Searcher arXiv 2503.05592v2 Table 2 (Llama-3.1-8B-Instruct, ACC_R = cover EM; BGE + KILT, up to 8 retrievals).
REFERENCE_EM = {
    "v01_l3b_ppo": {"nq": 0.394, "hotpotqa": 0.280, "2wiki": 0.264, "musique": 0.056, "bamboogle": 0.105},
    "v01_l3b_grpo": {"nq": 0.431, "hotpotqa": 0.300, "2wiki": 0.297, "musique": 0.067, "bamboogle": 0.104},
    "v01_l3b_it_ppo": {"nq": 0.357, "hotpotqa": 0.314, "2wiki": 0.233, "musique": 0.090, "bamboogle": 0.306},
    "v01_l3b_it_grpo": {"nq": 0.333, "hotpotqa": 0.229, "2wiki": 0.190, "musique": 0.047, "bamboogle": 0.192},
    "r1s_l8b": {"hotpotqa": 0.648, "2wiki": 0.594, "musique": 0.254, "bamboogle": 0.504},
}
GAP_FLAG = {"r1s_l8b": -0.20}        # metric and environment differ, so only a large gap flags (section 6)
DEFAULT_GAP_FLAG = -0.10


# ---------------------------------------------------------------- question pool E
def history(outputs, extra, ignore, design=DEFAULT_EXCLUDE):
    """Every canonical sample file of the project: all outputs/**/samples.jsonl plus the historical design files (the
    AC1/U1 exclusion list, which includes a file not named samples.jsonl) and any --exclude."""
    found = set(Path(outputs).rglob("samples.jsonl"))
    found |= {Path(p) for p in design} | {Path(p) for p in extra}
    skip = {Path(p).resolve() for p in ignore}
    paths = sorted(p for p in found if p.is_file() and p.resolve() not in skip)
    if not any(p.resolve() == Path(outputs, "u1_fresh", "samples.jsonl").resolve() for p in paths):
        raise ValueError("outputs/u1_fresh/samples.jsonl (the frozen questions) not found; run from the repository root")
    ids, questions, receipts = set(), set(), []
    for p in paths:
        rows = read_rows(p)
        bad = [i for i, r in enumerate(rows) if not (isinstance(r, dict) and isinstance(r.get("sample_id"), str)
                                                   and isinstance(r.get("question"), str))]
        if bad:
            raise ValueError("{} row {} has no sample_id/question; pass --ignore {} if it is not a question file".format(p, bad[0], p))
        ids |= {r["sample_id"] for r in rows}
        questions |= {question_key(r["question"]) for r in rows}
        receipts.append({"path": str(p), "sha256": file_hash(p), "rows": len(rows)})
    return ids, questions, receipts


def select_e(local, ids, questions, seed=E_SEED, per_dataset=E_PER_DATASET):
    chosen, seen, counts, sources = [], set(), {}, {}
    for dataset, key, canon in E_DATASETS:
        if key not in local or not local_path(local[key]).is_file():
            raise ValueError("local key {} missing or file not found (use the local file of the U1 freeze)".format(key))
        path = local_path(local[key])
        rows = [canon(r, dataset) for r in read_rows(path)]
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate ids in " + dataset)
        pool = sorted((r for r in rows if r["sample_id"] not in ids and question_key(r["question"]) not in questions),
                      key=lambda r: rank_key(seed, r["sample_id"]))
        picked = []
        for r in pool:
            q = question_key(r["question"])
            if q in seen:
                continue
            seen.add(q); picked.append(r)
            if len(picked) == per_dataset:
                break
        chosen += picked
        counts[dataset] = len(picked)
        sources[dataset] = {"path": str(path), "sha256": file_hash(path), "rows": len(rows)}
    return chosen, counts, sources


def freeze_e(args):
    if args.output.exists():
        raise ValueError("Output exists; keep frozen selections")
    ids, questions, receipts = history(args.outputs, args.exclude, args.ignore)
    chosen, counts, sources = select_e(load_json(args.local), ids, questions)
    write_selection(args.output, chosen, {
        "version": VERSION, "pool": "E", "seed": E_SEED, "per_dataset": E_PER_DATASET, "counts": counts,
        "shortfall": {k: E_PER_DATASET - v for k, v in counts.items() if v < E_PER_DATASET},
        "dataset_files": sources, "exclusions": receipts, "excluded_ids": len(ids), "excluded_questions": len(questions),
        "repository_commit": git_output("rev-parse", "HEAD"),
        "selection": "sha256(seed:sample_id) order per dataset; excluded by id and normalized question against every "
                     "outputs/**/samples.jsonl and the historical design files; cross-dataset dedup; outcome-blind"})
    print("Frozen pool E: {}; excluded {} ids / {} questions from {} files.".format(counts, len(ids), len(questions), len(receipts)))


# ---------------------------------------------------------------- fidelity
def agent_fidelity(root, label):
    path = Path(root) / label / "sources" / "sources.jsonl"
    if not path.is_file():
        return None
    rows = read_rows(path)
    em = {}
    for ds in sorted({r["dataset"] for r in rows}):
        xs = [r["em"] for r in rows if r["dataset"] == ds]
        em[ds] = sum(xs) / len(xs)
    ref = REFERENCE_EM.get(label, {})
    gaps = {ds: em[ds] - v for ds, v in ref.items() if ds in em}
    gap = sum(gaps.values()) / len(gaps) if gaps else None
    fail = sum(r["status"] in FAIL for r in rows) / len(rows)
    bound = GAP_FLAG.get(label, DEFAULT_GAP_FLAG)
    return {"questions": len(rows), "em": em, "em_all": sum(r["em"] for r in rows) / len(rows), "format_fail_rate": fail,
            "reference_em": ref or None, "gap_by_dataset": gaps or None, "macro_gap": gap, "flag_below": bound,
            "excluded_for_format_failure": fail >= .10, "possibly_misharnessed": bool(gap is not None and gap < bound)}


# ---------------------------------------------------------------- analysis
def runs(root, label, kind):
    d = Path(root) / label / kind
    return agent_rows(d) if (d / "states.jsonl").is_file() else []


def tag(rows, label, pool):
    return [dict(r, agent=label, pool=pool) for r in rows]


def parts(rows, view):
    y = lambda r, a: r[view][a]
    rows = [r for r in rows if all(a in r[view] for a in NEED)]
    return (rows, [r for r in rows if r["y0"] == 1 and y(r, "perm_1") == 0],
            [r for r in rows if r["y0"] == 1 and y(r, "perm_1") == 1])


def boot_sums(rows, comps, combine, resamples, seed=42):
    """Question-cluster bootstrap stratified by dataset (the main table's unit) of combine(totals), where the totals are
    sums of comps(r) over the resampled questions. Ratios of sums let one pass over the clusters serve every resample;
    multinomial counts per stratum are equivalent to drawing that stratum's questions with replacement."""
    clusters = {}
    for r in rows:
        v, key = comps(r), (r["dataset"], r["sample_id"])
        clusters[key] = v if key not in clusters else [a + b for a, b in zip(clusters[key], v)]
    if not clusters:
        return None
    width = len(next(iter(clusters.values())))
    total = [sum(v[i] for v in clusters.values()) for i in range(width)]
    out = {"n_states": len(rows), "n_questions": len(clusters), "estimate": combine(total), "ci95": None}
    strata = [[v for k, v in clusters.items() if k[0] == ds] for ds in sorted({k[0] for k in clusters})]
    if resamples == 0 or any(len(g) < 2 for g in strata):
        return out
    try:
        import numpy as np
        rng, totals = np.random.default_rng(seed), np.zeros((resamples, width))
        for g in strata:
            counts = rng.multinomial(len(g), [1.0 / len(g)] * len(g), size=resamples)
            totals += counts @ np.array(g, dtype=float)
        draws = [combine(list(t)) for t in totals]
    except ImportError:  # pure-Python fallback, same estimator
        import random
        rng, draws = random.Random(seed), []
        for _ in range(resamples):
            t = [0.0] * width
            for g in strata:
                for v in rng.choices(g, k=len(g)):
                    for i, x in enumerate(v):
                        t[i] += x
            draws.append(combine(t))
    draws = sorted(d for d in draws if d is not None)
    if len(draws) >= resamples * 0.9:
        out["ci95"] = [draws[int((len(draws) - 1) * .025)], draws[int((len(draws) - 1) * .975)]]
    return out


def luck_gap(rows, view, resamples):
    """F1: P1 repair on D minus the luck prediction harm(P1) |S| / |D| = (sum_D y(P1) - sum_S (1 - y(P1))) / |D|."""
    _, d, s = parts(rows, view)
    ds, ss = {id(r) for r in d}, {id(r) for r in s}
    comps = lambda r: [r[view]["perm_2"] if id(r) in ds else 0, 1 if id(r) in ds else 0,
                       1 - r[view]["perm_2"] if id(r) in ss else 0]
    return boot_sums(d + s, comps, lambda t: (t[0] - t[2]) / t[1] if t[1] else None, resamples)


def contrast(a_rows, b_rows, view, resamples):
    """Group A minus group B, per method, on repair (D) and on excess harm (S); joint question-cluster bootstrap, so a
    question shared by both groups is resampled once for both (the frozen questions are shared)."""
    y = lambda r, a: r[view][a]
    (_, ad, as_), (_, bd, bs) = parts(a_rows, view), parts(b_rows, view)
    diff = lambda t: t[0] / t[1] - t[2] / t[3] if t[1] and t[3] else None
    out = {}
    for m, p2 in METHODS.items():
        arm = "perm1_" + m
        rep = lambda r, a=arm, p=p2: y(r, a) - y(r, p)
        harm = lambda r, a=arm, p=p2: y(r, p) - y(r, a)
        out[m] = {"repair_minus_P2": boot_sums([dict(r, _g=0) for r in ad] + [dict(r, _g=1) for r in bd],
                                               lambda r, f=rep: [f(r), 1, 0, 0] if r["_g"] == 0 else [0, 0, f(r), 1], diff, resamples),
                  "harm_minus_P2": boot_sums([dict(r, _g=0) for r in as_] + [dict(r, _g=1) for r in bs],
                                             lambda r, f=harm: [f(r), 1, 0, 0] if r["_g"] == 0 else [0, 0, f(r), 1], diff, resamples)}
    return out


def pool_block(rows, view, resamples):
    rows, d, s = parts(rows, view)
    y = lambda r, a: r[view][a]
    out = {"states": len(rows), "degraded": len(d), "stable": len(s), "agents": sorted({r["agent"] for r in rows}),
           "P1_repair": sum(y(r, "perm_2") for r in d) / len(d) if d else None,
           "F1_luck_gap": luck_gap(rows, view, resamples) if d and s else None,
           "methods": {}}
    for m, p2 in METHODS.items():
        arm = "perm1_" + m
        rep = sum(y(r, arm) for r in d) / len(d) if d else None
        out["methods"][m] = {"repair": rep, "P2_repair": sum(y(r, p2) for r in d) / len(d) if d else None,
                             "repair_minus_P2": boot(d, lambda r, a=arm, p=p2: y(r, a) - y(r, p), resamples),
                             "repair_minus_P1": boot(d, lambda r, a=arm: y(r, a) - y(r, "perm_2"), resamples),
                             "harm_minus_P2": boot(s, lambda r, a=arm, p=p2: y(r, p) - y(r, a), resamples),
                             "all_minus_P2": boot(rows, lambda r, a=arm, p=p2: y(r, a) - y(r, p), resamples)}
    return out


def hi(e):
    return e["ci95"][1] if e and e.get("ci95") else None


def lo(e):
    return e["ci95"][0] if e and e.get("ci95") else None


def readings(block, versus_qwen):
    """F1-F5 on one non-Qwen pool (section 6); F4 per prompt: generalizes / family-dependent / undetermined."""
    m = block["methods"]
    f1 = block["F1_luck_gap"]
    out = {"F1": bool(f1 and f1["ci95"] and f1["ci95"][0] <= 0 <= f1["ci95"][1]),
           "F2": all(v["repair"] and v["P2_repair"] / v["repair"] >= F2_SHARE for v in m.values()),
           "F2_shares": {k: (v["P2_repair"] / v["repair"] if v["repair"] else None) for k, v in m.items()},
           "F3": all(hi(v["repair_minus_P1"]) is not None and hi(v["repair_minus_P1"]) < 0 for v in m.values()),
           "F5": all(hi(v["all_minus_P2"]) is not None and hi(v["all_minus_P2"]) <= F5_UPPER for v in m.values()),
           "F4": {}}
    for k, v in m.items():
        d = versus_qwen[k]["repair_minus_P2"]
        if lo(d) is not None and lo(d) > 0:
            verdict = "family_dependent"
        elif hi(v["repair_minus_P2"]) is not None and hi(v["repair_minus_P2"]) < F4_UPPER and lo(d) is not None:
            verdict = "generalizes"
        else:
            verdict = "undetermined"
        out["F4"][k] = {"verdict": verdict, "nonqwen_upper": hi(v["repair_minus_P2"]), "diff_vs_qwen": d}
    out["title_kept"] = out["F2"] and out["F3"]
    return out


def analyze(root, roster, resamples):
    agents = load_json(roster)["agents"]
    new = [a["label"] for a in agents if a["role"] == "new_nonqwen"]
    matched = [a["label"] for a in agents if a["role"] == "matched_qwen"]
    fidelity = {k: agent_fidelity(root, k) for k in new + matched + list(OLD_NONQWEN)}
    validity = {k: (diagnose(Path(root) / k / "pcr")["insertion_valid"] if (Path(root) / k / "pcr" / "states.jsonl").is_file() else None)
                for k in new + matched + list(OLD_NONQWEN)}
    status = {}
    for k in new + list(OLD_NONQWEN) + matched:
        f, v = fidelity[k], validity[k]
        status[k] = ("not_run" if f is None or v is None else "excluded_format_failure" if f["excluded_for_format_failure"]
                     else "insertion_invalid" if not v else "flagged_harness" if f["possibly_misharnessed"] else "included")
    included = [k for k in new + list(OLD_NONQWEN) if status[k] == "included"]
    if not included:
        raise ValueError("No included non-Qwen agent with a completed H1 run under " + str(root))
    frozen = {k: tag(runs(root, k, "pcr"), k, "frozen") for k in included + matched}
    extra = {k: tag(runs(root, k, "pcr_e"), k, "E") for k in included + matched}
    qwen = [r for k in QWEN_MAIN for r in tag(runs(root, k, "pcr"), k, "frozen")]
    if not qwen:
        raise ValueError("Main-table Qwen H1 runs not found under " + str(root))
    rows = lambda ks, pools=("frozen", "E"): [r for k in ks for src in (frozen, extra) for r in src.get(k, []) if r["pool"] in pools]
    versions = {"primary": rows(included), "new_only": rows([k for k in included if k not in OLD_NONQWEN]),
                "frozen_only": rows(included, ("frozen",))}
    report = {"version": VERSION, "analysis_commit": git_output("rev-parse", "HEAD"), "roster": str(roster),
              "agent_status": status, "fidelity": fidelity, "insertion_valid": validity, "included_nonqwen": included,
              "qwen_main": list(QWEN_MAIN), "pool_E_used": bool(any(extra[k] for k in included)),
              "bootstrap": {"unit": "question (all agents' states of a question together)", "strata": "dataset",
                            "resamples": resamples, "seed": 42},
              "views": {}}
    for view in ("sem", "raw"):
        out = {"qwen_main": pool_block(qwen, view, resamples), "versions": {}, "agents": {}}
        for name, rs in versions.items():
            if not rs:
                out["versions"][name] = None
                continue
            block = pool_block(rs, view, resamples)
            vs = contrast(rs, qwen, view, resamples)
            out["versions"][name] = dict(block, versus_qwen_main=vs, readings=readings(block, vs))
        for k in included + matched:
            rs = rows([k])
            out["agents"][k] = effects(rs, resamples, view) if rs else None
        llama_sr1 = [k for k in included if k.startswith("v01_l3b")]
        ok_matched = [k for k in matched if status[k] == "included"]
        out["F8_same_recipe"] = (contrast(rows(llama_sr1), rows(ok_matched), view, resamples)
                                 if llama_sr1 and ok_matched else None)
        out["F8_agents"] = {"llama": llama_sr1, "qwen": ok_matched}
        report["views"][view] = out
    # F6 / F7 on the frozen questions (AS1 and U1N are not run on pool E).
    as1 = {k: runs(root, k, "as1") for k in included}
    u1n = [r for k in included for r in runs(root, k, "u1n")]
    report["F6"] = {k: (sensitive(assay_block(v, resamples)) if v else None) for k, v in as1.items()}
    report["F6_all"] = bool(report["F6"]) and all(report["F6"].values())
    if u1n:
        nb = natural_block(u1n, resamples)
        report["F7"] = {m: {"N1_significant": bool(lo(v["rescue_minus_filler"]) is not None and lo(v["rescue_minus_filler"]) > 0
                                                    and v["rescue_minus_filler"]["estimate"] >= RESCUE_PP),
                            "N3_placebo_explains": bool(v["explained"] is not None and v["explained"] >= EXPLAINED),
                            "explained": v["explained"], "rescue_minus_filler": v["rescue_minus_filler"],
                            "all_minus_filler": v["all_minus_filler"]} for m, v in nb["methods"].items()}
    else:
        report["F7"] = None
    return report


def pct(e):
    if not isinstance(e, dict) or e.get("estimate") is None:
        return "NA"
    ci = e.get("ci95")
    return "{:+.1f} [{}]".format(100 * e["estimate"], "{:.1f}, {:.1f}".format(100 * ci[0], 100 * ci[1]) if ci else "no CI")


def markdown(r):
    L = ["# FAM1：非 Qwen 智能体上的推广（自动生成，供回传；决定由人写进 DECISION_LOG.md）", "",
         "分析提交：{}；题集 E 已用：{}".format(r["analysis_commit"], r["pool_E_used"]), "",
         "## 智能体状态", "", "| 智能体 | 状态 | 插入有效 | EM | 论文差 | 格式失败 |", "| --- | --- | --- | --- | --- | --- |"]
    f = lambda x: "NA" if x is None else "{:.3f}".format(x)
    for k, s in r["agent_status"].items():
        fi = r["fidelity"].get(k) or {}
        L.append("| {} | {} | {} | {} | {} | {} |".format(k, s, r["insertion_valid"].get(k), f(fi.get("em_all")),
                                                        f(fi.get("macro_gap")), f(fi.get("format_fail_rate"))))
    sem = r["views"]["sem"]
    L += ["", "## 判定（语义口径为主）", ""]
    for name, b in sem["versions"].items():
        if not b:
            L.append("- {}：无数据".format(name)); continue
        rd = b["readings"]
        L += ["### {}（D = {}，S = {}；智能体 {}）".format(name, b["degraded"], b["stable"], ", ".join(b["agents"])), "",
              "- F1 运气模型一致（设计检查）：{}，差 {}".format(rd["F1"], pct(b["F1_luck_gap"])),
              "- F2 安慰剂份额 ≥ 2/3：{}（{}）".format(rd["F2"], {k: f(v) for k, v in rd["F2_shares"].items()}),
              "- F3 方法 − P1 上界 < 0：{}".format(rd["F3"]),
              "- F5 全状态 方法 − P2 上界 ≤ +1 pp：{}".format(rd["F5"]),
              "- 标题保留条件（F2 且 F3）：{}".format(rd["title_kept"])]
        for m, v in rd["F4"].items():
            L.append("- F4 {}：**{}**；非 Qwen 修复内容效应 {}；与主表 13 个 Qwen 之差 {}".format(
                m, v["verdict"], pct(b["methods"][m]["repair_minus_P2"]), pct(v["diff_vs_qwen"])))
        L.append("")
    q = sem["qwen_main"]
    L += ["主表 13 个 Qwen 合并：复查 {}，反思 {}（D = {}）".format(pct(q["methods"]["review"]["repair_minus_P2"]),
                                                       pct(q["methods"]["reflect"]["repair_minus_P2"]), q["degraded"]), ""]
    L += ["## F6 阳性对照灵敏度（每个纳入的非 Qwen 智能体）", "", "```", json.dumps(r["F6"], ensure_ascii=False), "```", "",
          "## F7 自然失败", "", "```", json.dumps(r["F7"], ensure_ascii=False, default=str, indent=1), "```", "",
          "## F8 同配方族对比（四个 Search-R1 Llama − 四个 Qwen 配对）", "", "```",
          json.dumps({"agents": sem["F8_agents"], "contrast": sem["F8_same_recipe"]}, ensure_ascii=False, indent=1), "```", "",
          "## 逐智能体（语义口径）", "", "| 智能体 | D | 复查−P2 | 反思−P2 | 复查−P1 | 复查额外损害 | 全状态 复查−P2 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, e in sem["agents"].items():
        if not e:
            L.append("| {} | 未运行 | | | | | |".format(k)); continue
        mm = e["methods"]
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(k, e["degraded"], pct(mm["review"]["repair_minus_P2"]),
                 pct(mm["reflect"]["repair_minus_P2"]), pct(mm["review"]["repair_minus_P1"]), pct(mm["review"]["harm_minus_P2"]),
                 pct(mm["review"]["all_minus_P2"])))
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("freeze-e"); q.add_argument("--local", type=Path, required=True, help="the local file used for the U1 freeze")
    q.add_argument("--output", type=Path, default=Path("outputs/fam1_e")); q.add_argument("--outputs", type=Path, default=Path("outputs"))
    q.add_argument("--exclude", type=Path, nargs="*", default=[]); q.add_argument("--ignore", type=Path, nargs="*", default=[])
    q = sub.add_parser("fidelity"); q.add_argument("--root", type=Path, default=Path("outputs/u1"))
    q.add_argument("--roster", type=Path, default=ROSTER); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("analyze"); q.add_argument("--root", type=Path, default=Path("outputs/u1"))
    q.add_argument("--roster", type=Path, default=ROSTER); q.add_argument("--output", type=Path, required=True)
    q.add_argument("--resamples", type=int, default=2000)
    a = p.parse_args()
    try:
        if a.cmd == "freeze-e":
            freeze_e(a)
        elif a.cmd == "fidelity":
            labels = [x["label"] for x in load_json(a.roster)["agents"]] + list(OLD_NONQWEN)
            write_json(a.output, {k: agent_fidelity(a.root, k) for k in labels})
            print(json.dumps({k: v and {"em": round(v["em_all"], 3), "gap": v["macro_gap"], "flag": v["possibly_misharnessed"],
                                        "format_fail": v["format_fail_rate"]} for k, v in load_json(a.output).items()}, indent=1))
        else:
            if a.output.exists():
                raise ValueError("Output exists; use a new --output")
            report = analyze(a.root, a.roster, a.resamples)
            a.output.mkdir(parents=True)
            write_json(a.output / "fam1.json", report)
            (a.output / "fam1.md").write_text(markdown(report), encoding="utf-8")
            print(markdown(report))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

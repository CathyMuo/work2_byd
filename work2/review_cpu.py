"""CPU analyses requested by the third paper review (2026-11-04), on outputs that already exist. No model, no retrieval.

research/AS2_PROTOCOL.md section 8 (amendment 1). Five parts, all on the semantic view unless stated, with question-cluster
bootstrap stratified by dataset (all agents' states of a question resampled together):
  fourway   H1 degraded / stable states: method and its filler (and P1) cross-tabulated -- repaired by both, only one, neither;
            the overlap share of the method's repairs; the same for breaks on stable states.
  weights   content effect, excess harm and all-state effect pooled three ways: state-weighted (the paper's), every agent equal,
            every recipe equal; per-agent intervals with a Bonferroni level over the fourteen agents.
  alert     AS1: alert vs review directly (paired), and their content effects against their own fillers.
  policy    U1N as a question-level policy: intervene once, after the first retrieval, on every frozen question; questions the
            agent answers without searching keep their outcome. Final EM (raw), added searches and generated tokens per question.
  semantic  how often the semantic view changes an outcome label, per agent and arm, and the changed cases for hand review.
Outputs: <output>/review_cpu.json, review_cpu.md, semantic_cases.jsonl.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

from .as1u1n import AGENTS
from .common import file_hash, git_output, load_json, write_json
from .datasets import read_rows
from .paper_h1 import agent_rows
from .pcr import rank_key
from .pcr_analysis import flip_kind
from .placebo import usable_states

METHODS = {"review": ("perm1_review", "perm1_neutral"), "reflect": ("perm1_reflect", "perm1_neutral_reflect")}
POLICY_ARMS = ("clean_a", "review", "neutral", "reflect", "neutral_reflect", "perm_1")
POLICY_CONTRASTS = (("review", "clean_a"), ("review", "neutral"), ("reflect", "clean_a"), ("reflect", "neutral_reflect"),
                    ("perm_1", "clean_a"))
CASE_AGENTS = ("v03_32b_grpo", "research_32b")   # the two cells where the raw and semantic views disagree on a reading
CASE_SAMPLE = 300


def recipe_of(label):
    return ("ZeroSearch" if label.startswith("zs_") else "StepSearch" if label.startswith("stepsearch")
            else "ReSearch" if label.startswith("research") else "Search-R1")


class Boot:
    """Cluster bootstrap over (dataset, question) with per-group sums, so weighted means are cheap to resample."""
    def __init__(self, rows, group, resamples, seed=42):
        self.keys = sorted({(r["dataset"], r["sample_id"]) for r in rows})
        index = {k: i for i, k in enumerate(self.keys)}
        self.groups = sorted({group(r) for r in rows})
        gi = {g: j for j, g in enumerate(self.groups)}
        self.rows, self.cell = rows, [(index[(r["dataset"], r["sample_id"])], gi[group(r)]) for r in rows]
        strata = {}
        for i, k in enumerate(self.keys):
            strata.setdefault(k[0], []).append(i)
        rng = np.random.default_rng(seed)
        draws = [np.concatenate([rng.choice(np.array(v), size=len(v)) for _, v in sorted(strata.items())]) for _ in range(resamples)]
        # counts[b, c] = how often cluster c is drawn in resample b
        self.counts = np.stack([np.bincount(d, minlength=len(self.keys)) for d in draws]) if draws else np.zeros((0, len(self.keys)))

    def sums(self, value):
        s = np.zeros((len(self.keys), len(self.groups)))
        for r, (c, g) in zip(self.rows, self.cell):
            v = value(r)
            if v is not None:
                s[c, g] += v
        return s

    def interval(self, point, draws, alpha):
        draws = np.sort(draws[np.isfinite(draws)])
        if not len(draws):
            return None
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        return {"estimate": float(point), "ci95" if alpha == .05 else "ci": [float(lo), float(hi)], "alpha": alpha}

    def mean(self, value, weights="state", parent=None, alpha=.05):
        """Mean of value over rows where it is not None; weights = state | group (each group equal) | parent (groups averaged
        within their parent, then parents equal)."""
        num, den = self.sums(value), self.sums(lambda r: None if value(r) is None else 1.0)
        if not den.sum():
            return None

        def combine(n, d):
            with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # an agent or recipe absent from one resample
                if weights == "state":
                    return n.sum(-1) / d.sum(-1)
                m = n / d
                if weights == "group":
                    return np.nanmean(m, axis=-1)
                parents = sorted({parent(g) for g in self.groups})
                return np.nanmean(np.stack([np.nanmean(m[..., [j for j, g in enumerate(self.groups) if parent(g) == p]], axis=-1)
                                            for p in parents], axis=-1), axis=-1)
        point = combine(num.sum(0), den.sum(0))
        return self.interval(point, combine(self.counts @ num, self.counts @ den), alpha)

    def ratio(self, num, den, alpha=.05):
        n, d = self.sums(num).sum(1), self.sums(den).sum(1)
        if not d.sum():
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.interval(n.sum() / d.sum(), (self.counts @ n) / (self.counts @ d), alpha)


def h1_rows(root, labels):
    out = []
    for label in labels:
        d = root / label / "pcr"
        if (d / "states.jsonl").is_file():
            out += [dict(r, agent=label) for r in agent_rows(d)]
    return out


def split(rows, view="sem"):
    return ([r for r in rows if r["y0"] == 1 and r[view]["perm_1"] == 0], [r for r in rows if r["y0"] == 1 and r[view]["perm_1"] == 1],
            [r for r in rows if r["y0"] == 0])


def fourway(rows, resamples, view="sem"):
    deg, st, _ = split(rows, view)
    out = {"degraded": len(deg), "stable": len(st), "methods": {}}
    bd, bs = Boot(deg, lambda r: "all", resamples), Boot(st, lambda r: "all", resamples)
    for name, (arm, filler) in METHODS.items():
        entry = {}
        for other_name, other in (("filler", filler), ("P1", "perm_2")):
            y = lambda r, a: r[view][a]
            cells = {"both": lambda r, o=other: float(y(r, arm) == 1 and y(r, o) == 1),
                     "method_only": lambda r, o=other: float(y(r, arm) == 1 and y(r, o) == 0),
                     "other_only": lambda r, o=other: float(y(r, arm) == 0 and y(r, o) == 1),
                     "neither": lambda r, o=other: float(y(r, arm) == 0 and y(r, o) == 0)}
            broke = {"both": lambda r, o=other: float(y(r, arm) == 0 and y(r, o) == 0),
                     "method_only": lambda r, o=other: float(y(r, arm) == 0 and y(r, o) == 1),
                     "other_only": lambda r, o=other: float(y(r, arm) == 1 and y(r, o) == 0)}
            entry[other_name] = {
                "repair_cells": {k: bd.mean(f) for k, f in cells.items()},
                "repair_counts": {k: int(sum(f(r) for r in deg)) for k, f in cells.items()},
                "overlap_of_method_repairs": bd.ratio(cells["both"], lambda r: float(y(r, arm) == 1)),
                "overlap_of_other_repairs": bd.ratio(cells["both"], lambda r, o=other: float(y(r, o) == 1)),
                "break_cells": {k: bs.mean(f) for k, f in broke.items()},
                "break_counts": {k: int(sum(f(r) for r in st)) for k, f in broke.items()}}
        out["methods"][name] = entry
    return out


def weights(rows, labels, resamples, view="sem"):
    deg, st, _ = split(rows, view)
    agent = lambda r: r["agent"]
    boots = {"degraded": Boot(deg, agent, resamples), "stable": Boot(st, agent, resamples), "all": Boot(rows, agent, resamples)}
    out = {"agents": sorted({r["agent"] for r in rows}), "recipes": {}, "estimands": {}, "per_agent_bonferroni": {}}
    for a in out["agents"]:
        out["recipes"].setdefault(recipe_of(a), []).append(a)
    alpha = .05 / max(1, len(out["agents"]))
    for name, (arm, filler) in METHODS.items():
        values = {"repair_minus_filler": ("degraded", lambda r, a=arm, f=filler: r[view][a] - r[view][f]),
                  "harm_minus_filler": ("stable", lambda r, a=arm, f=filler: r[view][f] - r[view][a]),
                  "all_minus_filler": ("all", lambda r, a=arm, f=filler: r[view][a] - r[view][f])}
        for key, (pop, v) in values.items():
            b = boots[pop]
            out["estimands"]["{}_{}".format(name, key)] = {
                "state_weighted": b.mean(v), "agent_equal": b.mean(v, "group"), "recipe_equal": b.mean(v, "parent", recipe_of)}
            per = {}
            for a in out["agents"]:
                sub = [r for r in {"degraded": deg, "stable": st, "all": rows}[pop] if r["agent"] == a]
                per[a] = Boot(sub, agent, resamples).mean(v, alpha=alpha) if sub else None
            out["per_agent_bonferroni"]["{}_{}".format(name, key)] = {
                "alpha": alpha, "above": sum(bool(e and e["ci"][0] > 0) for e in per.values()),
                "below": sum(bool(e and e["ci"][1] < 0) for e in per.values()), "agents": per}
    return out


def alert_block(root, labels, resamples, view="sem"):
    rows = []
    for label in labels:
        d = root / label / "as1"
        if (d / "states.jsonl").is_file():
            rows += [dict(r, agent=label) for r in agent_rows(d)]
    if not rows:
        return None
    deg, st, _ = split(rows, view)
    bd, bs = Boot(deg, lambda r: "all", resamples), Boot(st, lambda r: "all", resamples)
    y = lambda r, a: r[view][a]
    return {"degraded": len(deg), "stable": len(st),
            "alert_minus_review_repair": bd.mean(lambda r: y(r, "perm1_alert") - y(r, "perm1_review")),
            "alert_content_minus_review_content": bd.mean(lambda r: (y(r, "perm1_alert") - y(r, "perm1_neutral_alert"))
                                                          - (y(r, "perm1_review") - y(r, "perm1_neutral"))),
            "alert_minus_review_harm": bs.mean(lambda r: y(r, "perm1_review") - y(r, "perm1_alert"))}


def policy_rows(root, label):
    """One row per frozen U1N question: arm outcomes after a single intervention at the first retrieval."""
    d = root / label / "u1n"
    m = load_json(d / "run_manifest.json")
    if m.get("status") != "completed" or file_hash(d / "states.jsonl") != m["states_sha256"]:
        raise ValueError("Need a completed, unmodified U1N run: " + str(d))
    plan = load_json(root / label / "u1n_plan" / "plan.json")
    sources = {r["sample_id"]: r for r in read_rows(plan["sources_path"])}
    raw_states = read_rows(d / "states.jsonl")
    kept = usable_states(raw_states, m)
    turn1 = {s["sample_id"]: s for s in kept if s["target_turn"] == 1}
    dropped = {s["sample_id"] for s in raw_states if s["target_turn"] == 1} - set(turn1)   # E0 mismatch: question left out
    # The frozen questions are the first N searchable ones in sha256(seed:id) order per dataset; the questions the agent answers
    # without searching that fall inside the same prefix of that order are the rest of the population.
    selected = set(plan["question_ids"])
    no_search = []
    for dataset in plan["datasets"]:
        order = sorted((r for r in sources.values() if r["dataset"] == dataset), key=lambda r: rank_key(plan["protocol"]["seed"], r["sample_id"]))
        picked = [i for i, r in enumerate(order) if r["sample_id"] in selected]
        last = picked[-1] if picked else -1
        no_search += [r for r in order[:last + 1] if not r["prefixes"]]
    rows = []
    for sid in plan["question_ids"]:
        s = turn1.get(sid)
        if s is None:
            continue
        clean = s["arms"]["clean_a"]
        row = {"agent": label, "sample_id": sid, "dataset": s["dataset"], "searched": True, "raw": {}, "sem": {}, "tokens": {}, "searches": {}}
        for a in POLICY_ARMS:
            x = s["arms"][a]
            row["raw"][a] = x["em"] or 0
            row["sem"][a] = clean["em"] if flip_kind(clean["final_answer"], clean["em"], x["final_answer"], x["em"] or 0) == "format" else (x["em"] or 0)
            row["tokens"][a] = x["generated_tokens_after_target"]
            row["searches"][a] = x["search_calls_after_target"]
        rows.append(row)
    for r in no_search:
        em = r["em"] or 0
        rows.append({"agent": label, "sample_id": r["sample_id"], "dataset": r["dataset"], "searched": False,
                     "raw": {a: em for a in POLICY_ARMS}, "sem": {a: em for a in POLICY_ARMS},
                     "tokens": {a: 0 for a in POLICY_ARMS}, "searches": {a: 0 for a in POLICY_ARMS}})
    return rows, {"frozen_questions": len(plan["question_ids"]), "with_turn1_state": len(turn1),
                  "no_search_questions": len(no_search), "excluded_for_control_mismatch": len(dropped)}


def policy(root, labels, resamples):
    rows, receipts = [], {}
    for label in labels:
        if not (root / label / "u1n" / "states.jsonl").is_file():
            continue
        r, receipts[label] = policy_rows(root, label)
        rows += r
    if not rows:
        return None
    b = Boot(rows, lambda r: "all", resamples)
    out = {"questions": len(rows), "receipts": receipts, "no_search_share": sum(not r["searched"] for r in rows) / len(rows),
           "em_raw": {a: b.mean(lambda r, a=a: r["raw"][a]) for a in POLICY_ARMS},
           "added_tokens": {a: float(np.mean([r["tokens"][a] for r in rows])) for a in POLICY_ARMS},
           "added_searches": {a: float(np.mean([r["searches"][a] for r in rows])) for a in POLICY_ARMS}, "contrasts": {}}
    for a, c in POLICY_CONTRASTS:
        out["contrasts"]["{}-{}".format(a, c)] = {
            "raw": b.mean(lambda r, a=a, c=c: r["raw"][a] - r["raw"][c]), "sem": b.mean(lambda r, a=a, c=c: r["sem"][a] - r["sem"][c]),
            "per_agent_raw": {k: Boot([r for r in rows if r["agent"] == k], lambda r: "all", resamples).mean(
                lambda r, a=a, c=c: r["raw"][a] - r["raw"][c]) for k in sorted(receipts)}}
    return out


def semantic(root, labels, output):
    """Counts of outcomes the semantic view changes, per agent and arm; changed cases written for hand review."""
    counts, cases = {}, []
    for label in labels:
        d = root / label / "pcr"
        if not (d / "states.jsonl").is_file():
            continue
        m = load_json(d / "run_manifest.json")
        per = counts.setdefault(label, {})
        for s in usable_states(read_rows(d / "states.jsonl"), m):
            clean = s["arms"]["clean_a"]
            for arm, x in s["arms"].items():
                if x is None or arm in ("clean_a", "clean_b"):
                    continue
                c = per.setdefault(arm, {"states": 0, "changed": 0})
                c["states"] += 1
                if flip_kind(clean["final_answer"], clean["em"], x["final_answer"], x["em"]) == "format":
                    c["changed"] += 1
                    cases.append({"agent": label, "sample_id": s["sample_id"], "turn": s["target_turn"], "arm": arm,
                                  "question": s["question"], "gold": s["gold_answer"], "clean_answer": clean["final_answer"],
                                  "clean_em": clean["em"], "arm_answer": x["final_answer"], "arm_em": x["em"]})
    key = lambda c: hashlib.sha256("{}|{}|{}|{}".format(c["agent"], c["sample_id"], c["turn"], c["arm"]).encode()).hexdigest()
    chosen = [c for c in cases if c["agent"] in CASE_AGENTS] + sorted((c for c in cases if c["agent"] not in CASE_AGENTS), key=key)[:CASE_SAMPLE]
    with (output / "semantic_cases.jsonl").open("w", encoding="utf-8") as f:
        for c in chosen:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    total = {"states": sum(v["states"] for a in counts.values() for v in a.values()),
             "changed": sum(v["changed"] for a in counts.values() for v in a.values())}
    return {"per_agent_arm": counts, "total": total, "cases_written": len(chosen), "cases_total": len(cases)}


def analyze(root, labels, resamples, output):
    rows = h1_rows(root, labels)
    if not rows:
        raise ValueError("No H1 runs under " + str(root))
    return {"analysis_commit": git_output("rev-parse", "HEAD"), "labels": list(labels), "resamples": resamples,
            "bootstrap": "question clusters (all agents together), stratified by dataset; numpy default_rng(42)",
            "fourway": {v: fourway(rows, resamples, v) for v in ("sem", "raw")}, "weights": weights(rows, labels, resamples),
            "alert": alert_block(root, labels, resamples), "policy": policy(root, labels, resamples),
            "semantic": semantic(root, labels, output)}


def f(e, pct=True):
    if not e:
        return "NA"
    k = 100 if pct else 1
    lo, hi = e.get("ci95") or e.get("ci")
    return "{:.1f} [{:.1f}, {:.1f}]".format(e["estimate"] * k, lo * k, hi * k)


def markdown(r):
    L = ["# 第三份审稿意见要求的 CPU 分析（自动生成，供回传）", "", "单位：百分点（比例为 %）；[] 为按题聚类 bootstrap 区间。语义口径为主。", ""]
    fw = r["fourway"]["sem"]
    L += ["## 1. 四格表：退化状态上方法与对照各修好了哪些（合并 14 个智能体，D = {}）".format(fw["degraded"]), "",
          "| 方法 | 对照 | 都修好 | 只有方法 | 只有对照 | 都没修好 | 方法修复中与对照重叠的比例 | 对照修复中与方法重叠的比例 | 稳定态：都弄坏 / 只方法 / 只对照 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, e in fw["methods"].items():
        for o, x in e.items():
            c, b = x["repair_cells"], x["break_cells"]
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} / {} / {} |".format(
                m, o, f(c["both"]), f(c["method_only"]), f(c["other_only"]), f(c["neither"]), f(x["overlap_of_method_repairs"]),
                f(x["overlap_of_other_repairs"]), f(b["both"]), f(b["method_only"]), f(b["other_only"])))
    L += ["", "计数（语义口径）：```{}```".format(json.dumps({m: {o: x["repair_counts"] for o, x in e.items()} for m, e in fw["methods"].items()})), ""]
    w = r["weights"]
    L += ["## 2. 三种加权：按状态 / 每个智能体等权 / 每种训练配方等权；逐智能体 Bonferroni 区间", "",
          "| 量 | 按状态 | 智能体等权 | 配方等权 | Bonferroni 下高于 0 / 低于 0 的智能体数 |", "| --- | --- | --- | --- | --- |"]
    for k, e in w["estimands"].items():
        p = w["per_agent_bonferroni"][k]
        L.append("| {} | {} | {} | {} | {} / {} |".format(k, f(e["state_weighted"]), f(e["agent_equal"]), f(e["recipe_equal"]), p["above"], p["below"]))
    L += ["", "配方分组：{}".format(json.dumps(w["recipes"], ensure_ascii=False)), ""]
    a = r["alert"]
    if a:
        L += ["## 3. AS1：alert 与 review 直接配对", "", "- 退化状态 alert − review 修复：{}".format(f(a["alert_minus_review_repair"])),
              "- 内容效应之差 (alert − 其文字) − (review − P2)：{}".format(f(a["alert_content_minus_review_content"])),
              "- 稳定状态 alert − review 伤害：{}".format(f(a["alert_minus_review_harm"])), ""]
    p = r["policy"]
    if p:
        L += ["## 4. 问题级策略：每题只在第一次检索后干预一次（U1N 题目，含不检索的题）", "",
              "题数 {}，其中不检索直接作答的占 {:.1f}%。".format(p["questions"], 100 * p["no_search_share"]), "",
              "| 臂 | 最终 EM（原始） | 每题新增检索 | 每题新增生成 token |", "| --- | --- | --- | --- |"]
        for arm in POLICY_ARMS:
            L.append("| {} | {} | {:.2f} | {:.0f} |".format(arm, f(p["em_raw"][arm]), p["added_searches"][arm], p["added_tokens"][arm]))
        L += ["", "| 对比 | 原始 EM 差 | 语义口径差 |", "| --- | --- | --- |"]
        for k, e in p["contrasts"].items():
            L.append("| {} | {} | {} |".format(k, f(e["raw"]), f(e["sem"])))
        L += ["", "逐智能体（原始 EM 差）：", "```", json.dumps({k: {a: f(v) for a, v in e["per_agent_raw"].items()} for k, e in p["contrasts"].items()},
                                                          ensure_ascii=False, indent=1), "```", "",
              "回执：```{}```".format(json.dumps(p["receipts"])), ""]
    s = r["semantic"]
    L += ["## 5. 语义口径改动的标签", "", "合计 {} 个臂结果中改动 {} 个；导出 {} 个案例到 semantic_cases.jsonl（共 {} 个）。".format(
        s["total"]["states"], s["total"]["changed"], s["cases_written"], s["cases_total"]), "", "```",
        json.dumps({a: {k: "{}/{}".format(v["changed"], v["states"]) for k, v in arms.items()} for a, arms in s["per_agent_arm"].items()},
                   ensure_ascii=False, indent=1), "```", ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--u1-root", type=Path, default=Path("outputs/u1"))
    p.add_argument("--labels", nargs="+", default=list(AGENTS))
    p.add_argument("--output", type=Path, required=True); p.add_argument("--resamples", type=int, default=2000)
    a = p.parse_args()
    try:
        if a.output.exists():
            raise ValueError("Output exists; use a new --output")
        a.output.mkdir(parents=True)
        report = analyze(a.u1_root, a.labels, a.resamples, a.output)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    write_json(a.output / "review_cpu.json", report)
    (a.output / "review_cpu.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

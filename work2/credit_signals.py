"""CT1 signal scoring on the SAME scored trajectories (HF, one model per GPU).

role=policy    : gold-answer mean log-prob at every s_j (IGPO Eq.3) and CIPO-style EALR
role=reference : gold-answer mean log-prob under the frozen reference model (TRACE-style)
Model-free signals (fact_hit, answer_hit, random) are computed in credit_analysis.
"""
import argparse
import json
import os
import time
from pathlib import Path

from .common import file_hash, load_json, local_path, write_json
from .credit_testbed import frozen, obs_end, spawn
from .datasets import read_rows

IGPO_PREFIX = "<think>Now there's enough information to answer</think><answer>"
IGPO_SUFFIX = "</answer>"
MASK_SELF_CHECK_TOLERANCE = 0.05


def gold_spans(trajectory, encode, gold):
    """(context_end, inserted ids, gold slice) for every state s_j, j = 0..T."""
    pre, ans, post = encode(IGPO_PREFIX), encode(gold), encode(IGPO_SUFFIX)
    out = []
    for j in range(trajectory["search_calls"] + 1):
        end = obs_end(trajectory, j)
        out.append({"j": j, "context_end": end, "ids": trajectory["token_ids"][:end] + pre + ans + post,
                    "target": (end + len(pre), end + len(pre) + len(ans))})
    return out


def ealr_spans(trajectory):
    """For observation t: its token span and the next generated segment (targets)."""
    segs = trajectory["segments"]
    out = []
    for i, s in enumerate(segs):
        if s["kind"] == "obs" and i + 1 < len(segs) and segs[i + 1]["kind"] == "gen" and segs[i + 1]["end"] > segs[i + 1]["start"]:
            out.append({"turn": s["turn"], "obs": (s["start"], s["end"]), "target": (segs[i + 1]["start"], segs[i + 1]["end"])})
    return out


def masked_rows(span):
    """Predictor rows whose attention to the observation block is removed (self kept)."""
    (o0, o1), (t0, t1) = span["obs"], span["target"]
    return [(r, o0, o1) for r in range(t0 - 1, t1 - 1)]


class Scorer:
    def __init__(self, path, attn):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                          attn_implementation=attn).cuda().eval()

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def target_logprobs(self, ids, start, end, rows=None):
        """Log-probs of ids[start:end]; optional additive 4D mask removing (row, k0, k1) attention."""
        torch = self.torch
        x = torch.tensor([ids[:end]], device="cuda")
        mask = None
        if rows is not None:
            n = end
            neg = torch.finfo(torch.bfloat16).min
            mask = torch.full((n, n), neg, dtype=torch.bfloat16, device="cuda").triu(1)
            for r, k0, k1 in rows:
                if r < n:
                    mask[r, k0:min(k1, r)] = neg
                    mask[r, min(k1, r + 1):k1] = neg  # stays causal; self (r) kept
            mask = mask[None, None]
        with torch.inference_mode():
            hidden = self.model.model(input_ids=x, attention_mask=mask).last_hidden_state[0]
            logits = self.model.lm_head(hidden[start - 1:end - 1]).float()
            lp = torch.log_softmax(logits, -1).gather(1, x[0, start:end, None])[:, 0]
        return lp.tolist()

    def self_check(self, ids, start, end, rows):
        plain = self.target_logprobs(ids, start, end)
        causal = self.target_logprobs(ids, start, end, rows=[])
        masked = self.target_logprobs(ids, start, end, rows=rows)
        diff = max(abs(a - b) for a, b in zip(plain, causal))
        if diff > MASK_SELF_CHECK_TOLERANCE:
            raise RuntimeError("Explicit 4D causal mask is not equivalent to the default mask (max diff {:.4f}); "
                               "this transformers/attention implementation does not honor custom masks".format(diff))
        if all(abs(a - b) < 1e-6 for a, b in zip(plain, masked)):
            raise RuntimeError("Evidence mask had no effect; custom 4D attention masks appear to be ignored")
        return {"explicit_causal_max_abs_diff": diff, "masked_differs": True}


def score_record(scorer, record, role):
    t = record["trajectory"]
    gold = []
    for s in gold_spans(t, scorer.encode, record["gold_answer"]):
        lp = scorer.target_logprobs(s["ids"], *s["target"])
        gold.append(sum(lp) / len(lp))
    out = {"sample_id": record["sample_id"], "role": role, "gold_mean_logprob": gold}
    if role == "policy":
        ealr = {}
        for span in ealr_spans(t):
            vis = scorer.target_logprobs(t["token_ids"], *span["target"])
            msk = scorer.target_logprobs(t["token_ids"], *span["target"], rows=masked_rows(span))
            ealr[str(span["turn"])] = sum(a - b for a, b in zip(vis, msk)) / len(vis)
        out["ealr"] = ealr
    return out


def worker(run_dir, index, role, attn):
    run, _ = frozen(run_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != run["gpus"][index]:
        raise ValueError("Worker GPU binding mismatch")
    local = load_json(run["local_file"])
    model = local_path(local["agent_checkpoint"] if role == "policy" else local["base_model"])
    records = read_rows(run_dir / "records.jsonl")[index::len(run["gpus"])]
    folder = run_dir / "signals" / role; folder.mkdir(parents=True, exist_ok=True)
    out = folder / "worker_{}.jsonl".format(index)
    done = read_rows(out) if out.exists() else []
    if [r["sample_id"] for r in done] != [r["sample_id"] for r in records[:len(done)]]:
        raise ValueError("Existing signal output does not match record order")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    scorer = Scorer(model, attn)
    manifest = {"role": role, "worker_index": index, "model": str(model), "attn_implementation": attn, "status": "running"}
    if role == "policy":
        probe = next((r for r in records if ealr_spans(r["trajectory"])), None)
        if probe is not None:
            span = ealr_spans(probe["trajectory"])[0]
            manifest["mask_self_check"] = scorer.self_check(probe["trajectory"]["token_ids"], *span["target"], masked_rows(span))
    write_json(folder / "worker_{}_manifest.json".format(index), manifest)
    started = time.monotonic()
    with out.open("a", encoding="utf-8") as stream:
        for i, record in enumerate(records[len(done):], len(done) + 1):
            stream.write(json.dumps(score_record(scorer, record, role)) + "\n"); stream.flush()
            if i % 20 == 0:
                print("signals {} worker {} {}/{}".format(role, index, i, len(records)), flush=True)
    write_json(folder / "worker_{}_manifest.json".format(index), dict(manifest, status="completed", seconds=time.monotonic() - started))


def merge(run_dir, role):
    run, _ = frozen(run_dir)
    records = read_rows(run_dir / "records.jsonl")
    rows = []
    for index in range(len(run["gpus"])):
        m = load_json(run_dir / "signals" / role / "worker_{}_manifest.json".format(index))
        got = read_rows(run_dir / "signals" / role / "worker_{}.jsonl".format(index))
        if m["status"] != "completed" or [r["sample_id"] for r in got] != [r["sample_id"] for r in records[index::len(run["gpus"])]]:
            raise ValueError("Signal worker {} incomplete".format(index))
        rows += got
    order = {r["sample_id"]: i for i, r in enumerate(records)}
    rows.sort(key=lambda r: order[r["sample_id"]])
    path = run_dir / "signals_{}.jsonl".format(role)
    with path.open("w", encoding="utf-8") as stream:
        for r in rows:
            stream.write(json.dumps(r) + "\n")
    m = load_json(run_dir / "run_manifest.json")
    m.setdefault("signals", {})[role] = {"sha256": file_hash(path), "records_sha256": file_hash(run_dir / "records.jsonl"), "rows": len(rows)}
    write_json(run_dir / "run_manifest.json", m)
    print("Merged {} {} signal rows.".format(len(rows), role))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("run"); q.add_argument("--run-dir", type=Path, required=True)
    q.add_argument("--role", choices=("policy", "reference"), default="policy"); q.add_argument("--attn", default="sdpa")
    q = sub.add_parser("worker"); q.add_argument("--run-dir", type=Path, required=True); q.add_argument("--index", type=int, required=True)
    q.add_argument("--role", choices=("policy", "reference"), required=True); q.add_argument("--attn", default="sdpa")
    q = sub.add_parser("merge"); q.add_argument("--run-dir", type=Path, required=True); q.add_argument("--role", choices=("policy", "reference"), required=True)
    a = p.parse_args()
    try:
        if a.command == "worker":
            worker(a.run_dir, a.index, a.role, a.attn)
        elif a.command == "merge":
            merge(a.run_dir, a.role)
        else:
            m = load_json(a.run_dir / "run_manifest.json")
            if m.get("status") != "completed" or file_hash(a.run_dir / "records.jsonl") != m["records_sha256"]:
                raise ValueError("Need a completed, unmodified CT1 MC run")
            run_plan, _ = frozen(a.run_dir)
            (a.run_dir / "logs").mkdir(exist_ok=True)
            spawn(a.run_dir, run_plan["gpus"], "work2.credit_signals", ["--role", a.role, "--attn", a.attn], "signals_" + a.role)
            merge(a.run_dir, a.role)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

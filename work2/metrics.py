import collections
import re
import string


def normalize(answer):
    value = answer.lower().translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def gold_list(gold):
    """Gold answers as a list: HotpotQA/2Wiki keep one string, FlashRAG sets (NQ, ...) carry aliases."""
    return [gold] if isinstance(gold, str) else list(gold)


def primary_gold(gold):
    return gold_list(gold)[0]


def answer_scores(prediction, gold):
    """EM/F1 against one gold string or the best-matching alias of a list (Search-R1 / FlashRAG convention)."""
    if not isinstance(gold, str):
        golds = gold_list(gold)
        if not golds:
            raise ValueError("Empty gold answer list")
        return max((answer_scores(prediction, g) for g in golds), key=lambda s: (s["em"], s["f1"]))
    prediction, gold = normalize(prediction), normalize(gold)
    em = int(prediction == gold)
    if prediction in ("yes", "no", "noanswer") or gold in ("yes", "no", "noanswer"):
        return {"em": em, "f1": float(em)}
    predicted, expected = prediction.split(), gold.split()
    if not predicted or not expected:
        return {"em": em, "f1": float(em)}
    shared = sum((collections.Counter(predicted) & collections.Counter(expected)).values())
    precision, recall = shared / len(predicted), shared / len(expected)
    return {"em": em, "f1": 2 * precision * recall / (precision + recall) if shared else 0.0}


def lexical_fact_hits(documents, facts):
    """Conservative sentence+title proxy, not corpus coverage or causal utility."""
    clean = lambda s: " ".join(s.casefold().split())
    return [i for i, fact in enumerate(facts) if fact.get("sentence")
            and any(clean(d["title"]) == clean(fact["title"])
                    and clean(fact["sentence"]) in clean(d["contents"])
                    for d in documents)]


def summarize(rows, expected):
    def group(items, planned):
        evaluable = [r for r in items if r["status"] != "infrastructure_error"]
        denominator = len(evaluable)
        mean = lambda field: sum(r[field] for r in evaluable) / denominator if denominator else None
        return {"expected_samples": planned, "recorded_samples": len(items),
                "evaluable_samples": denominator,
                "infrastructure_errors": len(items) - denominator,
                "em": mean("em"), "f1": mean("f1"),
                "valid_answer_rate": sum(r["status"] == "answered" for r in evaluable) / denominator if denominator else None,
                "at_least_one_search_rate": sum(r["search_calls"] >= 1 for r in evaluable) / denominator if denominator else None,
                "multi_turn_rate": sum(r["search_calls"] >= 2 for r in evaluable) / denominator if denominator else None,
                "avg_search_calls": mean("search_calls"), "avg_generated_tokens": mean("generated_tokens"),
                "avg_model_input_tokens": mean("model_input_tokens"),
                "avg_repeated_queries": mean("repeated_queries"),
                "generation_limit_rate": sum(r["status"] == "generation_limit" for r in evaluable) / denominator if denominator else None,
                "context_limit_rate": sum(r["status"] == "context_limit" for r in evaluable) / denominator if denominator else None,
                "search_budget_exhaustion_rate": sum(r["status"] == "search_budget_exhausted" for r in evaluable) / denominator if denominator else None,
                "known_supporting_sentences": sum(bool(f.get("sentence")) for r in evaluable for f in r["gold_supporting_facts"]),
                "missing_supporting_sentences": sum(not f.get("sentence") for r in evaluable for f in r["gold_supporting_facts"]),
                "statuses": dict(collections.Counter(r["status"] for r in items))}
    datasets = sorted(set(r["dataset"] for r in expected))
    return {"schema_version": 1, "overall": group(rows, len(expected)),
            "by_dataset": {d: group([r for r in rows if r["dataset"] == d],
                                    sum(r["dataset"] == d for r in expected)) for d in datasets},
            "interpretation": "A1 clean baseline only; no degradation, OffTarget, or Recovery metric is computed."}

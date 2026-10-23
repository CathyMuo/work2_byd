"""Read local HotpotQA/2Wiki data without downloading or synthesizing samples."""
import json
import random
from pathlib import Path

from .common import digest, file_hash, load_json, local_path, write_json


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("Expected a local JSON array or JSONL: {}".format(path))
    return rows


def canonicalize(row, dataset):
    if not isinstance(row, dict):
        raise ValueError("Dataset row must be an object")
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing nonempty question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Missing nonempty answer")
    question = question.strip()
    source_id = str(row.get("_id", row.get("id", digest(question.encode()))))
    context = row.get("context", [])
    if not isinstance(context, list):
        raise ValueError("Expected context as [[title, sentences], ...]")
    contexts = {}
    for item in context:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], list):
            raise ValueError("Malformed context entry")
        title, sentences = item
        if not isinstance(title, str) or not all(isinstance(s, str) for s in sentences):
            raise ValueError("Malformed context title/sentences")
        if title in contexts and contexts[title] != sentences:
            raise ValueError("Ambiguous duplicate context title: " + title)
        contexts[title] = sentences
    facts = []
    for item in row.get("supporting_facts", []):
        if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
                or not isinstance(item[1], int) or item[1] < 0):
            raise ValueError("Malformed supporting_facts entry")
        title, index = item
        sentences = contexts.get(title, [])
        facts.append({"title": title, "sentence_id": index,
                      "sentence": sentences[index] if index < len(sentences) else None})
    return {"sample_id": dataset + ":" + source_id, "source_id": source_id,
            "dataset": dataset, "question": question, "gold_answer": answer.strip(),
            "gold_supporting_facts": facts}


def canonicalize_flashrag(row, dataset):
    """FlashRAG / Search-R1 test format: {"id", "question", "golden_answers": [...]}; no supporting facts.

    gold_answer becomes the alias LIST (EM takes the best alias); gold_supporting_facts is empty,
    so lexical fact proxies (fact_hit, deletion arms) are unavailable for these datasets.
    """
    if not isinstance(row, dict):
        raise ValueError("Dataset row must be an object")
    question = row.get("question")
    answers = row.get("golden_answers", row.get("answers", row.get("answer")))
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing nonempty question")
    if not isinstance(answers, list) or not answers or not all(isinstance(a, str) for a in answers):
        raise ValueError("Missing golden_answers list")
    answers = list(dict.fromkeys(a.strip() for a in answers if a.strip()))
    if not answers:
        raise ValueError("Empty golden_answers")
    question = question.strip()
    source_id = str(row.get("id", row.get("_id", digest(question.encode()))))
    return {"sample_id": dataset + ":" + source_id, "source_id": source_id, "dataset": dataset,
            "question": question, "gold_answer": answers, "gold_supporting_facts": []}


def freeze_pilot(local, destination, count=100, seed=42):
    """Select local rows before outcomes; no network, new data, or label inference."""
    if count < 1:
        raise ValueError("count must be positive")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Output already exists; keep the frozen sample manifest: " + str(destination))
    selected, sources = [], []
    for dataset, key in (("hotpotqa", "hotpotqa_file"), ("2wiki", "two_wiki_file")):
        path = local_path(local[key])
        rows = [canonicalize(r, dataset) for r in read_rows(path)]
        ids = [r["sample_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate source IDs in " + dataset)
        question_keys = [" ".join(r["question"].casefold().split()) for r in rows]
        if len(question_keys) != len(set(question_keys)):
            raise ValueError("Duplicate question text in " + dataset)
        if len(rows) < count:
            raise ValueError("{} has {} rows, fewer than requested {}".format(dataset, len(rows), count))
        rows.sort(key=lambda r: r["sample_id"])
        random.Random(str(seed) + ":" + dataset).shuffle(rows)
        selected.extend(rows[:count])
        sources.append({"dataset": dataset, "path": str(path), "sha256": file_hash(path),
                        "total_rows": len(rows), "selected_ids": [r["sample_id"] for r in rows[:count]]})
    keys = [" ".join(r["question"].casefold().split()) for r in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Selected question text overlaps across datasets; audit sources before running")
    destination.mkdir(parents=True)
    samples = destination / "pilot.jsonl"
    with samples.open("x", encoding="utf-8") as output:
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(destination / "selection_manifest.json", {
        "schema_version": 1, "seed": seed, "count_per_dataset": count,
        "sources": sources, "pilot_sha256": file_hash(samples),
        "selection": "sorted unique IDs then seeded shuffle, before model outcomes",
        "split_note": "Source files must be development splits. No training/test overlap audit has run.",
    })
    return samples

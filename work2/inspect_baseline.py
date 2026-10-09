"""Write a bounded set of real baseline cases for human debugging."""
import argparse
import json
from pathlib import Path

from .common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    failures, successes = [], []
    with (args.run_dir / "clean_rollouts.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            bucket = successes if row["status"] == "answered" and row["em"] == 1 else failures
            if len(bucket) < args.count:
                bucket.append(row)
    selected = (failures + successes)[:args.count]
    output = args.run_dir / "review_cases.json"
    write_json(output, {"purpose": "Debugging cases, failures first; not a random research evaluation sample.",
                        "source": "clean_rollouts.jsonl", "cases": selected})
    print(output)


if __name__ == "__main__":
    main()

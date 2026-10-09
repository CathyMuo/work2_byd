"""Export bounded real replay cases for feedback, prioritizing engineering failures."""
import argparse
import json
from pathlib import Path

from .common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/a2_sanity"))
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    buckets = {0: [], 1: [], 2: []}
    with (args.run_dir / "replay_sanity.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            priority = (0 if row["status"] in ("control_failed", "infrastructure_error") else
                        1 if row.get("checks", {}).get("degradation") or row.get("checks", {}).get("improvement") else 2)
            if len(buckets[priority]) < args.count:
                buckets[priority].append(row)
    selected = (buckets[0] + buckets[1] + buckets[2])[:args.count]
    output = args.run_dir / "review_cases.json"
    write_json(output, {"purpose": "Engineering failures/outcome flips first; not a random evaluation sample.", "cases": selected})
    print(output)


if __name__ == "__main__":
    main()

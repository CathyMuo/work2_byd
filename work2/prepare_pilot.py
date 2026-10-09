"""Freeze a local pilot selection, without downloading or creating data."""
import argparse
from pathlib import Path

from .common import load_json
from .datasets import freeze_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=Path("configs/local.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/samples"))
    parser.add_argument("--count-per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        path = freeze_pilot(load_json(args.local), args.output, args.count_per_dataset, args.seed)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print("Frozen pilot: {}".format(path))
    print("Selection manifest: {}".format(path.parent / "selection_manifest.json"))


if __name__ == "__main__":
    main()

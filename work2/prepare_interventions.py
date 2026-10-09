"""Prepare an offline A3 review packet from frozen A1 and passed A2 artifacts."""
import argparse
from pathlib import Path

from .common import load_json, write_json
from .interventions import make_packet, validate_packet
from .run_replay import load_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=Path("outputs/a1_clean_200"))
    parser.add_argument("--a2-run", type=Path, default=Path("outputs/a2_sanity"))
    parser.add_argument("--output", type=Path, default=Path("outputs/a3_review/proposals.json"))
    parser.add_argument("--validate", type=Path, help="Check a finalized review packet, without model/service calls")
    args = parser.parse_args()
    try:
        source, rows = load_source(args.source_run)
        packet = make_packet(args.source_run, args.a2_run, source, rows)
        if args.validate:
            count = validate_packet(load_json(args.validate), packet)
            print("Finalized packet valid: {} approved; {} planned condition states. No inference performed.".format(count, len(packet["cases"])*3))
        else:
            if args.output.exists():
                raise ValueError("Output exists; preserve original proposals and use a new path")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, packet)
            print("Wrote {} target states to {}. Review all E2/E3/E4 entries before inference.".format(len(packet["cases"]), args.output))
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Data-sufficiency gate: is the SFT model actually using the <reasoning>/<answer> tag format
on this domain? Report 10's MATH pilot spent the full PRM/GRPO budget on three models that
turned out to have 0/500 tag-compliant completions -- this catches that failure right after SFT,
cheaply, before any of the expensive downstream steps run.

  python -m training.reasoning.check_format_compliance data/full_x_math.json --min-rate 0.5
"""
import argparse
import json
import sys
from pathlib import Path

from . import formats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json")
    ap.add_argument("--min-rate", type=float, default=0.5)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    rows = json.loads((root / args.eval_json).read_text())["rows"]
    n_ok = sum(1 for r in rows if formats.parse_completion(r["text"])["ok"])
    rate = n_ok / len(rows) if rows else 0.0
    print(f"format compliance: {n_ok}/{len(rows)} ({rate:.1%})")
    if rate < args.min_rate:
        print(f"FAIL: below --min-rate {args.min_rate:.1%}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

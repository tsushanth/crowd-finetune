import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import formats, sandbox
from .teacher import Teacher

sbx = sandbox.Sandbox()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/code_repair_train.jsonl")
    parser.add_argument("--out", default="data/code_repair_sft.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=1536)
    args = parser.parse_args()

    teacher = Teacher(args.teacher_model, system=formats.REPAIR_SYSTEM)
    root = Path(__file__).resolve().parent
    data_path = root / args.data if (root / args.data).exists() else args.data
    rows = [
        json.loads(line)
        for line in Path(data_path).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[args.skip : args.skip + args.limit]
    else:
        rows = rows[args.skip:]

    out = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def work(row):
        problem = {
            "tests": row["tests"],
            "imports": row.get("imports") or [],
            "entry_point": row.get("entry_point"),
        }
        try:
            text = teacher.generate(
                row["question"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception:
            return None
        if not text.strip():
            return None
        code = formats.extract_code(text)
        if not code:
            return None
        result = sbx.run(code, problem)
        if not result["ok"]:
            return None
        parsed = formats.parse_completion(text)
        return {
            "question": row["question"],
            "reasoning": parsed["reasoning"],
            "answer": code,
            "tests": row["tests"],
            "imports": row.get("imports") or [],
            "entry_point": row.get("entry_point"),
            "source": row.get("source", ""),
            "bug_type": row.get("bug_type", ""),
            "system": formats.REPAIR_SYSTEM,
        }

    kept = 0
    with out.open("w") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for trace in pool.map(work, rows, chunksize=1):
            if trace:
                kept += 1
                fh.write(json.dumps(trace) + "\n")
                fh.flush()
    print(f"kept {kept}/{len(rows)} repair traces -> {out}")


if __name__ == "__main__":
    main()
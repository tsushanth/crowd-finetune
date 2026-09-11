import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, db, gold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument(
        "--bench",
        default=str(config.BASE_DIR / "data" / "bench.jsonl"),
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-new", type=int, default=384)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(line) for line in Path(args.bench).read_text().splitlines() if line.strip()]
    rows = rows[: args.samples]
    if not rows:
        print("bench is empty; split some corpus items into data/bench.jsonl")
        return

    def bench_score() -> float:
        scores = []
        for row in rows:
            messages = [{"role": "user", "content": row["question"]}]
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            out = model.generate(
                ids.to(model.device),
                max_new_tokens=args.max_new,
                do_sample=False,
            )
            answer = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            res = gold.judge(
                "bench_grade", row["question"], answer, row.get("golden_answer")
            )
            scores.append(res["score"])
            print(f"{res['score']:.2f}\t{row['corpus_id']}")
        return sum(scores) / len(scores)

    base_score = bench_score()
    tuned_score = bench_score() if args.adapter else base_score
    delta = tuned_score - base_score
    verdict = "release" if delta >= config.MIN_EVAL_DELTA else "hold"
    print(f"base={base_score:.4f} tuned={tuned_score:.4f} delta={delta:+.4f} -> {verdict}")

    conn = db.connect()
    conn.execute(
        "INSERT INTO eval_runs (model_tag, base_bench, tuned_bench, delta, verdict) VALUES (?,?,?,?,?)",
        (args.adapter or args.base, base_score, tuned_score, delta, verdict),
    )
    conn.commit()


if __name__ == "__main__":
    main()
import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, db, gold, reward  # noqa: E402


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
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
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
            enc = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
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
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        tuned_score = bench_score()
    else:
        tuned_score = base_score
    delta = tuned_score - base_score
    verdict = "release" if delta >= config.MIN_EVAL_DELTA else "hold"
    print(f"base={base_score:.4f} tuned={tuned_score:.4f} delta={delta:+.4f} -> {verdict}")

    conn = db.connect()
    cur = conn.execute(
        "INSERT INTO eval_runs (model_tag, base_bench, tuned_bench, delta, verdict) VALUES (?,?,?,?,?)",
        (args.adapter or args.base, base_score, tuned_score, delta, verdict),
    )
    eval_run_id = cur.lastrowid
    conn.commit()

    if verdict == "release":
        accepted_sample_ids = [
            row["id"]
            for row in conn.execute(
                """
                SELECT a.id FROM accepted_samples a
                LEFT JOIN release_samples rs ON rs.accepted_sample_id = a.id
                WHERE rs.accepted_sample_id IS NULL
                """
            ).fetchall()
        ]
        if accepted_sample_ids:
            result = reward.settle(conn, eval_run_id, accepted_sample_ids)
            print(
                f"settled eval_run {eval_run_id}: credited={result['credited']} "
                f"paid_players={result['paid_players']}"
            )
        else:
            print(f"eval_run {eval_run_id} released but no unpaid accepted samples to settle")


if __name__ == "__main__":
    main()
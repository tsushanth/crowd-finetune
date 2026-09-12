import argparse
import json
import math
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, db, gold, reward  # noqa: E402

COMPETITOR_PRICES = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
}
DEFAULT_COMPETITORS = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]
COMPETITOR_ALIASES = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "haiku": "anthropic/claude-3-haiku",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "haiku-4.5": "anthropic/claude-haiku-4.5",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
}


def resolve_competitor(model_id: str) -> str:
    return COMPETITOR_ALIASES.get(model_id, model_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument(
        "--bench",
        default=str(config.BASE_DIR / "data" / "bench.jsonl"),
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-new", type=int, default=384)
    parser.add_argument("--competitor", action="store_true")
    parser.add_argument("--competitors", default="")
    parser.add_argument("--gpu-rate", type=float, default=0.47)
    parser.add_argument("--local-tokens-per-sec", type=float, default=0.0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.bench).read_text().splitlines() if line.strip()]
    rows = rows[: args.samples]
    if not rows:
        print("bench is empty; split some corpus items into data/bench.jsonl")
        return

    competitors = []
    if args.competitor:
        competitors = list(DEFAULT_COMPETITORS)
    if args.competitors:
        competitors = [resolve_competitor(m) for m in args.competitors.split(",") if m.strip()]
    if args.local_tokens_per_sec < 0:
        parser.error("--local-tokens-per-sec must be >= 0")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    prompts = [
        [
            {"role": "system", "content": gold._BASE_SYSTEM},
            {"role": "user", "content": row["question"]},
        ]
        for row in rows
    ]

    def local_generator(stats: dict):
        def generate(row: dict, prompt: list[dict]) -> str:
            enc = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
            stats["prompt_tokens"] += ids.shape[1]
            t0 = time.perf_counter()
            out = model.generate(ids.to(model.device), max_new_tokens=args.max_new, do_sample=False)
            stats["secs"] += time.perf_counter() - t0
            stats["out_tokens"] += out.shape[1] - ids.shape[1]
            return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        return generate

    def api_generator(model_id: str, stats: dict):
        def generate(row: dict, prompt: list[dict]) -> str:
            stats["prompt_tokens"] += len(
                tokenizer.encode(" ".join(m["content"] for m in prompt))
            )
            answer = gold.chat(gold._BASE_SYSTEM, row["question"], model=model_id).strip()
            stats["out_tokens"] += len(tokenizer.encode(answer))
            return answer

        return generate

    def score(generate, label: str) -> float:
        scores = []
        for row, prompt in zip(rows, prompts):
            answer = generate(row, prompt)
            res = gold.judge("bench_grade", row["question"], answer, row.get("golden_answer"))
            scores.append(res["score"])
            print(f"{res['score']:.2f}\t{row['corpus_id']}\t{label}")
        return sum(scores) / len(scores)

    table = []

    def add_local(label: str, run_score: float, stats: dict) -> None:
        tokens = stats["prompt_tokens"] + stats["out_tokens"]
        per_query = tokens / len(rows)
        if args.local_tokens_per_sec > 0:
            tps = args.local_tokens_per_sec
        elif stats["secs"] > 0:
            tps = tokens / stats["secs"]
        else:
            tps = 0.0
        cost = args.gpu_rate / (tps * 3600.0) * per_query * 1000.0 if tps > 0 else math.nan
        table.append((label, run_score, cost, per_query, tps))

    def add_api(label: str, run_score: float, stats: dict) -> None:
        per_in = stats["prompt_tokens"] / len(rows)
        per_out = stats["out_tokens"] / len(rows)
        price = COMPETITOR_PRICES.get(label)
        if price:
            cost = (per_in * price[0] + per_out * price[1]) / 1e6 * 1000.0
        else:
            cost = math.nan
        table.append((label, run_score, cost, per_in + per_out, None))

    local_stats = {"prompt_tokens": 0, "out_tokens": 0, "secs": 0.0}
    base_score = score(local_generator(local_stats), "local-base")
    tuned_stats = {"prompt_tokens": 0, "out_tokens": 0, "secs": 0.0}
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        tuned_score = score(local_generator(tuned_stats), "local-tuned")
    else:
        tuned_score = base_score
        tuned_stats = local_stats
    add_local(
        f"{Path(args.base).name} base", base_score, local_stats
    )
    add_local(
        f"{Path(args.base).name} tuned ({Path(args.adapter).name})" if args.adapter else "tuned=base",
        tuned_score,
        tuned_stats,
    )

    for model_id in competitors:
        api_stats = {"prompt_tokens": 0, "out_tokens": 0}
        api_score = score(api_generator(model_id, api_stats), model_id)
        add_api(model_id, api_score, api_stats)

    print()
    print(f"{'model':<42}{'score':>8}{'$/1K q':>10}{'tok/q':>8}{'tps':>10}")
    for label, run_score, cost, per_query, tps in table:
        cost_s = "n/a" if math.isnan(cost) else f"${cost:.5f}"
        tps_s = f"{tps:.0f}" if tps else "-"
        print(f"{label:<42}{run_score:>8.4f}{cost_s:>10}{per_query:>8.0f}{tps_s:>10}")

    for label, run_score, cost, per_query, tps in table:
        if tps:
            print(
                f"{label}: {tps:.0f} tok/s measured locally; "
                f"cost assumes GPU ${args.gpu_rate:.2f}/hr and {per_query:.0f} tokens/query"
            )

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
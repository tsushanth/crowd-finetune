import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend import config, gold  # noqa: E402

from . import formats, sandbox  # noqa: E402

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

sbx = sandbox.Sandbox()


def resolve_competitor(model_id: str) -> str:
    return COMPETITOR_ALIASES.get(model_id, model_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/code_eval.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--competitor", action="store_true")
    parser.add_argument("--competitors", default="")
    parser.add_argument("--gpu-rate", type=float, default=0.47)
    parser.add_argument("--local-tokens-per-sec", type=float, default=0.0)
    parser.add_argument("--out", default="data/code_eval_results.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_path = root / args.data if (root / args.data).exists() else args.data
    rows = [
        json.loads(line)
        for line in Path(data_path).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("eval set is empty; run datasets.py first")
        return

    competitors = []
    if args.competitor:
        competitors = list(DEFAULT_COMPETITORS)
    if args.competitors:
        competitors = [
            resolve_competitor(m) for m in args.competitors.split(",") if m.strip()
        ]
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    def local_samples(row, prompt, stats, k):
        enc = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
        stats["prompt_tokens"] += ids.shape[1]
        t0 = time.perf_counter()
        outs = model.generate(
            ids.to(model.device),
            max_new_tokens=args.max_new,
            do_sample=k > 1,
            temperature=0.8 if k > 1 else None,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
        )
        stats["secs"] += time.perf_counter() - t0
        stats["out_tokens"] += (outs.shape[1] - ids.shape[1]) * k
        return [
            tokenizer.decode(outs[j][ids.shape[1]:], skip_special_tokens=True)
            for j in range(outs.shape[0])
        ]

    def api_samples(row, prompt, stats, k):
        user = prompt[-1]["content"]
        samples = []
        for _ in range(k):
            stats["prompt_tokens"] += len(tokenizer.encode(" ".join(m["content"] for m in prompt)))
            answer = gold.chat(formats.CODE_SYSTEM, user, model=row["model"]).strip()
            stats["out_tokens"] += len(tokenizer.encode(answer))
            samples.append(answer)
        return samples

    def score_any_pass(problem, samples):
        for s in samples:
            res = sbx.run(formats.extract_code(s), problem, timeout=args.timeout)
            if res["ok"]:
                return 1.0
        return 0.0

    prompts = [
        [
            {"role": "system", "content": formats.CODE_SYSTEM},
            {"role": "user", "content": row["question"]},
        ]
        for row in rows
    ]
    problems = [
        {
            "tests": row["tests"],
            "imports": row.get("imports") or [],
            "entry_point": row.get("entry_point"),
        }
        for row in rows
    ]

    table = []

    def add_local(label, run_score, stats):
        tokens = stats["prompt_tokens"] + stats["out_tokens"]
        per_problem = tokens / len(rows)
        if args.local_tokens_per_sec > 0:
            tps = args.local_tokens_per_sec
        elif stats["secs"] > 0:
            tps = tokens / stats["secs"]
        else:
            tps = 0.0
        cost = args.gpu_rate / (tps * 3600.0) * per_problem * 1000.0 if tps > 0 else math.nan
        table.append((label, run_score, cost, per_problem, tps))

    def add_api(label, run_score, stats):
        per_in = stats["prompt_tokens"] / len(rows)
        per_out = stats["out_tokens"] / len(rows)
        price = COMPETITOR_PRICES.get(label)
        if price:
            cost = (per_in * price[0] + per_out * price[1]) / 1e6 * 1000.0
        else:
            cost = math.nan
        table.append((label, run_score, cost, per_in + per_out, None))

    def run_local(label):
        stats = {"prompt_tokens": 0, "out_tokens": 0, "secs": 0.0}
        scores = []
        for row, prompt, problem in zip(rows, prompts, problems):
            samples = local_samples(row, prompt, stats, args.pass_k)
            score = score_any_pass(problem, samples)
            scores.append(score)
            print(f"{score:.2f}\t{row['source']}\t{label}")
        run_score = sum(scores) / len(scores)
        add_local(label, run_score, stats)
        return run_score

    def run_api(label):
        stats = {"prompt_tokens": 0, "out_tokens": 0}
        scores = []
        for row, prompt, problem in zip(rows, prompts, problems):
            gen_row = {**row, "model": label}
            samples = api_samples(gen_row, prompt, stats, args.pass_k)
            score = score_any_pass(problem, samples)
            scores.append(score)
            print(f"{score:.2f}\t{row['source']}\t{label}")
        run_score = sum(scores) / len(scores)
        add_api(label, run_score, stats)
        return run_score

    label_base = Path(args.model).name
    base_score = run_local(f"{label_base} base")
    tuned_score = base_score
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        tuned_score = run_local(f"{label_base} tuned ({Path(args.adapter).name})")
    per_tuned = table[-1][3] if args.adapter else table[-1][3]

    for model_id in competitors:
        try:
            run_api(model_id)
        except Exception as exc:
            print(f"competitor {model_id} failed: {exc}")

    print()
    header = f"pass@{args.pass_k}"
    print(f"{'model':<46}{header:>8}{'$/1K gen':>10}{'tok/gen':>8}{'tps':>10}")
    for label, run_score, cost, per_query, tps in table:
        cost_s = "n/a" if math.isnan(cost) else f"${cost:.5f}"
        tps_s = f"{tps:.0f}" if tps else "-"
        print(f"{label:<46}{run_score:>8.4f}{cost_s:>10}{per_query:>8.0f}{tps_s:>10}")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.data,
                "pass_k": args.pass_k,
                "rows": len(rows),
                "rows_table": [
                    {"model": label, "pass": s, "cost_per_1k_gen": c,
                     "tokens_per_gen": t}
                    for label, s, c, t, _ in table
                ],
            },
            indent=2,
        )
    )
    print(f"base={base_score:.4f} tuned={tuned_score:.4f} delta={tuned_score - base_score:+.4f} -> {out}")


if __name__ == "__main__":
    main()
import argparse
import json
import time
from pathlib import Path

import httpx
from datasets import load_dataset

from . import formats

_model = None
_tokenizer = None


def local_completion(model_name, adapter, question, max_new, control=None, control_style="system"):
    global _model, _tokenizer
    if _model is None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if adapter:
            _model = PeftModel.from_pretrained(_model, adapter)
        _model.eval()
    messages = []
    system = formats.render_system(control, control_style) if control else None
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": formats.render_control(control, question, control_style)
            if control else question,
        }
    )
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[1]
    t0 = time.monotonic()
    out = _model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False
    )
    elapsed = time.monotonic() - t0
    completion_tokens = int(out.shape[1]) - input_len
    text = _tokenizer.decode(
        out[0][input_len:], skip_special_tokens=True
    )
    usage = {
        "prompt_tokens": int(input_len),
        "completion_tokens": completion_tokens,
        "elapsed": elapsed,
    }
    return text, usage


def remote_completion(endpoint, model, question, max_new, api_key=None, control=None, control_style="system"):
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    messages = []
    system = formats.render_system(control, control_style) if control else None
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": formats.render_control(control, question, control_style)
            if control else question,
        }
    )
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_new,
    }
    t0 = time.monotonic()
    resp = httpx.post(url, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    elapsed = time.monotonic() - t0
    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage") or {}
    usage = {
        "prompt_tokens": usage.get("prompt_tokens") or max(len(question) // 4, 1),
        "completion_tokens": usage.get("completion_tokens") or len(text.split()),
        "elapsed": elapsed,
    }
    return text, usage


def reference_mode(raw_answer: str) -> str:
    if "\n#### " in raw_answer:
        return "number"
    if formats.extract_last_number(raw_answer) == formats.normalize_number(raw_answer):
        return "number"
    return "string"


def matches(completion: str, raw_answer: str, mode: str, strict: bool = False) -> bool:
    if mode == "number":
        expected = formats.extract_last_number(raw_answer)
        if strict:
            parsed = formats.parse_completion(completion)
            got = parsed["answer"] if parsed["ok"] else None
            return bool(expected) and got is not None and formats.extract_last_number(got) == expected
        return bool(expected) and formats.exact_match(completion, expected)
    expected = formats.clean_for_tags(raw_answer)
    parsed = formats.parse_completion(completion)
    got = formats.clean_for_tags(parsed.get("answer") or "")
    return bool(got) and got == expected


def run_mode(mode, examples, args):
    stats = {
        "n": 0,
        "ok": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "elapsed": 0.0,
        "rows": [],
    }
    for example in examples:
        question_col = args.question_col or (
            "problem" if "problem" in example else "question"
        )
        question = example[question_col]
        raw_answer = example.get(args.answer_col, "") or ""
        match_mode = (
            reference_mode(raw_answer)
            if args.answer_mode == "auto"
            else args.answer_mode
        )
        if args.endpoint:
            predicted, usage = remote_completion(
                args.endpoint, args.served_model, question, args.max_new,
                args.api_key, control=mode, control_style=args.control_style,
            )
        else:
            candidate = Path(__file__).resolve().parent / args.model
            model_path = str(candidate) if candidate.exists() else args.model
            adapter_path = None
            if args.adapter:
                candidate = Path(__file__).resolve().parent / args.adapter
                adapter_path = str(candidate) if candidate.exists() else args.adapter
            predicted, usage = local_completion(
                model_path, adapter_path, question, args.max_new,
                control=mode, control_style=args.control_style,
            )
        ok = matches(predicted, raw_answer, match_mode, strict=args.strict)
        answer = formats.parse_completion(predicted)["answer"] or predicted.strip()
        stats["n"] += 1
        stats["ok"] += int(ok)
        stats["prompt_tokens"] += usage["prompt_tokens"]
        stats["completion_tokens"] += usage["completion_tokens"]
        stats["elapsed"] += usage["elapsed"]
        stats["rows"].append(
            {
                "expected": raw_answer[:80],
                "predicted": answer[:80],
                "mode": match_mode,
                "ok": ok,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "elapsed": round(usage["elapsed"], 4),
            }
        )
        print(
            f"[{mode:9}] {'OK ' if ok else 'XX '} [{match_mode:6}] "
            f"expected={stats['rows'][-1]['expected']!r} predicted={stats['rows'][-1]['predicted']!r}"
        )
    n = stats["n"] or 1
    stats["accuracy"] = round(stats["ok"] / n, 4)
    stats["avg_output_tokens"] = round(stats["completion_tokens"] / n, 2)
    stats["avg_prompt_tokens"] = round(stats["prompt_tokens"] / n, 2)
    stats["tokens_per_sec"] = (
        round(stats["completion_tokens"] / stats["elapsed"], 2)
        if stats["elapsed"] > 0 else None
    )
    cost_per_1k = (
        1000.0
        * (stats["prompt_tokens"] * args.price_per_1m_input
           + stats["completion_tokens"] * args.price_per_1m_output)
        / 1e6
        / n
    )
    stats["cost_per_1k_queries_usd"] = round(cost_per_1k, 6)
    return stats


def print_table(modes):
    print("\n== hybrid control eval ==")
    print(f"{'mode':9} {'acc':6} {'ok/n':8} {'tok/q':>7} {'tok/s':>8} {'$/1kq':>9}")
    for mode, s in modes.items():
        print(
            f"{mode:9} {s['accuracy']:.4f} "
            f"{s['ok']}/{s['n']:<6} "
            f"{s['avg_output_tokens']:>6.1f} "
            f"{s['tokens_per_sec'] if s['tokens_per_sec'] is not None else 0.0:>8.1f} "
            f"{s['cost_per_1k_queries_usd']:>9.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--served-model", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--mode", choices=["both", "think", "no_think"], default="both")
    parser.add_argument(
        "--control-style",
        choices=["system", "token"],
        default="system",
        help="must match how train_sft.py was run and how the checkpoint is served",
    )
    parser.add_argument(
        "--price-per-1m-input", type=float, default=0.15,
        help="USD per 1M prompt tokens (billed cost model for $/1k queries)",
    )
    parser.add_argument(
        "--price-per-1m-output", type=float, default=0.25,
        help="USD per 1M completion tokens (billed cost model for $/1k queries)",
    )
    parser.add_argument("--answer-mode", choices=["auto", "number", "string"], default="auto")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--question-col", default=None)
    parser.add_argument("--answer-col", default="answer")
    parser.add_argument("--output", default="data/eval_results.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    tag = args.served_model or args.model or "unknown"

    if args.config_name:
        ds = load_dataset(args.dataset, args.config_name, split=args.split)
    else:
        try:
            ds = load_dataset(args.dataset, split=args.split)
        except ValueError as e:
            if "Config name is missing" not in str(e):
                raise
            from datasets import get_dataset_config_names

            name = get_dataset_config_names(args.dataset)[0]
            ds = load_dataset(args.dataset, name, split=args.split)
    ds = ds.select(range(min(args.limit, len(ds))))
    examples = list(ds)

    modes_to_run = (
        [formats.CONTROL_THINK, formats.CONTROL_NO_THINK]
        if args.mode == "both"
        else [
            formats.CONTROL_THINK if args.mode == "think" else formats.CONTROL_NO_THINK
        ]
    )
    modes = {}
    for mode in modes_to_run:
        if args.mode != "both":
            print(f"== {mode} mode ==")
        modes[mode] = run_mode(mode, examples, args)

    primary = formats.CONTROL_THINK if formats.CONTROL_THINK in modes else modes_to_run[0]
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": tag,
                "dataset": args.dataset,
                "n": modes[primary]["n"],
                "control_style": args.control_style,
                "accuracy": modes[primary]["accuracy"],
                "price_per_1m": {
                    "input": args.price_per_1m_input,
                    "output": args.price_per_1m_output,
                },
                "modes": {m: s for m, s in modes.items()},
            },
            indent=2,
        )
    )
    print_table(modes)
    summary = "  ".join(
        f"{m}: acc={s['accuracy']:.4f} ({s['ok']}/{s['n']}) "
        f"tok/q={s['avg_output_tokens']} tok/s={s['tokens_per_sec']} "
        f"$/1k={s['cost_per_1k_queries_usd']}"
        for m, s in modes.items()
    )
    print(f"{tag}: {summary} -> {out}")


if __name__ == "__main__":
    main()
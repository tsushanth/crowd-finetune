import argparse
import json
from pathlib import Path

import httpx
from datasets import load_dataset

from . import formats

_model = None
_tokenizer = None


def local_completion(model_name, adapter, question, max_new):
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
    messages = [{"role": "user", "content": question}]
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    out = _model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False
    )
    return _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def remote_completion(endpoint, model, question, max_new, api_key=None):
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "max_tokens": max_new,
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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

    rows = []
    for example in ds:
        question_col = args.question_col or (
            "problem" if "problem" in example else "question"
        )
        question = example[question_col]
        raw_answer = example.get(args.answer_col, "") or ""
        mode = (
            reference_mode(raw_answer)
            if args.answer_mode == "auto"
            else args.answer_mode
        )
        if args.endpoint:
            predicted = remote_completion(
                args.endpoint, args.served_model, question, args.max_new, args.api_key
            )
        else:
            candidate = root / args.model
            model_path = str(candidate) if candidate.exists() else args.model
            adapter_path = None
            if args.adapter:
                candidate = root / args.adapter
                adapter_path = str(candidate) if candidate.exists() else args.adapter
            predicted = local_completion(
                model_path, adapter_path, question, args.max_new
            )
        ok = matches(predicted, raw_answer, mode, strict=args.strict)
        answer = formats.parse_completion(predicted)["answer"] or predicted.strip()
        rows.append(
            {"expected": raw_answer[:80], "predicted": answer[:80], "mode": mode, "ok": ok}
        )
        print(f"{'OK ' if ok else 'XX '} [{mode:6}] expected={rows[-1]['expected']!r} predicted={rows[-1]['predicted']!r}")

    accuracy = sum(1 for r in rows if r["ok"]) / len(rows) if rows else 0.0
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": tag,
                "dataset": args.dataset,
                "n": len(rows),
                "accuracy": round(accuracy, 4),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(
        f"{tag}: accuracy={accuracy:.4f} ({sum(1 for r in rows if r['ok'])}/{len(rows)}) -> {out}"
    )


if __name__ == "__main__":
    main()
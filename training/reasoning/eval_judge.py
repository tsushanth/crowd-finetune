import argparse
import json
from pathlib import Path

import httpx
from datasets import load_dataset

from . import formats, rewards

_model = None
_tokenizer = None


def local_completion(model_name, adapter, question, max_new, compute_ppl=False):
    global _model, _tokenizer
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _model is None:
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
    out = _model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    completion = _tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    meta = {"len_tokens": None, "ppl": None}
    completion_tokens = int(out[0].numel() - inputs["input_ids"].shape[1])
    meta["len_tokens"] = completion_tokens
    if compute_ppl and completion_tokens > 0:
        full = _tokenizer(prompt + completion, return_tensors="pt").to(_model.device)
        prompt_len = inputs["input_ids"].shape[1]
        ids = full["input_ids"][0]
        comp_len = len(ids) - prompt_len
        if comp_len > 0:
            with torch.no_grad():
                logits = _model(**full).logits[0]
            log_probs = logits[:-1].log_softmax(dim=-1)[prompt_len - 1 :]
            nll = -log_probs.gather(1, ids[prompt_len:].unsqueeze(1)).squeeze(1)
            meta["ppl"] = float(torch.exp(nll.mean()))
    return completion, meta


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
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, {
        "len_tokens": usage.get("completion_tokens"),
        "ppl": None,
    }


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


def rm_scores(prompt: str, completion: str, reference: str) -> dict:
    exact = rewards.exact_match_reward([prompt], [completion], answer=[reference])[0]
    fmt = rewards.format_reward([prompt], [completion])[0]
    length = rewards.length_reward([prompt], [completion])[0]
    return {"exact": exact, "format": fmt, "length": length}


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
    parser.add_argument("--ppl", action="store_true", help="compute local perplexity")
    parser.add_argument("--no-rm", action="store_true", help="skip reward-model scores")
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
            predicted, meta = remote_completion(
                args.endpoint, args.served_model, question, args.max_new, args.api_key
            )
        else:
            candidate = root / args.model
            model_path = str(candidate) if candidate.exists() else args.model
            adapter_path = None
            if args.adapter:
                candidate = root / args.adapter
                adapter_path = str(candidate) if candidate.exists() else args.adapter
            predicted, meta = local_completion(
                model_path, adapter_path, question, args.max_new, compute_ppl=args.ppl
            )
        ok = matches(predicted, raw_answer, mode, strict=args.strict)
        answer = formats.parse_completion(predicted)["answer"] or predicted.strip()
        row = {
            "expected": raw_answer[:80],
            "predicted": answer[:80],
            "mode": mode,
            "ok": ok,
            "len_chars": len(predicted),
            "len_tokens": meta["len_tokens"],
            "ppl": meta["ppl"],
        }
        if not args.no_rm:
            row["rm"] = rm_scores("", predicted, raw_answer)
        rows.append(row)
        print(
            f"{'OK ' if ok else 'XX '} [{mode:6}] expected={row['expected']!r} predicted={row['predicted']!r}"
        )

    accuracy = sum(1 for r in rows if r["ok"]) / len(rows) if rows else 0.0
    metrics = {}
    if rows:
        metrics["avg_len_chars"] = round(
            sum(r["len_chars"] for r in rows) / len(rows), 1
        )
        tokens = [r["len_tokens"] for r in rows if r["len_tokens"] is not None]
        metrics["avg_len_tokens"] = (
            round(sum(tokens) / len(tokens), 1) if tokens else None
        )
        ppls = [r["ppl"] for r in rows if r["ppl"] is not None]
        metrics["ppl"] = round(sum(ppls) / len(ppls), 4) if ppls else None
        rm_rows = [r for r in rows if "rm" in r]
        if rm_rows:
            for key in ("exact", "format", "length"):
                vals = [r["rm"][key] for r in rm_rows]
                metrics[f"rm_{key}_avg"] = round(sum(vals) / len(vals), 4)
        mode_acc = {}
        for m in sorted({r["mode"] for r in rows}):
            sub = [r for r in rows if r["mode"] == m]
            mode_acc[m] = {"n": len(sub), "accuracy": round(sum(1 for r in sub if r["ok"]) / len(sub), 4)}
        metrics["mode_accuracy"] = mode_acc

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": tag,
                "dataset": args.dataset,
                "n": len(rows),
                "accuracy": round(accuracy, 4),
                "metrics": metrics,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(
        f"{tag}: accuracy={accuracy:.4f} ({sum(1 for r in rows if r['ok'])}/{len(rows)}) "
        f"-> {out} metrics={metrics}"
    )


if __name__ == "__main__":
    main()
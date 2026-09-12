import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402

CHAT_TMPL = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.BASE_DIR / "data" / "crowd_sft.jsonl"))
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default=str(config.BASE_DIR / "outputs" / "crowd-lora"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if not rows:
        print("no training rows in", args.data)
        return

    import torch
    from peft import LoraConfig, get_peft_model
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
    model.config.use_cache = False
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    peft = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft)
    model.print_trainable_parameters()

    texts = [
        CHAT_TMPL.format(instruction=r.get("input", "") or r["instruction"])
        + r["output"]
        for r in rows
    ]
    enc = tokenizer(
        texts, truncation=True, padding=True, max_length=args.max_len, return_tensors="pt"
    )
    labels = enc["input_ids"].clone()
    for i, n in enumerate(enc["attention_mask"].sum(dim=1).tolist()):
        labels[i, : max(0, n - len(tokenizer.encode(rows[i].get("input", "") or rows[i]["instruction"])))] = -100

    train_size = int(0.9 * len(enc["input_ids"]))
    print(f"rows={len(texts)} train_chars={sum(len(t) for t in texts)} device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    steps = opt_steps = 0
    for epoch in range(int(args.epochs)):
        idxs = torch.randperm(train_size)
        for b in range(0, train_size, args.batch):
            batch_ids = idxs[b : b + args.batch]
            inp = {k: v[batch_ids].to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            lab = labels[batch_ids].to(device)
            out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], labels=lab)
            loss = out.loss / args.grad_accum
            loss.backward()
            steps += 1
            if steps % args.grad_accum == 0 or b + args.batch >= train_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                opt_steps += 1
                print(f"epoch {epoch + 1} opt_step {opt_steps} loss~{loss.item() * args.grad_accum:.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("adapter saved to", args.out)


if __name__ == "__main__":
    main()
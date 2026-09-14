import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from . import formats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/code_sft.jsonl")
    parser.add_argument("--output", default="outputs/code-sft")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu"], default="auto",
        help="cpu forces use_cpu (macOS MPS + pin_memory segfaults in the "
        "weight-load path; pass --device cpu on this Mac); auto uses CUDA "
        "when available",
    )
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    root = Path(__file__).resolve().parent
    data_path = root / args.data if (root / args.data).exists() else args.data
    output_dir = root / args.output

    rows = [
        json.loads(line)
        for line in Path(data_path).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("no training rows; run distill.py first")
    print(f"{len(rows)} training rows from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    def build_text(example):
        messages = [
            {"role": "system", "content": example.get("system") or formats.CODE_SYSTEM},
            {"role": "user", "content": example["question"]},
            {
                "role": "assistant",
                "content": formats.build_completion(
                    example.get("reasoning", ""), example["answer"]
                ),
            },
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    ds = Dataset.from_list(rows).map(build_text)
    tokenizer.pad_token = tokenizer.eos_token

    report_to = "none"
    if args.wandb:
        import wandb
        wandb.init(
            project="crowdcheck-coderepair",
            name=f"sft-{Path(args.base).name}-seed{args.seed}",
            config=vars(args),
        )
        report_to = "wandb"

    sft_config = SFTConfig(
        use_cpu=args.device == "cpu",
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        max_length=args.seq_length,
        truncation_mode="keep_start",
        dataset_text_field="text",
        packing=False,
        loss_type="nll",
        logging_steps=10,
        save_steps=200,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        report_to=report_to,
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
    )

    trainer = SFTTrainer(
        model=args.base,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"saved adapter -> {output_dir}")


if __name__ == "__main__":
    main()
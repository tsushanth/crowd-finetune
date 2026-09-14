import argparse
import json
import random
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default="data/dpo_pairs.jsonl")
    parser.add_argument("--output", default="outputs/reasoning-dpo")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    root = Path(__file__).resolve().parent
    data_path = root / args.data
    output_dir = root / args.output

    rows = [
        json.loads(line)
        for line in data_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("no dpo pairs; run collect_rollouts.py then make_dpo.py")
    print(f"{len(rows)} DPO pairs from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    report_to = "none"
    if args.wandb:
        import wandb

        wandb.init(
            project="crowdcheck-reasoning",
            name=f"dpo-{Path(args.base).name}-seed{args.seed}",
            config=vars(args),
        )
        report_to = "wandb"

    ds = Dataset.from_list(rows)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        beta=args.beta,
        bf16=torch.cuda.is_available(),
        max_length=args.seq_length,
        logging_steps=1,
        save_strategy="epoch",
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

    model = args.base
    if args.device:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map=args.device
        )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"saved adapter -> {output_dir}")


if __name__ == "__main__":
    main()
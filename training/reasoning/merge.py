import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output = root / args.output

    def resolve(p):
        candidate = root / p
        if candidate.exists():
            return str(candidate)
        return p

    base = resolve(args.base)
    adapter = resolve(args.adapter)

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(output)
    AutoTokenizer.from_pretrained(base).save_pretrained(output)
    print(f"merged -> {output}")


if __name__ == "__main__":
    main()
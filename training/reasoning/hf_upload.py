"""Upload trained artifacts to PRIVATE Hugging Face model repos. Token from the HF_TOKEN environment variable.

  HF_TOKEN=... python -m training.reasoning.hf_upload --prefix prm-track
Repos (under the token's user): <prefix>-prm, <prefix>-sft-adapter, <prefix>-grpo-outcome,
<prefix>-grpo-prm-outcome, <prefix>-grpo-prm. Missing local folders are skipped.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent


def latest_ckpt(d):
    c = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return c[-1] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="prm-track")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    plan = {
        f"{args.prefix}-prm": (ROOT / "outputs/prm_real", "Step-level process reward model (Qwen2.5-0.5B + linear head). Load with training/reasoning/prm.py (PRM.load)."),
        f"{args.prefix}-sft-adapter": (ROOT / "outputs/reasoning-sft", "LoRA adapter: Qwen2.5-3B-Instruct SFT on 742 GSM8K teacher traces."),
        f"{args.prefix}-grpo-outcome": (latest_ckpt(ROOT / "outputs/grpo_outcome") if (ROOT / "outputs/grpo_outcome").exists() else None, "GRPO LoRA adapter on the SFT model, exact-match reward (control)."),
        f"{args.prefix}-grpo-prm-outcome": (latest_ckpt(ROOT / "outputs/grpo_prm_outcome") if (ROOT / "outputs/grpo_prm_outcome").exists() else None, "GRPO LoRA adapter on the SFT model, PRM + exact-match reward."),
        f"{args.prefix}-grpo-prm": (latest_ckpt(ROOT / "outputs/grpo_prm") if (ROOT / "outputs/grpo_prm").exists() else None, "GRPO LoRA adapter on the SFT model, PRM-only reward."),
    }
    for name, (path, blurb) in plan.items():
        if path is None or not Path(path).exists():
            print("skip", name)
            continue
        repo = f"{user}/{name}"
        api.create_repo(repo, private=not args.public, exist_ok=True)
        # Always overwrite: trainer-generated cards (SFTTrainer/GRPOTrainer) can set a `base_model:`
        # metadata field to a local checkout path, which the Hub rejects as invalid YAML.
        (Path(path) / "README.md").write_text(
            f"---\nlicense: apache-2.0\n---\n# {name}\n\n{blurb}\n\nPart of the PRM-track experiments in "
            f"github.com/tsushanth/crowd-finetune (branch reasoning/prm, docs/prm-reports).\n")
        api.upload_folder(folder_path=str(path), repo_id=repo, ignore_patterns=["optimizer.pt", "*.pyc", "rng_state*", "scheduler.pt", "checkpoint-*/**", "checkpoint-*"])
        print("uploaded", repo, "(private)" if not args.public else "(public)")


if __name__ == "__main__":
    main()

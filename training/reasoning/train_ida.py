import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
LOG_PATH = ROOT / "data" / "ida_rounds.json"
SEED_TEACHER = os.environ.get("TEACHER_MODEL", "deepseek/deepseek-r1-0528")


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def count_rows(path):
    p = ROOT / path
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip())


def load_accuracy(path):
    return json.loads((ROOT / path).read_text())["accuracy"]


def merge_traces(round_idx, out_name):
    seen = {}
    for r in range(round_idx + 1):
        p = ROOT / f"data/ida_round{r}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen[row["question"]] = row
    rows = list(seen.values())
    random.Random(42).shuffle(rows)
    out = ROOT / out_name
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def latest_checkpoint(dir_path):
    ckpts = sorted((ROOT / dir_path).glob("checkpoint-*"))
    if not ckpts:
        raise RuntimeError(f"no GRPO checkpoint under {dir_path}")
    return str(ckpts[-1])


def eval_checkpoint(model, dataset, limit, out):
    cmd = [
        sys.executable, "-m", "training.reasoning.eval_judge",
        "--model", model, "--limit", str(limit), "--output", out,
    ]
    if dataset == "math":
        cmd += ["--dataset", "HuggingFaceH4/MATH-500", "--strict"]
    run(cmd)
    return load_accuracy(out)


def write_log(log):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2) + "\n")
    print(f"ida_rounds log updated -> {LOG_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="IDA loop: distill -> SFT -> GRPO -> eval per round, "
        "self-distilling from the previous round's GRPO-merged champion; "
        "stops on no-improvement or --max-rounds. Logs every round to "
        "training/reasoning/data/ida_rounds.json."
    )
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--teacher", default=SEED_TEACHER)
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="total rounds incl. baseline round 0")
    parser.add_argument("--distill-limit", type=int, default=250)
    parser.add_argument("--distill-workers", type=int, default=6)
    parser.add_argument("--grpo-limit", type=int,
                        default=int(os.environ.get("GRPO_LIMIT", 400)))
    parser.add_argument("--grpo-max", type=int,
                        default=int(os.environ.get("GRPO_MAX_COMPLETION", 1024)))
    parser.add_argument("--grpo-gens", type=int, default=8)
    parser.add_argument("--grpo-batch", type=int, default=4)
    parser.add_argument("--eval-limit", type=int,
                        default=int(os.environ.get("MATH_LIMIT", 100)))
    args = parser.parse_args()

    log = {
        "meta": {
            "base": args.base,
            "seed_teacher": args.teacher,
            "max_rounds": args.max_rounds,
            "distill_limit": args.distill_limit,
            "grpo_limit": args.grpo_limit,
            "grpo_max_completion": args.grpo_max,
            "grpo_gens": args.grpo_gens,
            "grpo_batch": args.grpo_batch,
            "eval_limit": args.eval_limit,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        "rounds": [],
    }
    if LOG_PATH.exists():
        backup = LOG_PATH.with_name(
            LOG_PATH.stem + "." + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
        )
        backup.write_text(LOG_PATH.read_text())
        print(f"existing {LOG_PATH.name} preserved -> {backup}")

    for r in range(args.max_rounds):
        teacher = args.teacher if r == 0 else (
            f"outputs/ida_round{r - 1}/reasoning-grpo-merged"
        )
        distill_out = f"data/ida_round{r}.jsonl"
        sft_out = f"data/ida_sft_round{r}.jsonl"
        sft_dir = f"outputs/ida_round{r}/reasoning-sft"
        sft_merged = f"outputs/ida_round{r}/reasoning-sft-merged"
        grpo_dir = f"outputs/ida_round{r}/reasoning-grpo"
        grpo_merged = f"outputs/ida_round{r}/reasoning-grpo-merged"

        run([sys.executable, "-m", "training.reasoning.distill",
             "--limit", str(args.distill_limit),
             "--workers", str(args.distill_workers),
             "--out", distill_out,
             "--teacher-model", teacher])
        distill_rows = count_rows(distill_out)
        sft_rows = merge_traces(r, sft_out)
        print(f"round {r}: distill kept {distill_rows}, cumulative SFT {sft_rows} rows")
        if sft_rows == 0:
            log["stop"] = {"reason": "round produced no usable SFT traces", "round": r}
            write_log(log)
            break

        run([sys.executable, "-m", "training.reasoning.train_sft",
             "--base", args.base, "--data", sft_out, "--output", sft_dir])
        run([sys.executable, "-m", "training.reasoning.merge",
             "--base", args.base, "--adapter", sft_dir, "--output", sft_merged])
        run([sys.executable, "-m", "training.reasoning.train_grpo",
             "--base", sft_merged, "--limit", str(args.grpo_limit),
             "--gens", str(args.grpo_gens), "--batch", str(args.grpo_batch),
             "--max-completion", str(args.grpo_max), "--output", grpo_dir])
        grpo_ckpt = latest_checkpoint(grpo_dir)
        run([sys.executable, "-m", "training.reasoning.merge",
             "--base", sft_merged, "--adapter", grpo_ckpt, "--output", grpo_merged])

        evals = {}
        if r == 0:
            evals["base"] = {
                "gsm8k": eval_checkpoint(
                    args.base, "gsm8k", args.eval_limit,
                    f"data/ida_round{r}_base_gsm8k.json"),
                "math500": eval_checkpoint(
                    args.base, "math", args.eval_limit,
                    f"data/ida_round{r}_base_math.json"),
            }
        evals["reasoning-sft-merged"] = {
            "gsm8k": eval_checkpoint(
                sft_merged, "gsm8k", args.eval_limit,
                f"data/ida_round{r}_sft_gsm8k.json"),
            "math500": eval_checkpoint(
                sft_merged, "math", args.eval_limit,
                f"data/ida_round{r}_sft_math.json"),
        }
        evals["reasoning-grpo-merged"] = {
            "gsm8k": eval_checkpoint(
                grpo_merged, "gsm8k", args.eval_limit,
                f"data/ida_round{r}_grpo_gsm8k.json"),
            "math500": eval_checkpoint(
                grpo_merged, "math", args.eval_limit,
                f"data/ida_round{r}_grpo_math.json"),
        }

        entry = {
            "round": r,
            "teacher": teacher,
            "distill_data": distill_out,
            "distill_rows": distill_rows,
            "sft_data": sft_out,
            "sft_rows": sft_rows,
            "grpo_checkpoint": grpo_ckpt,
            "grpo_merged": grpo_merged,
            "eval": evals,
        }
        log["rounds"].append(entry)
        write_log(log)

        cur = evals["reasoning-grpo-merged"]["gsm8k"]
        if r >= 1:
            prev = log["rounds"][-2]["eval"]["reasoning-grpo-merged"]["gsm8k"]
            if cur <= prev:
                log["stop"] = {
                    "reason": f"round {r} GSM8K {cur:.4f} did not improve "
                              f"over round {r - 1} {prev:.4f}",
                    "round": r, "prev_gsm8k": prev, "current_gsm8k": cur,
                }
                write_log(log)
                print("STOP:", log["stop"]["reason"])
                break
        print(f"round {r} champion (GRPO-merged) GSM8K={cur:.4f}")
    else:
        log["stop"] = {"reason": f"reached max_rounds={args.max_rounds}"}
        write_log(log)


if __name__ == "__main__":
    main()
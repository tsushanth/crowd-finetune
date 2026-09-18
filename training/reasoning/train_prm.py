import argparse
import json
import random
import time
from pathlib import Path

import torch

from . import prm as P


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate(model, examples, batch_size=16):
    scores = model.score_steps([e["question"] for e in examples], [e["steps"] for e in examples], batch_size)
    step_ok = step_n = 0
    chain = {"positive": [], "negative": []}
    by_kind = {}
    for e, s in zip(examples, scores):
        for lab, p in zip(e["labels"], s):
            if lab in (0, 1):
                step_n += 1
                step_ok += int((p > 0.5) == bool(lab))
        c = P.aggregate(s, "min")
        chain["positive" if e["kind"] == "positive" else "negative"].append(c)
        if e["kind"] != "positive":
            by_kind.setdefault(e["kind"], []).append(c)
    pos = chain["positive"]
    res = {
        "step_acc": step_ok / max(step_n, 1),
        "chain_auroc_min": auroc(pos, chain["negative"]),
        "n_pos_chains": len(pos), "n_neg_chains": len(chain["negative"]),
        "auroc_by_corruption": {k: auroc(pos, v) for k, v in by_kind.items()},
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", nargs="+", default=[
        "data/reasoning_sft.jsonl", "data/reasoning_sft_more.jsonl",
        "data/reasoning_sft_more2.jsonl", "data/reasoning_sft_more3.jsonl"])
    ap.add_argument("--data-root", default=None, help="dir the --data paths are relative to (default: this package)")
    ap.add_argument("--extra", nargs="*", default=[], help="prebuilt chains jsonl (gen_real_negatives) added to TRAIN only")
    ap.add_argument("--output", default="outputs/prm")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--train-layers", type=int, default=0, help="train only the top N layers (0 = all)")
    ap.add_argument("--negs", type=int, default=2, help="corrupted variants per question")
    ap.add_argument("--limit", type=int, default=None, help="cap unique questions (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else Path(__file__).resolve().parent
    rows = P.dedupe_rows([root / p for p in args.data if (root / p).exists()])
    if args.limit:
        rows = rows[: args.limit]
    train_rows, val_rows = P.split_by_question(rows, 0.15, args.seed)
    train = P.build_examples(train_rows, args.seed, args.negs)
    val = P.build_examples(val_rows, args.seed + 1, args.negs)
    for p in args.extra:
        extra = [json.loads(l) for l in (root / p).read_text().splitlines() if l.strip()]
        train += extra
        print(f"+{len(extra)} real chains from {p} "
              f"({sum(e['kind'] == 'real_neg' for e in extra)} neg)", flush=True)
    print(f"{len(rows)} unique questions -> train {len(train_rows)}q/{len(train)} chains, "
          f"val {len(val_rows)}q/{len(val)} chains (split by question)", flush=True)

    dtype = torch.float32  # fp32 master weights; bf16 AdamW updates at lr 5e-5 underflow
    torch.backends.cuda.matmul.allow_tf32 = True
    model = P.PRM.build(args.base, dtype)
    model.backbone.to(args.device)
    model.head.to(args.device)
    layers = model.backbone.layers
    for p in model.backbone.parameters():
        p.requires_grad = args.train_layers == 0
    if args.train_layers:
        for l in layers[-args.train_layers:]:
            for p in l.parameters():
                p.requires_grad = True
        for p in model.backbone.norm.parameters():
            p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)

    random.seed(args.seed)
    steps_total = int(len(train) / args.batch * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1 - s / max(steps_total, 1)))
    enc = [(P.encode(model.tokenizer, e["question"], e["steps"]), e["labels"]) for e in train]

    model.backbone.train()
    t0 = time.time()
    step = 0
    order = []
    while step < steps_total:
        if not order:
            order = list(range(len(enc)))
            random.shuffle(order)
        idx, order = order[: args.batch], order[args.batch:]
        ids = [enc[i][0][0] for i in idx]
        pos = [enc[i][0][1] for i in idx]
        lg = model.logits(ids, pos)
        loss_num, loss_den = 0.0, 0
        for l, i in zip(lg, idx):
            lab = torch.tensor(enc[i][1][: len(l)], device=l.device)
            m = lab >= 0
            if m.any():
                loss_num = loss_num + torch.nn.functional.binary_cross_entropy_with_logits(
                    l[m], lab[m].float(), reduction="sum")
                loss_den += int(m.sum())
        loss = loss_num / max(loss_den, 1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 10 == 0 or step == steps_total:
            print(f"step {step}/{steps_total} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)

    model.backbone.eval()
    metrics = {"args": vars(args), "n_train_q": len(train_rows), "n_val_q": len(val_rows),
               "train_seconds": round(time.time() - t0), "val": evaluate(model, val)}
    print(json.dumps(metrics["val"], indent=2))
    out = Path(__file__).resolve().parent / args.output
    out.mkdir(parents=True, exist_ok=True)
    model.save(out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved PRM -> {out}")


if __name__ == "__main__":
    main()

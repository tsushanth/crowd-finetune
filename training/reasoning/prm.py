"""Step-level process reward model (PRM).

A causal-LM backbone reads `question + step1 <sep> step2 <sep> ...` and a linear
head reads the hidden state at each <sep>: P(step is correct | everything before
it). Causal attention means step k's score cannot see later steps.
"""
import json
import random
import re
from pathlib import Path

SEP_TOKEN = "<|file_sep|>"  # a trained-but-unused special token in the Qwen2.5 vocab
HEAD_FILE = "prm_head.pt"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z$\d(])")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def split_steps(reasoning: str) -> list[str]:
    """Newlines first, then sentence boundaries. Same splitter for train and RL."""
    steps = []
    for line in reasoning.strip().splitlines():
        for s in _SENT_SPLIT.split(line.strip()):
            s = s.strip()
            if len(s) >= 3:
                steps.append(s)
    return steps


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def _num_val(tok: str) -> float:
    return float(tok.replace(",", ""))


def _perturb_number(step: str, rng: random.Random, pool: list[str]):
    """Change the step's result number (last number) to a wrong-but-plausible value."""
    ms = list(_NUM.finditer(step))
    if not ms:
        return None
    m = ms[-1]
    v = _num_val(m.group())
    kind = rng.choice(["off_by_small", "scale", "digit"])
    if kind == "off_by_small":
        new = v + rng.choice([-2, -1, 1, 2, 10, -10])
    elif kind == "scale":
        new = v * rng.choice([2, 0.5, 10])
    else:
        new = v + rng.choice([-1, 1]) * (10 ** rng.randint(0, max(0, len(str(int(abs(v)))) - 1)))
    if new == v or new < 0 <= v:
        return None
    return step[: m.start()] + _fmt(new) + step[m.end():], "perturb_result"


def _swap_operand(step: str, rng: random.Random, pool: list[str]):
    """Replace a non-final number in the step with a different quantity from the problem."""
    ms = list(_NUM.finditer(step))
    if len(ms) < 2:
        return None
    m = rng.choice(ms[:-1])
    cands = [p for p in pool if _num_val(p) != _num_val(m.group())]
    if not cands:
        return None
    return step[: m.start()] + rng.choice(cands) + step[m.end():], "swap_operand"


def _flip_op(step: str, rng: random.Random, pool: list[str]):
    swaps = [(" + ", " - "), (" - ", " + "), (" * ", " / "), (" / ", " * "),
             (" × ", " ÷ "), (" plus ", " minus ")]
    present = [(a, b) for a, b in swaps if a in step]
    if not present:
        return None
    a, b = rng.choice(present)
    return step.replace(a, b, 1), "flip_op"


_CORRUPTORS = [_perturb_number, _swap_operand, _flip_op]


def corrupt(question: str, steps: list[str], rng: random.Random):
    """Corrupt one step. Returns (new_steps, index, kind) or None if nothing applies.

    Everything after `index` is unreliable and must be masked in the labels
    (the usual PRM convention: label up to and including the first error).
    """
    pool = _NUM.findall(question)
    order = list(range(len(steps)))
    rng.shuffle(order)
    for i in order:
        fns = _CORRUPTORS[:]
        rng.shuffle(fns)
        for fn in fns:
            out = fn(steps[i], rng, pool)
            if out and out[0] != steps[i]:
                return steps[:i] + [out[0]] + steps[i + 1:], i, out[1]
    return None


def build_examples(rows, seed=0, negatives_per_question=2):
    """rows: dicts with question/reasoning. -> list of {question, steps, labels}.

    labels: 1 correct, 0 first error, -100 masked (after the first error).
    """
    rng = random.Random(seed)
    out = []
    for r in rows:
        steps = split_steps(r["reasoning"])
        if not steps:
            continue
        out.append({"question": r["question"], "steps": steps,
                    "labels": [1] * len(steps), "kind": "positive"})
        seen = set()
        for _ in range(negatives_per_question * 3):
            if len(seen) >= negatives_per_question:
                break
            c = corrupt(r["question"], steps, rng)
            if not c or (c[1], c[2]) in seen:
                continue
            seen.add((c[1], c[2]))
            new_steps, i, kind = c
            labels = [1] * i + [0] + [-100] * (len(steps) - i - 1)
            out.append({"question": r["question"], "steps": new_steps,
                        "labels": labels, "kind": kind})
    return out


def dedupe_rows(paths):
    seen = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen.setdefault(row["question"], row)
    return list(seen.values())


def split_by_question(rows, val_frac=0.15, seed=0):
    rows = sorted(rows, key=lambda r: r["question"])
    random.Random(seed).shuffle(rows)
    k = int(len(rows) * val_frac)
    return rows[k:], rows[:k]


def encode(tokenizer, question, steps, max_len=512):
    """-> (input_ids, sep_positions). Question is clipped so every step fits."""
    sep = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    ids = tokenizer("Problem: " + question + "\nSolution:", add_special_tokens=False)["input_ids"]
    ids = ids[:max_len // 2]
    pos = []
    for s in steps:
        ids = ids + tokenizer("\n" + s, add_special_tokens=False)["input_ids"] + [sep]
        pos.append(len(ids) - 1)
    if len(ids) > max_len:  # truncate long chains; drop steps that no longer fit
        ids = ids[:max_len]
        pos = [p for p in pos if p < max_len]
    return ids, pos


class PRM:
    def __init__(self, backbone, head, tokenizer):
        self.backbone, self.head, self.tokenizer = backbone, head, tokenizer

    @staticmethod
    def build(base="Qwen/Qwen2.5-0.5B-Instruct", dtype=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(base)
        backbone = AutoModel.from_pretrained(base, torch_dtype=dtype or torch.float32)
        head = torch.nn.Linear(backbone.config.hidden_size, 1).to(backbone.dtype)
        return PRM(backbone, head, tok)

    @staticmethod
    def load(path, device="cpu", dtype=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        backbone = AutoModel.from_pretrained(path, torch_dtype=dtype or torch.float32)
        head = torch.nn.Linear(backbone.config.hidden_size, 1).to(backbone.dtype)
        head.load_state_dict(torch.load(Path(path) / HEAD_FILE, map_location="cpu"))
        m = PRM(backbone.to(device).eval(), head.to(device).eval(), tok)
        m.device = device
        return m

    def save(self, path):
        import torch

        self.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.head.state_dict(), Path(path) / HEAD_FILE)

    def logits(self, batch_ids, batch_pos):
        """batch_ids: list[list[int]]; batch_pos: list[list[int]] -> list[Tensor[n_steps]]."""
        import torch

        dev = next(self.backbone.parameters()).device
        L = max(len(x) for x in batch_ids)
        pad = self.tokenizer.pad_token_id or 0
        ids = torch.tensor([x + [pad] * (L - len(x)) for x in batch_ids], device=dev)
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in batch_ids], device=dev)
        h = self.backbone(input_ids=ids, attention_mask=att).last_hidden_state
        out = []
        for b, pos in enumerate(batch_pos):
            out.append(self.head(h[b, pos]).squeeze(-1).float() if pos else h.new_zeros(0).float())
        return out

    def score_steps(self, questions, steps_list, batch_size=16):
        """Per-step P(correct) for each (question, steps). Runs in eval/no_grad."""
        import torch

        res = [None] * len(questions)
        enc = [encode(self.tokenizer, q, s) for q, s in zip(questions, steps_list)]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
        with torch.no_grad():
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                lg = self.logits([enc[j][0] for j in idx], [enc[j][1] for j in idx])
                for j, l in zip(idx, lg):
                    res[j] = torch.sigmoid(l).tolist()
        return res


def aggregate(step_probs, mode="min"):
    """Chain score in [0,1]. Empty chain (no parseable steps) scores 0."""
    if not step_probs:
        return 0.0
    if mode == "min":
        return min(step_probs)
    if mode == "mean":
        return sum(step_probs) / len(step_probs)
    raise ValueError(mode)

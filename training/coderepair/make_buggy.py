import argparse
import ast
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import datasets, formats, sandbox
from .teacher import Teacher

sbx = sandbox.Sandbox()

BUG_TYPES = ["wrong_operator", "off_by_one", "wrong_var", "invert_cond", "drop_guard"]


class _Single(ast.NodeTransformer):
    def __init__(self, rng=None):
        super().__init__()
        self.done = False
        self.rng = rng or random


class _WrongOperator(_Single):
    _MAP = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
    }

    def _flip(self, node):
        op = type(node.op)
        if op in self._MAP:
            self.done = True
            node.op = self._MAP[op]()
            return node
        return node

    visit_BinOp = lambda self, node: self._flip(node) if not self.done else node


class _OffByOne(_Single):
    def _range(self, node):
        args = node.args
        if (
            len(args) >= 1
            and isinstance(args[0], ast.Constant)
            and isinstance(args[0].value, int)
            and args[0].value >= 2
        ):
            self.done = True
            args[0].value += self.rng.choice((-1, 1))
        return node

    def _compare(self, node):
        for i, op in enumerate(node.ops):
            if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                repl = {
                    ast.Lt: ast.LtE,
                    ast.LtE: ast.Lt,
                    ast.Gt: ast.GtE,
                    ast.GtE: ast.Gt,
                }[type(op)]
                self.done = True
                node.ops[i] = repl()
                break
        return node

    def visit_Call(self, node):
        if self.done:
            return node
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            return self._range(node)
        return node

    def visit_Compare(self, node):
        if self.done:
            return node
        return self._compare(node)


class _WrongVar(_Single):
    def visit_Module(self, node):
        if self.done:
            return node
        counts = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                counts[sub.id] = counts.get(sub.id, 0) + 1
        cands = [(n, c) for n, c in counts.items() if c >= 2 and c < 20]
        if len(cands) < 2:
            return node
        cands.sort(key=lambda t: -t[1])
        (x, _), (y, _) = cands[:2]
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == x:
                sub.id = y
        self.done = True
        return node


class _InvertCond(_Single):
    _MAP = {
        ast.Lt: ast.Gt,
        ast.Gt: ast.Lt,
        ast.LtE: ast.GtE,
        ast.GtE: ast.LtE,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
    }

    def _flip(self, node):
        for i, op in enumerate(node.ops):
            if type(op) in self._MAP:
                self.done = True
                node.ops[i] = self._MAP[type(op)]()
                break
        return node

    def _maybe(self, node):
        if isinstance(node.test, ast.Compare):
            self._flip(node.test)
        return node

    def visit_If(self, node):
        return node if self.done else self._maybe(node)

    def visit_While(self, node):
        return node if self.done else self._maybe(node)


class _DropGuard(_Single):
    def visit_If(self, node):
        if self.done:
            return node
        for child in node.body:
            if isinstance(child, (ast.Return, ast.Break, ast.Continue)):
                self.done = True
                return []
        return node


def mutate(code, bug_type=None, seed=0):
    rng = random.Random(seed)
    tree = ast.parse(code)
    mutators = {
        "wrong_operator": _WrongOperator,
        "off_by_one": _OffByOne,
        "wrong_var": _WrongVar,
        "invert_cond": _InvertCond,
        "drop_guard": _DropGuard,
    }
    order = list(mutators)
    if bug_type:
        order = [bug_type]
        rng.shuffle(order)
    else:
        rng.shuffle(order)
    base_text = ast.unparse(tree)
    for name in order:
        mt = mutators[name](rng)
        try:
            out = mt.visit(tree)
            ast.fix_missing_locations(out)
            new = ast.unparse(out)
            ast.parse(new)
        except Exception:
            continue
        if new != base_text:
            return name, new
        tree = ast.parse(code)
    return None


def make_buggy_variant(code, problem, seed):
    mutated = mutate(code, seed=seed)
    if not mutated:
        return None
    bug_type, buggy = mutated
    result = sbx.run(buggy, problem)
    if not result.get("ok", False) and result.get("total", 0) > 0:
        return bug_type, buggy
    return None


def build_row(question, code, problem, source, bug_type, buggy):
    return {
        "question": formats.build_repair_question(question, buggy, problem["tests"]),
        "tests": list(problem["tests"]),
        "imports": list(problem.get("imports") or []),
        "entry_point": problem.get("entry_point"),
        "source": source,
        "bug_type": bug_type,
        "buggy_code": buggy,
        "answer": code,
        "system": formats.REPAIR_SYSTEM,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/code_sft.jsonl")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--out", default="data/code_repair_train.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bug-type", choices=BUG_TYPES, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    all_rows = []
    for d in args.data.split(","):
        d = d.strip()
        if not d:
            continue
        data_path = root / d if (root / d).exists() else d
        all_rows.extend([
            json.loads(line)
            for line in Path(data_path).read_text().splitlines()
            if line.strip()
        ])
    if args.limit:
        all_rows = all_rows[: args.limit]

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    def problem_of(row):
        return {
            "tests": row["tests"],
            "imports": row.get("imports") or [],
            "entry_point": row.get("entry_point"),
        }

    def work(row):
        if args.mode == "train":
            question = row["question"]
            code = row["answer"]
        else:
            question = row["question"]
            problem = problem_of(row)
            teacher = Teacher()
            prompt = datasets.render_question(row, include_tests=True)
            try:
                text = teacher.generate(prompt)
            except Exception as exc:
                print(f"teacher failed {row.get('source')}: {exc}")
                return None
            candidate = formats.extract_code(text)
            if not candidate:
                raise ValueError("no code from teacher")
            if not sbx.run(candidate, problem)["ok"]:
                return None
            code = candidate
        norm = ast.unparse(ast.parse(code))
        if norm != code and not sbx.run(norm, problem_of(row))["ok"]:
            norm = code
        variants = []
        tried = set()
        for _ in range(args.variants * 5):
            if len(variants) >= args.variants:
                break
            seed = rng.randrange(1 << 30)
            if seed in tried:
                continue
            tried.add(seed)
            variant = make_buggy_variant(norm, problem_of(row), seed=seed)
            if not variant:
                continue
            bug_type, buggy = variant
            key = (bug_type, buggy)
            if key in {(_b, _c) for _b, _c in variants}:
                continue
            variants.append((bug_type, buggy))
        if not variants:
            return None
        traces = []
        for i, (bug_type, buggy) in enumerate(variants):
            src = row.get("source", "")
            if i > 0:
                src = f"{src}#v{i}"
            traces.append(build_row(question, norm, problem_of(row), src, bug_type, buggy))
        return traces

    kept = 0
    seen_key = set()
    with out.open("w") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for traces in pool.map(work, all_rows, chunksize=1):
            if not traces:
                continue
            for trace in traces:
                key = (trace["source"], trace["bug_type"], trace["buggy_code"])
                if key in seen_key:
                    continue
                seen_key.add(key)
                kept += 1
                fh.write(json.dumps(trace) + "\n")
                fh.flush()
    print(f"kept {kept}/{len(all_rows)} buggy variants -> {out}")


if __name__ == "__main__":
    main()
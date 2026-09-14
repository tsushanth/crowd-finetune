import argparse
import json
import math
import queue as qmod
import random
import sys
import threading
import time

from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_prompts(bench: str, samples: int) -> list[list[dict]]:
    from backend import toolcall

    rows = [json.loads(l) for l in Path(bench).read_text().splitlines() if l.strip()]
    if samples:
        rows = rows[:samples]
    return [
        [
            {"role": "system", "content": toolcall.SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        for row in rows
    ]


def _usage(data: dict, key: str) -> int:
    return (data.get("usage") or {}).get(key, 0)


def worker(client: httpx.Client, model: str, queue, results, errors, clock, idx) -> None:
    while True:
        item = queue.get()
        if item is None:
            return
        row, prompt = item
        t0 = time.perf_counter()
        try:
            resp = client.post(
                "/chat/completions",
                json={"model": model, "messages": prompt, "max_tokens": clock.max_new, "temperature": 0},
            )
            dt = time.perf_counter() - t0
            data = resp.json()
            results[idx] = results.get(idx, 0) + 1
            clock.out_tokens += _usage(data, "completion_tokens")
            clock.in_tokens += _usage(data, "prompt_tokens")
            clock.last_done = time.perf_counter()
        except Exception as exc:
            errors.append(f"row {row}: {exc}")
        finally:
            clock.done += 1


class Clock:
    def __init__(self, max_new: int):
        self.max_new = max_new
        self.out_tokens = 0
        self.in_tokens = 0
        self.done = 0
        self.last_done = 0.0


def run_batch(prompts: list[list[dict]], model: str, base_url: str, workers_n: int, repeats: int, max_new: int):
    tasks = []
    for rep in range(repeats):
        for i, prompt in enumerate(prompts):
            tasks.append((f"{rep}:{i}", prompt))
    rng = random.Random(42)
    rng.shuffle(tasks)
    q = qmod.Queue()
    for t in tasks:
        q.put(t)
    for _ in range(workers_n):
        q.put(None)
    clock = Clock(max_new)
    errors = []
    t0 = time.perf_counter()
    results = {}
    threads = []
    with httpx.Client(base_url=base_url, timeout=300) as client:
        for i in range(workers_n):
            th = threading.Thread(target=worker, args=(client, model, q, results, errors, clock, i))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
    wall = time.perf_counter() - t0
    return clock, wall, len(tasks), errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default=str(BASE_DIR / "data" / "toolcall_corpus.jsonl"))
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--gpu-rate", type=float, default=0.47)
    args = ap.parse_args()

    prompts = _load_prompts(args.bench, args.samples) if args.samples else _load_prompts(args.bench, 0)
    if not prompts:
        print("bench is empty")
        return
    total_requests = len(prompts) * args.repeats
    print(f"bench rows={len(prompts)} repeats={args.repeats} concurrency={args.concurrency} requests={total_requests}")

    clock, wall, n_tasks, errors = run_batch(
        prompts, args.model, args.base_url, args.concurrency, args.repeats, args.max_new
    )
    if errors:
        print(f"{len(errors)} request errors, first: {errors[0]}")
    if clock.done == 0:
        print("no completions; check --base-url/--model")
        return

    sustained_tps = clock.out_tokens / wall
    per_q_out = clock.out_tokens / clock.done
    per_q_in = clock.in_tokens / clock.done
    per_q = per_q_in + per_q_out
    cost = args.gpu_rate / 3600.0 / sustained_tps * per_q * 1000.0 if sustained_tps > 0 else math.nan

    naive_task = prompts[0]
    t0 = time.perf_counter()
    with httpx.Client(base_url=args.base_url, timeout=300) as client:
        r = client.post(
            "/chat/completions",
            json={"model": args.model, "messages": naive_task, "max_tokens": args.max_new, "temperature": 0},
        )
    dt = time.perf_counter() - t0
    naive_tps = _usage(r.json(), "completion_tokens") / dt if dt > 0 else 0.0
    naive_cost = args.gpu_rate / 3600.0 / naive_tps * per_q * 1000.0 if naive_tps > 0 else math.nan

    print(f"\nwall={wall:.1f}s done={clock.done}/{n_tasks}")
    print(f"sustained throughput (continuous batching): {sustained_tps:.1f} out-tok/s ({clock.out_tokens}/{wall:.1f}s)")
    print(f"per query: {per_q_in:.0f} in / {per_q_out:.0f} out tokens ({per_q:.0f} total)")
    print(f"cost @${args.gpu_rate:.2f}/hr: ${cost:.5f}/1K queries")
    lift = (sustained_tps / naive_tps) if naive_tps > 0 else 0.0
    lift_cost = (naive_cost / cost) if cost > 0 else 0.0
    print(f"\nsingle-stream (naive) reference: {naive_tps:.1f} out-tok/s -> ${naive_cost:.5f}/1K q")
    print(f"batching lift: {lift:.1f}x throughput, {lift_cost:.1f}x cheaper per query")


if __name__ == "__main__":
    main()
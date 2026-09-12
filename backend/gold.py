import json

from typing import Optional

import httpx

from . import config

JUDGE_RUBRIC = (config.BASE_DIR / "prompts" / "judge.md").read_text()
_HEADERS = {
    "Authorization": f"Bearer {config.BASE_LLM_API_KEY}",
    "Content-Type": "application/json",
}
_ENDPOINT = config.BASE_LLM_URL.rstrip("/") + "/chat/completions"
_BASE_SYSTEM = "Answer the question directly and concisely, only from the provided domain source."


def chat(system: str, user: str, model: Optional[str] = None, json_mode: bool = False) -> str:
    if not config.BASE_LLM_API_KEY:
        raise RuntimeError("BASE_LLM_API_KEY is not set")
    payload = {
        "model": model or config.BASE_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = httpx.post(_ENDPOINT, headers=_HEADERS, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def base_answer(question: str) -> str:
    return chat(_BASE_SYSTEM, question).strip()


def judge(mode: str, question: str, response: str, golden: Optional[str] = None) -> dict:
    user = f"mode: {mode}\nquestion: {question}\nresponse: {response}\n"
    if golden:
        user += f"golden/reference answer: {golden}\n"
    out = chat(JUDGE_RUBRIC, user, model=config.JUDGE_MODEL, json_mode=True)
    data = json.loads(out)
    data.setdefault("verdict", "fail")
    data["score"] = float(data.get("score", 0.0))
    data["pass"] = data["score"] >= config.JUDGE_MIN_SCORE
    return data


def taint_answer(question: str, golden: str) -> str:
    sys = "You produce a single subtly-wrong answer for a QA exercise. It must look plausible and stay close in length and structure to the correct answer while changing exactly one decisive fact (a number, threshold, name, or condition). Do not add explanations."
    user = f"correct answer:\n{golden}\n\nquestion:\n{question}"
    return chat(sys, user).strip()


def load_corpus() -> list[dict]:
    path = config.CORPUS_PATH
    if not path.exists():
        fallback = config.BASE_DIR / "data" / "demo_corpus.jsonl"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError("No corpus found. Put items at data/corpus.jsonl.")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


_NICHE_TOOLCALL = "tool-call"


def _toolcall():
    from . import toolcall

    return toolcall


def is_correct(niche: str, emitted: str, item: dict) -> bool:
    if niche == _NICHE_TOOLCALL:
        return _toolcall().is_correct(emitted, item)["correct"]
    return judge("verify_gold", item["question"], emitted, item.get("golden_answer"))["pass"]


def warm(conn, limit: Optional[int] = None) -> None:
    corpus = load_corpus()
    if limit:
        corpus = corpus[:limit]
    done = {
        r["corpus_id"]
        for r in conn.execute("SELECT corpus_id FROM items").fetchall()
    }
    for row in corpus:
        if row["corpus_id"] in done:
            continue
        question = row["question"]
        niche = row.get("niche", "demo")
        is_gold = int(row.get("is_gold", 0))
        golden = row.get("golden_answer")
        payload = row.get("payload")
        tainted_reference = None
        base_verified = 0
        if niche == _NICHE_TOOLCALL:
            tc = _toolcall()
            base_response = tc.base_answer(row)
            if is_gold and payload:
                check = tc.is_correct(base_response, row)
                base_verified = int(check["correct"])
                if base_verified and tc.expects_decline(row):
                    base_verified = int(tc.judge_decline(row)["pass"])
                if base_verified:
                    tainted_reference = tc.taint_answer(row)
        else:
            base_response = base_answer(question)
            if is_gold and golden:
                check = judge("verify_gold", question, base_response, golden)
                base_verified = int(check["pass"])
                if base_verified:
                    tainted_reference = taint_answer(question, golden)
        conn.execute(
            "INSERT INTO items (corpus_id, niche, question, source, golden_answer, is_gold, base_verified, base_response, tainted_reference, payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                row["corpus_id"],
                niche,
                question,
                row.get("source", ""),
                golden,
                is_gold,
                base_verified,
                base_response,
                tainted_reference,
                payload if isinstance(payload, str) else (json.dumps(payload, ensure_ascii=False) if payload else None),
            ),
        )
    conn.commit()
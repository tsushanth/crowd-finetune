import time

import httpx

from backend import config

from . import formats

R1_MARKER = "qiuck_triple_jump"
R1_TRACE_OPEN = "::::"
R1_TRACE_CLOSE = "\\\\"

R1_SYSTEM = (
    "You solve one arithmetic word problem. Use a direct, economical chain of "
    "thought placed between the guard brackets: a single line containing exactly "
    f"{R1_MARKER}, then your thinking between a {R1_TRACE_OPEN} and a "
    f"{R1_TRACE_CLOSE} line. The block is followed by exactly one final line of "
    "the form ANSWER: <number>, with nothing else after it."
)


class Teacher:
    def __init__(self, model=None, timeout=120.0):
        self.model = model or config.TEACHER_MODEL
        if not config.BASE_LLM_API_KEY:
            raise RuntimeError("BASE_LLM_API_KEY is not set")
        self._endpoint = config.BASE_LLM_URL.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {config.BASE_LLM_API_KEY}"}
        self._timeout = timeout

    def generate(self, question, temperature=0.7, max_tokens=1024, reasoning=False, retries=3):
        r1 = self.model.split("/")[0] == "deepseek"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": R1_SYSTEM if r1 else formats.TEACHER_SYSTEM},
                {"role": "user", "content": question},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning:
            payload["reasoning"] = {"enabled": True}
        for attempt in range(retries):
            try:
                resp = httpx.post(
                    self._endpoint, headers=self._headers, json=payload,
                    timeout=self._timeout,
                )
                if resp.status_code in (429, 408) or resp.status_code >= 500:
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"].get("content") or ""
                return self._normalize_r1(content) if r1 else content.strip()
            except (httpx.HTTPStatusError, httpx.HTTPError, httpx.TimeoutException):
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("teacher request failed")

    def _normalize_r1(self, content: str) -> str:
        if R1_MARKER not in content:
            return content.strip()
        _, _, rest = content.partition(R1_MARKER)
        reasoning, _, tail = rest.partition(R1_TRACE_CLOSE)
        reasoning = formats.clean_for_tags(reasoning.split(R1_TRACE_OPEN, 1)[-1])
        answer = formats.extract_answer(tail)
        if not reasoning or not answer:
            return content.strip()
        return formats.build_completion(reasoning, answer)

    def trace(self, question, reference=None, **kwargs):
        try:
            text = self.generate(question, **kwargs)
        except Exception:
            return None
        if not text.strip():
            return None
        parsed = formats.parse_completion(text)
        if not parsed["ok"] or not parsed["reasoning"]:
            return None
        if reference is not None and not formats.exact_match(text, reference):
            return None
        return {
            "question": question,
            "reasoning": parsed["reasoning"],
            "answer": parsed["answer"],
        }
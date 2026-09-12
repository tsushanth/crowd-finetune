import re

REASONING_OPEN = "<reasoning>"
REASONING_CLOSE = "</reasoning>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

TEACHER_SYSTEM = (
    "You are a careful problem solver. Reason step by step, then give a final answer. "
    f"Return EXACTLY this shape:\n"
    f"{REASONING_OPEN}\n...your step-by-step reasoning...\n{REASONING_CLOSE}\n"
    f"{ANSWER_OPEN}\n...just the final numeric answer...\n{ANSWER_CLOSE}"
)


def build_completion(reasoning: str, answer: str) -> str:
    return (
        f"{REASONING_OPEN}\n{reasoning.strip()}\n{REASONING_CLOSE}\n"
        f"{ANSWER_OPEN}\n{answer.strip()}\n{ANSWER_CLOSE}"
    )


def clean_for_tags(text: str) -> str:
    text = re.sub(r"<\s*reasoning\s*>", "", text)
    text = re.sub(r"<\s*/\s*reasoning\s*>", "", text)
    text = re.sub(r"<\s*answer\s*>", "", text)
    text = re.sub(r"<\s*/\s*answer\s*>", "", text)
    text = re.sub(r"<\s*think\s*>", "", text)
    text = re.sub(r"<\s*/\s*think\s*>", "", text)
    text = text.replace("　　", " ").replace("\\boxed{", "").replace("}", "").strip()
    return text


def parse_completion(text: str) -> dict:
    reasoning = _between(text, REASONING_OPEN, REASONING_CLOSE)
    answer = _between(text, ANSWER_OPEN, ANSWER_CLOSE)
    return {"reasoning": reasoning, "answer": answer, "ok": bool(answer)}


def _between(text: str, open_tag: str, close_tag: str) -> str:
    match = re.search(
        re.escape(open_tag) + r"\s*(.*?)\s*" + re.escape(close_tag), text, re.DOTALL
    )
    return match.group(1).strip() if match else ""


def extract_answer(completion: str) -> str:
    parsed = parse_completion(completion)
    source = parsed["answer"] if parsed["ok"] else completion
    return normalize_number(extract_last_number(source))


def extract_last_number(text: str) -> str:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
    return matches[-1].strip() if matches else ""


def normalize_number(value: str) -> str:
    return value.strip().lstrip("+").replace(",", "")


def exact_match(completion: str, reference: str) -> bool:
    expected = normalize_number(reference)
    return bool(expected) and extract_answer(completion) == expected
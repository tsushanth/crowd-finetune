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


def _latex_to_plain(text: str) -> str:
    """Turns \\frac{a}{b}/\\dfrac{a}{b} into a/b and strips \\boxed{}, $, and stray braces."""
    text = re.sub(r"\\d?frac\s*{\s*([^{}]+?)\s*}\s*{\s*([^{}]+?)\s*}", r"(\1)/(\2)", text)
    text = re.sub(r"\\boxed\s*{\s*(.*?)\s*}", r"\1", text, flags=re.DOTALL)
    text = text.replace("$", "").replace("\\!", "").replace("\\,", "")
    text = text.replace("\\left", "").replace("\\right", "")
    return text


def extract_numeric_token(text: str) -> str:
    """Like extract_last_number, but also recognizes simple fractions (a/b or LaTeX \\frac{a}{b})."""
    text = _latex_to_plain(text).replace(",", "")
    matches = re.findall(r"[-+]?\(?-?\d+(?:\.\d+)?\)?\s*/\s*\(?-?\d+(?:\.\d+)?\)?|[-+]?\d+(?:\.\d+)?", text)
    return matches[-1].strip() if matches else ""


def to_float(token: str):
    token = token.strip().strip("()").replace(" ", "")
    try:
        if "/" in token:
            num, den = token.split("/", 1)
            num, den = float(num.strip("()")), float(den.strip("()"))
            return num / den if den != 0 else None
        return float(token)
    except (ValueError, ZeroDivisionError):
        return None


def numbers_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    """Numeric-equivalence comparison: 3/4 == 0.75 == \\frac{3}{4}. Falls back to False if either
    side doesn't parse as a number (callers should keep their existing string-match fallback)."""
    fa, fb = to_float(a), to_float(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) <= tol * max(1.0, abs(fb))


def numeric_match(completion: str, reference: str, strict: bool = False) -> bool:
    """Numeric-equivalence version of exact_match: accepts plain decimals, fractions, and
    \\frac{}{} LaTeX on both sides, comparing by value rather than by string."""
    expected = extract_numeric_token(reference)
    if not expected:
        return False
    if strict:
        parsed = parse_completion(completion)
        source = parsed["answer"] if parsed["ok"] else None
        if source is None:
            return False
    else:
        source = completion
    got = extract_numeric_token(source)
    return bool(got) and numbers_equal(got, expected)
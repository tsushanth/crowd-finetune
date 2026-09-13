import re

from ..reasoning import formats as _r

REASONING_OPEN = _r.REASONING_OPEN
REASONING_CLOSE = _r.REASONING_CLOSE
ANSWER_OPEN = _r.ANSWER_OPEN
ANSWER_CLOSE = _r.ANSWER_CLOSE

CODE_SYSTEM = (
    "You are a Python engineer solving a programming problem. "
    "Think through the approach inside " + REASONING_OPEN + "..." + REASONING_CLOSE + ", "
    "then output the COMPLETE Python function definition (def line included, no "
    "further imports unless required) inside " + ANSWER_OPEN + "..." + ANSWER_CLOSE + ". "
    "The code must be runnable on its own and pass every listed unit test. "
    "Put nothing but the code inside the " + ANSWER_OPEN + " tags."
)

REPAIR_SYSTEM = (
    "You are a Python engineer debugging existing code. A developer wrote a "
    "function that has a bug, and the unit tests below fail on it. "
    "Think through the bug inside " + REASONING_OPEN + "..." + REASONING_CLOSE + ", "
    "then output the COMPLETE corrected Python function definition (def line "
    "included, no further imports unless required) inside "
    + ANSWER_OPEN + "..." + ANSWER_CLOSE + ". "
    "The corrected code must be runnable on its own and pass every listed "
    "unit test. Put nothing but the corrected code inside the "
    + ANSWER_OPEN + " tags."
)

TEST_MARK = "\n\nUnit tests the implementation must satisfy:"


def strip_tests_from_question(question):
    return question.split(TEST_MARK)[0].strip()


def build_repair_question(docstring, buggy_code, tests):
    lines = "\n".join(tests)
    return (
        strip_tests_from_question(docstring)
        + "\n\nA developer wrote the following solution, but it has a bug and "
        "the unit tests below fail on it. Find the bug and fix it.\n\n"
        "Buggy solution:\n```python\n"
        + buggy_code
        + "\n```\n\nUnit tests that must pass:\n```python\n"
        + lines
        + "\n```"
    )


def build_completion(reasoning, code):
    return _r.build_completion(reasoning, code)


def parse_completion(text):
    return _r.parse_completion(text)


def extract_code(completion):
    parsed = _r.parse_completion(completion)
    source = parsed["answer"] if parsed["ok"] else completion
    return strip_code_fence(source)


def strip_code_fence(text):
    blocks = re.findall(r"```(?:[^\s`]*\n)?(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text.strip()
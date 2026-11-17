"""Automatic correctness checks for the side-by-side demo's preset questions.

Two kinds of check:
  contains : every regex pattern must appear in the answer (facts, simple math)
  python   : extract the model's code and run it against test cases

Free-form questions have no check; judge those by eye.

SAFETY NOTE: the python check executes code written by a language model. It runs
in a separate `python -I` process with a timeout, in a temp folder, and refuses
code that uses imports, file access, eval/exec and similar. A denylist is NOT a
real sandbox, so only use it on the preset questions below.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

PRESETS: list[dict] = [
    {"id": "capital", "title": "Fact: capital of France", "max_tokens": 80,
     "question": "What is the capital of France? Answer in one sentence.",
     "check": {"kind": "contains", "patterns": ["paris"]}},
    {"id": "author", "title": "Fact: Romeo and Juliet", "max_tokens": 80,
     "question": "Who wrote the play Romeo and Juliet? Answer in one sentence.",
     "check": {"kind": "contains", "patterns": ["shakespeare"]}},
    {"id": "math_add", "title": "Math: 17 + 25", "max_tokens": 60,
     "question": "What is 17 + 25? Reply with just the number.",
     "check": {"kind": "contains", "patterns": [r"\b42\b"]}},
    {"id": "math_word", "title": "Math: word problem", "max_tokens": 120,
     "question": "Tom has 3 apples and buys 4 more. How many apples does Tom have now?",
     "check": {"kind": "contains", "patterns": [r"\b7\b|seven"]}},
    {"id": "code_reverse", "title": "Code: reverse a string", "max_tokens": 200,
     "question": "Write a Python function named reverse_string(s) that returns the string s reversed. Reply with only the code.",
     "check": {"kind": "python", "tests": [("reverse_string('hello')", "olleh"),
                                           ("reverse_string('')", ""), ("reverse_string('ab')", "ba")]}},
    {"id": "code_even", "title": "Code: is a number even", "max_tokens": 200,
     "question": "Write a Python function named is_even(n) that returns True if the integer n is even and False otherwise. Reply with only the code.",
     "check": {"kind": "python", "tests": [("is_even(4)", True), ("is_even(7)", False), ("is_even(0)", True)]}},
    {"id": "explain", "title": "Open-ended: neural network (judge by eye)", "max_tokens": 200,
     "question": "Explain what a neural network is in simple terms.", "check": None},
]

_BLOCKED = re.compile(
    r"\b(import|exec|eval|open|compile|globals|locals|getattr|setattr|delattr|input|breakpoint|exit|quit)\b|__"
)
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)

# Appended after the model's code. Our own test expressions are evaluated here.
_HARNESS = """

_fails = []
for _expr, _want in __TESTS__:
    try:
        _got = eval(_expr)
    except Exception as _e:
        _fails.append(_expr + " raised " + type(_e).__name__)
        continue
    if _got != _want:
        _fails.append(_expr + " gave " + repr(_got) + ", expected " + repr(_want))
print("RESULT:" + ("PASS" if not _fails else "FAIL: " + "; ".join(_fails)))
"""


def public_presets() -> list[dict]:
    """What the browser sees: no test cases or patterns."""
    return [{k: p[k] for k in ("id", "title", "question", "max_tokens")} for p in PRESETS]


def extract_code(text: str) -> str:
    """The first fenced code block, else everything from the first 'def' on."""
    match = _FENCE.search(text)
    if match:
        return match.group(1)
    start = text.find("def ")
    return text[start:] if start >= 0 else text


def check_contains(patterns: list[str], text: str) -> dict:
    missing = [p for p in patterns if not re.search(p, text, re.IGNORECASE)]
    if missing:
        return {"passed": False, "detail": "expected answer not found in the text"}
    return {"passed": True, "detail": "expected answer found"}


def check_python(text: str, tests: list[tuple], timeout: float = 5.0) -> dict:
    code = extract_code(text)
    blocked = _BLOCKED.search(code)
    if blocked:
        return {"passed": False, "detail": f"not run: code uses '{blocked.group(0)}'"}
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "answer_check.py"
        script.write_text(code + _HARNESS.replace("__TESTS__", repr(tests)), encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(script)], capture_output=True,
                                  text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return {"passed": False, "detail": f"timed out after {timeout:.0f}s"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            verdict = line[len("RESULT:"):]
            return {"passed": verdict == "PASS", "detail": "all tests passed" if verdict == "PASS" else verdict[6:]}
    err = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return {"passed": False, "detail": f"code did not run: {err[:120]}"}


def check_answer(preset_id: str | None, text: str, timeout: float = 5.0) -> dict | None:
    """Verdict for a preset question; None when there is nothing to check."""
    preset = next((p for p in PRESETS if p["id"] == preset_id), None)
    check = preset["check"] if preset else None
    if not check:
        return None
    if check["kind"] == "contains":
        return check_contains(check["patterns"], text)
    return check_python(text, check["tests"], timeout)

"""Frozen probe-v1 parser and exact-answer verifier; parsing semantics are unchanged."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any
import re

VERIFIER_VERSION = "rlvr-probe-compatible-v1"


def verify(prompt: str, completion: str, domain: str, gold_answer: str,
           option_labels: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """Keep correctness independent from formatting/termination validity."""
    if domain not in ("knowledge", "math"):
        raise ValueError(f"Unsupported domain: {domain}")
    row = {"canonical_answer": str(gold_answer), "options_bo": dict.fromkeys(option_labels, "")}
    parsed = parse_protocol(prompt, completion)
    extractable, answer, correct = parse_answer(Candidate(domain, row, b""), parsed["final_region"])
    final = parsed["final_region"]
    starts = completion.count(THINK_OPEN) + int(parsed["prefix_open"])
    closes = completion.count(THINK_CLOSE)
    return {**parsed, "answer_extractable": extractable, "extracted_answer": answer,
            "correctness": int(correct), "thinking_closed": THINK_CLOSE in completion,
            "thinking_unclosed": starts > closes,
            "no_reliable_final": final is None,
            "no_answer": None if final is None else not bool(final.strip()),
            "empty_final": final is not None and not bool(final.strip()),
            "extraction_failed": not extractable}
PROMPT_VERSION = "tibetan_rl_probe_v1_enable_thinking_true"

PROMPT_KNOWLEDGE_SUFFIX = (
    "\n\nཁྱེད་ཀྱིས་ཐད་ཀར་ལན་འདེབས་ཆོག དགོས་ན་བསམ་གཞིག་བྱས་ཆོག མཐའ་མའི་ལན་བོད་སྐད་དུ་བྲིས་ནས། འདེམས་གཞིའི་གསལ་བྱེད་གང་ཡིན་པ་གསལ་པོར་བཤད།"
)

PROMPT_MATH_SUFFIX = (
    "\n\nཁྱེད་ཀྱིས་ཐད་ཀར་ལན་འདེབས་ཆོག དགོས་ན་བསམ་གཞིག་བྱས་ཆོག མཐའ་མའི་ལན་བོད་སྐད་དུ་བྲིས་ནས། གྲངས་ཀའི་ལན་མཐའ་མ་དེ་གསལ་པོར་སྟོན།"
)

THINK_OPEN = "<think>"

THINK_CLOSE = "</think>"

END_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")

NUMBER_RE = re.compile(
    r"(?<![\w.])([+\-−]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:\s*/\s*[+\-−]?(?:\d+(?:\.\d*)?|\.\d+))?)(?![\w.])"
)

LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([A-J])(?![A-Za-z0-9])", re.IGNORECASE)

LABEL_AT_END_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-J])\s*[.\)\]】:：,，;；!?།༎。]*\s*$", re.IGNORECASE
)

ANSWER_CUE_RE = re.compile(
    r"(?:answer|option|choice|ལན|འདེམས་གཞི)[^A-J]{0,32}([A-J])\b",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class Candidate:
    domain: str
    row: dict[str, Any]
    raw_line: bytes

    @property
    def row_id(self) -> str:
        return str(self.row["id"])

def make_prompt(candidate: Candidate) -> str:
    row = candidate.row
    body = row["question_bo"].strip()
    if candidate.domain == "knowledge":
        option_lines = [f"{label}. {text.strip()}" for label, text in row["options_bo"].items()]
        body = body + "\n\n" + "\n".join(option_lines) + PROMPT_KNOWLEDGE_SUFFIX
    else:
        body = body + PROMPT_MATH_SUFFIX
    return body

def render_prompt(tokenizer: Any, candidate: Candidate) -> tuple[str, bool]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": make_prompt(candidate)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    marker = "<|im_start|>assistant"
    suffix = prompt.rsplit(marker, 1)[-1] if marker in prompt else ""
    prefix_open = suffix.count(THINK_OPEN) > suffix.count(THINK_CLOSE)
    return prompt, prefix_open

def strip_completion_ending(text: str) -> str:
    value = text.rstrip()
    changed = True
    while changed:
        changed = False
        for marker in END_MARKERS:
            if value.endswith(marker):
                value = value[: -len(marker)].rstrip()
                changed = True
    return value

def parse_protocol(prompt: str, completion: str) -> dict[str, Any]:
    """Parse format compliance and answer region independently."""
    raw = strip_completion_ending(completion)
    assistant_marker = "<|im_start|>assistant"
    assistant_prefix = prompt.rsplit(assistant_marker, 1)[-1] if assistant_marker in prompt else ""
    prefix_open = assistant_prefix.count(THINK_OPEN) > assistant_prefix.count(THINK_CLOSE)
    starts = [m.start() for m in re.finditer(re.escape(THINK_OPEN), raw)]
    ends = [m.start() for m in re.finditer(re.escape(THINK_CLOSE), raw)]
    protocol_valid = False
    body: str | None = None
    final: str | None = None
    parse_note = "malformed_think_boundary"

    if prefix_open:
        if not starts and ends:
            boundary = ends[0]
            body = raw[:boundary]
            final = raw[boundary + len(THINK_CLOSE) :].strip()
            protocol_valid = len(ends) == 1
            parse_note = "template_prefix_closed" if protocol_valid else "template_prefix_structure_error"
        elif not starts:
            body = raw
            parse_note = "template_prefix_unclosed"
        else:
            first_close = ends[0] if ends else None
            body = raw[:first_close] if first_close is not None else raw
            parse_note = "template_prefix_structure_error"
    elif not starts and not ends:
        # A direct response has a reliably identifiable final region even though
        # the strict <think>...</think> protocol was not followed.
        final = raw.strip()
        body = ""
        parse_note = "direct_no_think_without_tags"
    elif not starts and ends:
        # The last unmatched close still gives a reliable final boundary.
        body = ""
        boundary = ends[-1]
        final = raw[boundary + len(THINK_CLOSE) :].strip()
        parse_note = "unmatched_close_final_recovered" if len(ends) == 1 else "multiple_unmatched_closes_final_recovered"
    elif len(starts) == 1 and ends and starts[0] < ends[0]:
        boundary = next((pos for pos in ends if pos > starts[0]), None)
        assert boundary is not None
        body = raw[starts[0] + len(THINK_OPEN) : boundary]
        final = raw[boundary + len(THINK_CLOSE) :].strip()
        protocol_valid = len(ends) == 1 and not raw[: starts[0]].strip()
        parse_note = "balanced_think_tags" if protocol_valid else "extra_think_boundary_final_recovered"
    else:
        if starts:
            first_start = starts[0] + len(THINK_OPEN)
            next_end = next((pos for pos in ends if pos >= first_start), None)
            body = raw[first_start:next_end] if next_end is not None else raw[first_start:]
        parse_note = "ambiguous_think_boundary"

    if body is None:
        think_mode = "no_think" if not starts and not prefix_open else "think"
        think_body = ""
    else:
        think_body = body
        think_mode = "think" if body.strip() else "no_think"
    return {
        "protocol_valid": protocol_valid,
        "think_mode": think_mode,
        "think_body": think_body,
        "final_region": final,
        "parse_note": parse_note,
        "prefix_open": prefix_open,
    }

def normalize_number(value: str) -> Fraction:
    normalized = value.strip().replace("−", "-").replace(" ", "")
    if "/" in normalized:
        numerator, denominator = normalized.split("/", 1)
        left = Fraction(Decimal(numerator))
        right = Fraction(Decimal(denominator))
        if right == 0:
            raise ZeroDivisionError("fraction answer denominator is zero")
        return left / right
    return Fraction(Decimal(normalized))

def extract_knowledge_answer(final_region: str, option_labels: set[str]) -> tuple[bool, str | None]:
    text = final_region.strip()
    if not text:
        return False, None
    end_match = LABEL_AT_END_RE.search(text)
    if end_match:
        label = end_match.group(1).upper()
        return True, label
    cues = list(ANSWER_CUE_RE.finditer(text))
    if cues:
        label = cues[-1].group(1).upper()
        return True, label
    labels = [match.group(1).upper() for match in LABEL_RE.finditer(text)]
    unique = set(labels)
    if len(unique) == 1:
        return True, labels[-1]
    return False, None

def extract_math_answer(final_region: str) -> tuple[bool, str | None]:
    if not final_region.strip():
        return False, None
    matches = list(NUMBER_RE.finditer(final_region.replace("−", "-")))
    if not matches:
        return False, None
    return True, re.sub(r"\s+", "", matches[-1].group(1))

def parse_answer(candidate: Candidate, final_region: str | None) -> tuple[bool, str | None, bool]:
    if final_region is None:
        return False, None, False
    if candidate.domain == "knowledge":
        labels = set(candidate.row["options_bo"])
        extractable, answer = extract_knowledge_answer(final_region, labels)
        correct = bool(extractable and answer in labels and answer == candidate.row["canonical_answer"])
        return extractable, answer, correct
    extractable, answer = extract_math_answer(final_region)
    if not extractable or answer is None:
        return False, answer, False
    try:
        correct = normalize_number(answer) == normalize_number(str(candidate.row["canonical_answer"]))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        correct = False
    return True, answer, bool(correct)

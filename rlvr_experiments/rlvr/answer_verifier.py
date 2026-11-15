"""CPU-only verifier shared by probes and explicit v2 RL reward adapters."""
from __future__ import annotations
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
import unicodedata
from typing import Any
from rlvr.verifier import parse_protocol

VERIFIER_VERSION = "rlvr-answer-v2"
MAX_FINAL_CHARS = 100_000
_ATOM = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_VALUE = rf"{_ATOM}(?:\s*/\s*{_ATOM})?"
NUMBER = re.compile(rf"(?<![A-Za-z0-9_.,])({_VALUE})(\s*%)?(?![A-Za-z0-9_.,])")
SCALAR = re.compile(rf"^\s*({_VALUE})(\s*%)?\s*$")
FRAC = re.compile(r"([+-]?)\s*\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
CUE = re.compile(
    r"\b(?:(?:final|correct)\s+answer|answer|final\s+result)\b"
    r"\s*(?:is\b|=|:|：)?"
    r"|(?:མཐའ་མའི་)?ལན(?:་(?:ཡང་དག་པ|ཡང་དག|ངོ་མ|དག|ནི))*"
    r"|ཡང་དག་པའི་(?:འདེམས་གཞི|འདེམས་ཚན|གདམ་ག)(?:་ནི)?"
    r"|(?:འདེམས་གཞི|འདེམས་ཚན|གདམ་ག)་ཡང་དག་པ(?:་ནི)?",
    re.IGNORECASE)
OPTION_CUE = re.compile(r"\b(?:option|choice)\b\s*(?:is\b|=|:|：)?", re.IGNORECASE)
SENTENCE_END = re.compile(r"\n|[。།༎!?;；]|\.(?=\s|$)")
ANY_LABEL = re.compile(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])")
ALTERNATIVE = re.compile(
    r"^\s*(?:[,/]|or\b|and\b|and/or\b|ཡང་ན|དང|或者|或)\s*"
    r"[\s་:：(\[]*([A-J])", re.IGNORECASE)

def surface(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u2212", "-").replace("\u2044", "/")
    value = value.replace("\\%", "%").replace("\\,", "").replace("\\!", "")
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in value)

def _atom_fraction(text: str) -> Fraction:
    value = text.strip()
    if "," in value:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?", value):
            raise ValueError("invalid_thousands_separator")
        value = value.replace(",", "")
    if len(value) > 256:
        raise ValueError("numeric_literal_too_long")
    exponent = re.search(r"[eE]([+-]?\d+)$", value)
    if exponent and abs(int(exponent.group(1))) > 1000:
        raise ValueError("numeric_exponent_out_of_range")
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError("nonfinite_number")
    return Fraction(number)

def rational(text: str, policy: dict[str, Any] | None = None) -> Fraction:
    """Parse a bounded literal; never evaluate arbitrary model expressions."""
    policy = policy or {}
    value = surface(text).strip().strip("$").strip()
    for _ in range(3):
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1].strip()
    match = SCALAR.fullmatch(value)
    if not match:
        raise ValueError("unsupported_numeric")
    expr, percent = match.groups()
    if "/" in expr:
        left, right = expr.split("/", 1)
        denominator = _atom_fraction(right)
        if not denominator:
            raise ValueError("zero_denominator")
        result = _atom_fraction(left) / denominator
    else:
        result = _atom_fraction(expr)
    if percent:
        mode = policy.get("percent_mode")
        if mode == "ratio":
            result /= 100
        elif mode != "literal":
            raise ValueError("undeclared_percent_semantics")
    return result

def canonical(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"

def _latex(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        sign, numerator, denominator = match.groups()
        value = rational(numerator) / rational(denominator)
        if sign == "-":
            value = -value
        return "(" + canonical(value) + ")"
    return FRAC.sub(replace, text)

def _boxes(text: str) -> list[str]:
    boxes = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, start, pos = 1, match.end(), match.end()
        while pos < len(text) and depth:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth:
            raise ValueError("unclosed_answer_box")
        boxes.append(text[start:pos - 1])
    return boxes

def _regions(text: str) -> list[str]:
    result = []
    for match in list(CUE.finditer(text)) or list(OPTION_CUE.finditer(text)):
        tail = text[match.end():].lstrip(" \t\r\n:：=*_\x60་།༎")
        stop = SENTENCE_END.search(tail)
        result.append(tail[:stop.start()] if stop else tail)
    return [r for r in result if r.strip()]

def _choice_region(text: str, explicit: bool) -> tuple[list[str], str | None]:
    text = surface(text)
    matches = list(ANY_LABEL.finditer(text))
    if not matches:
        lower = re.fullmatch(r"[\s*\x60$([{]*([a-j])[\s*\x60$)\]}.。།༎:：]*", text)
        return ([lower.group(1).upper()], None) if lower else ([], None)
    if explicit:
        first = matches[0]
        other = ALTERNATIVE.match(text[first.end():])
        if other and other.group(1).upper() != first.group(1):
            return [first.group(1), other.group(1).upper()], "ambiguous"
        return [first.group(1)], None
    return [m.group(1) for m in matches], None

def extract_choice(final: str, labels: tuple[str, ...]) -> dict[str, Any]:
    text = surface(final)
    try:
        boxes = _boxes(text)
    except ValueError as exc:
        return {"parse_status": "ambiguous", "reason": str(exc)}
    regions = _regions(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    terminal = []
    if lines and re.match(r"^[\s*\x60$([{]*[A-Ja-j](?=[\s*\x60$)\]}]*(?:[.:：་།༎]|$))", lines[-1]):
        terminal = [lines[-1]]
    values, reasons = [], []
    if boxes or regions or terminal:
        for part in boxes + regions + terminal:
            found, reason = _choice_region(part, True)
            values.extend(found)
            if reason:
                reasons.append(reason)
    else:
        values, reason = _choice_region(text, False)
        if reason:
            reasons.append(reason)
    unique = set(values)
    if reasons or len(unique) > 1:
        return {"parse_status": "ambiguous", "reason": "conflicting_final_labels"}
    if not unique:
        return {"parse_status": "no_answer", "reason": "no_final_label"}
    value = next(iter(unique))
    if value not in labels:
        return {"parse_status": "invalid_option", "extracted_answer_raw": value,
                "reason": "label_outside_option_set"}
    return {"parse_status": "ok", "extracted_answer_raw": value, "normalized_answer": value}

def _numeric_values(text: str, policy: dict[str, Any]) -> list[tuple[str, Fraction]]:
    text = _latex(surface(text))
    text = re.sub(r"(?<=\d),(?=\s+[-+]?\d)", " ; ", text)
    if re.search(r"\\(?:frac|dfrac|tfrac|sqrt)\b", text):
        raise ValueError("unsupported_numeric_expression")
    if re.search(r"\\(?:pi|sqrt|sin|cos|tan|log|ln|exp|binom|lvert|rvert)\b|[π∑|]", text):
        raise ValueError("unsupported_numeric_expression")
    # Final numeric answers are literals, not expressions whose repeated operands
    # could accidentally collapse to one extracted value (for example, 2**2).
    if re.search(r"(?:\d|\))\s*(?:\*\*?|[×÷^+\-]|\\(?:times|cdot|div|pm|mp)\b)"
                 r"\s*[+\-]?\s*(?:\d|\()", text):
        raise ValueError("unsupported_numeric_expression")
    if re.search(r"\d\s*\(\s*[+\-]?\d|\d\s*!", text):
        raise ValueError("unsupported_numeric_expression")
    if re.search(r"\b(?:nan|inf|infinity)\b|∞", text, re.IGNORECASE):
        raise ValueError("nonfinite_number")
    values = []
    for match in NUMBER.finditer(text):
        raw = match.group(1) + ("%" if match.group(2) else "")
        values.append((raw, rational(raw, policy)))
    return values

def extract_numeric(final: str, policy: dict[str, Any]) -> dict[str, Any]:
    try:
        text = surface(final)
        boxes = _boxes(text)
        regions = _regions(text)
        terminal = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            last = lines[-1]
            literal = last.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
            literal = literal.strip(" *\x60$").rstrip("。།༎")
            try:
                rational(_latex(literal), policy)
                terminal = [last]
            except (ValueError, InvalidOperation, ZeroDivisionError, OverflowError):
                if re.match(r"^(?:[*#\s]*)(?:དེས་ན|དེ་ལྟར|དེར་བརྟེན|དེའི་ཕྱིར|Therefore\b|Thus\b|Hence\b)", last, re.IGNORECASE):
                    terminal = [last]
        parts = boxes + regions + terminal if boxes or regions or terminal else [text]
        values = []
        for part in parts:
            values.extend(_numeric_values(part, policy))
        unique = {value for _, value in values}
        if len(unique) > 1:
            return {"parse_status": "ambiguous", "reason": "conflicting_final_values"}
        if not unique:
            return {"parse_status": "no_answer", "reason": "no_numeric_answer"}
        result = next(iter(unique))
        return {"parse_status": "ok", "extracted_answer_raw": values[0][0],
                "normalized_answer": canonical(result)}
    except (ValueError, InvalidOperation, ZeroDivisionError, OverflowError) as exc:
        return {"parse_status": "unsupported_numeric", "reason": str(exc)}

def _sample_contract(sample: dict[str, Any]) -> tuple[str, Any, tuple[str, ...], dict[str, Any]]:
    domain = sample["domain"]
    labels = tuple(sample.get("option_labels") or sample.get("options_bo") or ())
    policy = dict(sample.get("numeric_policy") or {})
    gold = str(sample["canonical_answer"]).strip()
    if domain == "knowledge":
        if not labels or len(labels) != len(set(labels)) or any(x not in "ABCDEFGHIJ" or len(x) != 1 for x in labels):
            raise ValueError("Invalid option-label metadata")
        if gold not in labels:
            raise ValueError("Gold outside option-label metadata")
        return domain, gold, labels, policy
    if domain != "math":
        raise ValueError(f"Unsupported domain: {domain}")
    return domain, rational(gold, policy), labels, policy

def verify_answer(
    sample: dict[str, Any], *, prompt: str, completion: str,
    finish_reason: str | None, output_tokens: int | None = None,
    max_new_tokens: int | None = None, backend_error: bool = False,
) -> dict[str, Any]:
    domain, gold, labels, policy = _sample_contract(sample)
    if output_tokens is not None and (output_tokens < 0 or (max_new_tokens is not None and output_tokens > max_new_tokens)):
        raise ValueError("Output token count violates generation contract")
    termination = ("runtime_error" if backend_error or finish_reason in ("abort", "aborted", "error", "cancelled")
                   else "completed" if finish_reason == "stop"
                   else "truncated" if finish_reason == "length" else "unknown")
    parsed = parse_protocol(prompt, completion)
    final = parsed["final_region"]
    if final is None:
        answer = {"parse_status": "no_final", "reason": parsed["parse_note"]}
    elif not final.strip():
        answer = {"parse_status": "no_answer", "reason": "empty_final"}
    elif len(final) > MAX_FINAL_CHARS:
        answer = {"parse_status": "unsupported", "reason": "final_region_too_long"}
    else:
        answer = extract_choice(final, labels) if domain == "knowledge" else extract_numeric(final, policy)
    value = answer.get("normalized_answer")
    correct = int(answer["parse_status"] == "ok" and
                  (value == gold if domain == "knowledge" else rational(value) == gold))
    execution_valid = termination in ("completed", "truncated")
    depth = int(parsed["prefix_open"])
    for marker in re.findall(r"<think>|</think>", completion):
        depth = depth + 1 if marker == "<think>" else max(0, depth - 1)
    result = {
        **parsed, **answer, "verifier_version": VERIFIER_VERSION,
        "protocol_version": "english_think_tibetan_final_v1",
        "correctness": correct if execution_valid else None,
        "answer_correct": bool(correct) if execution_valid else None,
        "answer_extractable": answer["parse_status"] == "ok",
        "extracted_answer": value, "normalized_gold": gold if domain == "knowledge" else canonical(gold),
        "termination_status": termination, "execution_valid": execution_valid,
        "truncated": termination == "truncated", "finish_reason": finish_reason,
        "output_tokens": output_tokens, "max_new_tokens": max_new_tokens,
        "pair_eligible": termination == "completed" and answer["parse_status"] == "ok",
        "thinking_unclosed": depth > 0,
        "thinking_closed": "</think>" in completion and depth == 0,
        "no_reliable_final": final is None, "final_present": final is not None,
        "empty_final": final is not None and not bool(final.strip()),
        "no_answer": None if final is None else not bool(final.strip()),
        "extraction_failed": answer["parse_status"] != "ok",
        "reason_codes": [answer["reason"]] if answer.get("reason") else [],
    }
    if not execution_valid:
        result["reason_codes"].append("unresolved_or_failed_execution")
    return result

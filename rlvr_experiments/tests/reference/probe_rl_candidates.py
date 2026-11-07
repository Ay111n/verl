#!/usr/bin/env python3
"""Probe Tibetan RL candidates with reproducible Qwen3 rollouts and sample final train data."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator


SEED = 42
PILOT_PER_DOMAIN = 500
ROLLOUTS_PER_QUESTION = 4
FINAL_PER_DOMAIN = 8_000
TARGET_BUCKET_RATIOS = {"B0": 0.05, "B1": 0.15, "B2": 0.30, "B3": 0.30, "B4": 0.20}
BUCKET_ORDER = ("B0", "B1", "B2", "B3", "B4")
PROMPT_VERSION = "tibetan_rl_probe_v1_enable_thinking_true"
PROMPT_KNOWLEDGE_SUFFIX = (
    "\n\nཁྱེད་ཀྱིས་ཐད་ཀར་ལན་འདེབས་ཆོག དགོས་ན་བསམ་གཞིག་བྱས་ཆོག མཐའ་མའི་ལན་བོད་སྐད་དུ་བྲིས་ནས། འདེམས་གཞིའི་གསལ་བྱེད་གང་ཡིན་པ་གསལ་པོར་བཤད།"
)
PROMPT_MATH_SUFFIX = (
    "\n\nཁྱེད་ཀྱིས་ཐད་ཀར་ལན་འདེབས་ཆོག དགོས་ན་བསམ་གཞིག་བྱས་ཆོག མཐའ་མའི་ལན་བོད་སྐད་དུ་བྲིས་ནས། གྲངས་ཀའི་ལན་མཐའ་མ་དེ་གསལ་པོར་སྟོན།"
)
KNOWLEDGE_SOURCE = "nemotron_rl_knowledge_mcqa"
MATH_SOURCE = "ace_reason_math"

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = "/home/nlp/data/lyl/model/qwen3-8bbo-sft-new2-mixed-checkpoint6000-merged"
DEFAULT_MAX_MODEL_LEN = 8_192

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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(*parts: object) -> bytes:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def rollout_seed(domain: str, row_id: str, rollout_index: int, base_seed: int = SEED) -> int:
    return int.from_bytes(stable_digest(base_seed, domain, row_id, rollout_index)[:8], "big") & 0x7FFFFFFF


def atomic_write_bytes(path: Path, content: bytes, *, refuse_different: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        if refuse_different:
            raise FileExistsError(f"Refusing to overwrite differing output: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def recover_jsonl_tail(path: Path) -> None:
    """Discard only a possible incomplete last JSONL line after an interrupted write."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        stream.seek(end - 1)
        if stream.read(1) == b"\n":
            return
        search_start = max(0, end - 8 * 1024 * 1024)
        stream.seek(search_start)
        tail = stream.read()
        last_newline = tail.rfind(b"\n")
        keep = search_start + last_newline + 1 if last_newline >= 0 else 0
        stream.truncate(keep)
        stream.flush()
        os.fsync(stream.fileno())


def iter_jsonl(path: Path, *, recover_tail: bool = False) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    if recover_tail:
        recover_jsonl_tail(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield value


def read_jsonl(path: Path, *, recover_tail: bool = False) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, recover_tail=recover_tail))


def load_candidates(root: Path) -> tuple[list[Candidate], dict[str, Any]]:
    domain_paths = {
        "knowledge": root / "Nemotron-RL-knowledge-mcqa" / "processed_data" / "knowledge_rl_train_candidates.jsonl",
        "math": root / "AceReason-Math" / "processed_data" / "math_rl_train_candidates.jsonl",
    }
    candidates: list[Candidate] = []
    input_manifest: dict[str, Any] = {}
    seen: set[str] = set()
    expected = {"knowledge": KNOWLEDGE_SOURCE, "math": MATH_SOURCE}
    for domain, path in domain_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        count = 0
        with path.open("rb") as stream:
            for line_no, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    raise ValueError(f"Bad UTF-8 JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_no}")
                row_id = row.get("id")
                if not isinstance(row_id, str) or not row_id or row_id in seen:
                    raise ValueError(f"Missing or duplicate ID at {path}:{line_no}: {row_id!r}")
                seen.add(row_id)
                if row.get("domain") != domain or row.get("source") != expected[domain]:
                    raise ValueError(f"Unexpected domain/source at {path}:{line_no}")
                if not isinstance(row.get("question_bo"), str) or not row["question_bo"].strip():
                    raise ValueError(f"Empty question_bo at {path}:{line_no}")
                if not row.get("canonical_answer"):
                    raise ValueError(f"Missing canonical_answer at {path}:{line_no}")
                if domain == "knowledge":
                    options = row.get("options_bo")
                    if not isinstance(options, dict) or len(options) < 2:
                        raise ValueError(f"Invalid options_bo at {path}:{line_no}")
                    if any(not isinstance(k, str) or not re.fullmatch(r"[A-J]", k) for k in options):
                        raise ValueError(f"Invalid Knowledge option label at {path}:{line_no}")
                    if any(not isinstance(v, str) or not v.strip() for v in options.values()):
                        raise ValueError(f"Empty Knowledge option text at {path}:{line_no}")
                    if row["canonical_answer"] not in options:
                        raise ValueError(f"Canonical answer not present among options at {path}:{line_no}")
                elif row.get("answer_type") not in {"integer", "decimal", "fraction"}:
                    raise ValueError(f"Unsupported Math answer_type at {path}:{line_no}")
                candidates.append(Candidate(domain, row, raw if raw.endswith(b"\n") else raw + b"\n"))
                count += 1
        input_manifest[domain] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": count,
        }
    counts = collections.Counter(item.domain for item in candidates)
    if counts != {"knowledge": 11_754, "math": 12_512}:
        raise ValueError(f"Unexpected candidate counts: {dict(counts)}")
    return candidates, input_manifest


def model_manifest(model_path: Path, cache_path: Path | None = None) -> dict[str, Any]:
    fixed_names = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    files = [model_path / name for name in fixed_names if (model_path / name).is_file()]
    files.extend(sorted(model_path.glob("model-*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No model assets found at {model_path}")
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        current = {path.name: path for path in files}
        cached_files = cached.get("files", {})
        if set(cached_files) == set(current) and all(
            int(cached_files[name].get("bytes", -1)) == current[name].stat().st_size
            and int(cached_files[name].get("mtime_ns", -1)) == current[name].stat().st_mtime_ns
            for name in current
        ):
            return cached
    result = {
        "path": str(model_path),
        "files": {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in files
        },
    }
    if cache_path is not None:
        atomic_write_json(cache_path, result)
    return result


def package_manifest() -> dict[str, Any]:
    import importlib.metadata

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    try:
        import torch

        versions["torch_cuda"] = str(torch.version.cuda)
        versions["cuda_available"] = str(torch.cuda.is_available())
        versions["cuda_devices"] = str(torch.cuda.device_count())
    except Exception:
        pass
    return versions


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


def bucket_for(correct_count: int, total: int) -> str:
    if total == 4:
        return f"B{correct_count}"
    if total == 8:
        if correct_count == 0:
            return "B0"
        if correct_count <= 2:
            return "B1"
        if correct_count <= 5:
            return "B2"
        if correct_count <= 7:
            return "B3"
        return "B4"
    raise ValueError(f"Unexpected rollout count for difficulty: {total}")


def self_test() -> None:
    cases = [
        ("direct_no_think", "<|im_start|>assistant\n", "ལན་ A", False, "no_think", "ལན་ A"),
        ("empty_think", "<|im_start|>assistant\n", "<think></think>42", True, "no_think", "42"),
        ("reasoned_think", "<|im_start|>assistant\n", "<think>work</think>42", True, "think", "42"),
        ("template_prefix_empty", "<|im_start|>assistant\n<think>\n", "</think>42", True, "no_think", "42"),
        ("template_prefix_reasoning", "<|im_start|>assistant\n<think>\n", "work</think>42", True, "think", "42"),
        ("template_prefix_unclosed", "<|im_start|>assistant\n<think>\n", "work 42", False, "think", None),
        ("stray_close", "<|im_start|>assistant\n", "</think>A", False, "no_think", "A"),
        ("multiple_stray_closes", "<|im_start|>assistant\n", "</think></think>A", False, "no_think", "A"),
        ("extra_close_final_recovered", "<|im_start|>assistant\n", "<think>work</think></think>42", False, "think", "</think>42"),
    ]
    for name, prompt, completion, want_valid, want_mode, want_final in cases:
        parsed = parse_protocol(prompt, completion)
        if (parsed["protocol_valid"], parsed["think_mode"], parsed["final_region"]) != (
            want_valid,
            want_mode,
            want_final,
        ):
            raise AssertionError(f"Protocol self-test failed {name}: {parsed}")
    if extract_knowledge_answer("The answer is B.", {"A", "B"}) != (True, "B"):
        raise AssertionError("Knowledge answer extraction self-test failed")
    if extract_knowledge_answer("I considered A, then later saw B as well.", {"A", "B"})[0]:
        raise AssertionError("Ambiguous Knowledge answer extraction should fail")
    if extract_math_answer("ལན་ནི -3/7 ཡིན།") != (True, "-3/7"):
        raise AssertionError("Math answer extraction self-test failed")
    for a, b in (("42", "42.0"), ("0.5", "1/2"), ("-3/7", "-3/7")):
        if normalize_number(a) != normalize_number(b):
            raise AssertionError(f"Numeric normalization self-test failed: {a}, {b}")
    direct = parse_protocol("<|im_start|>assistant\n", "B")
    fake_candidate = Candidate(
        "knowledge",
        {
            "id": "self-test",
            "source": KNOWLEDGE_SOURCE,
            "domain": "knowledge",
            "canonical_answer": "B",
            "options_bo": {"A": "a", "B": "b"},
        },
        b"",
    )
    extractable, answer, correct = parse_answer(fake_candidate, direct["final_region"])
    if direct["protocol_valid"] or direct["think_mode"] != "no_think" or not extractable or answer != "B" or not correct:
        raise AssertionError("Protocol validity and answer extraction are not decoupled")
    if bucket_for(2, 4) != "B2" or bucket_for(2, 8) != "B1" or bucket_for(7, 8) != "B3":
        raise AssertionError("Difficulty bucket self-test failed")
    quota_test = choose_difficulty_quotas(
        collections.Counter({"B0": 100, "B1": 1000, "B2": 2500, "B3": 2500, "B4": 5000}),
        FINAL_PER_DOMAIN,
    )
    if sum(quota_test.values()) != FINAL_PER_DOMAIN or quota_test["B0"] > 800:
        raise AssertionError("Difficulty quota allocation self-test failed")
    print("self-test=PASS parser=9 answer_extraction=4 numeric_normalization=3 difficulty=PASS", flush=True)


def load_rollout_index(path: Path, *, recover_tail: bool = True) -> tuple[list[dict[str, Any]], set[tuple[str, str, int]]]:
    rows = read_jsonl(path, recover_tail=recover_tail)
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (str(row.get("domain")), str(row.get("id")), int(row.get("rollout_index", -1)))
        if key in seen:
            raise ValueError(f"Duplicate rollout key in {path}: {key}")
        seen.add(key)
    return rows, seen


def load_pilot_attempts(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, int, int]]]:
    rows = read_jsonl(path, recover_tail=True)
    seen: set[tuple[str, str, int, int]] = set()
    for row in rows:
        key = (
            str(row.get("domain")),
            str(row.get("id")),
            int(row.get("rollout_index", -1)),
            int(row.get("pilot_attempt", -1)),
        )
        if key in seen:
            raise ValueError(f"Duplicate pilot attempt key in {path}: {key}")
        seen.add(key)
    return rows, seen


def candidate_by_key(candidates: list[Candidate]) -> dict[tuple[str, str], Candidate]:
    return {(candidate.domain, candidate.row_id): candidate for candidate in candidates}


def prompt_token_count(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def create_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(f"Worker has no visible CUDA device (count={torch.cuda.device_count()})")
    return LLM(
        model=str(args.model_path),
        tokenizer=str(args.model_path),
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        seed=SEED,
        disable_log_stats=False,
    )


def inference_runtime(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "gpus": args.gpus,
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "batch_size": args.batch_size,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
    }


def enforce_probe_input_boundary() -> None:
    forbidden = {
        "knowledge_rl_validation.jsonl", "math_rl_validation.jsonl",
        "rl_validation.jsonl", "pilot_attempts.jsonl",
    }

    def audit(event: str, values: tuple[Any, ...]) -> None:
        if event == "open" and values and isinstance(values[0], (str, bytes)):
            if os.path.basename(os.fsdecode(values[0])) in forbidden:
                raise RuntimeError(f"Probe-only must not open {os.path.basename(os.fsdecode(values[0]))}")

    sys.addaudithook(audit)


def rollout_aliases(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for alias, original in (("sample_id", "id"), ("raw_output", "generated_text")):
        if alias in row and row[alias] != row[original]:
            raise ValueError(f"Conflicting rollout alias: {alias}")
        result[alias] = row[original]
    return result


def status_update(path: Path, payload: dict[str, Any]) -> None:
    payload = {**payload, "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write_json(path, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run_generation_batches(
    *,
    llm: Any,
    tokenizer: Any,
    candidates: list[Candidate],
    rollout_indices: dict[tuple[str, str], list[int]],
    output_path: Path,
    output_seen: set[tuple[str, str, int]],
    phase: str,
    caps: dict[str, int],
    args: argparse.Namespace,
    status_path: Path,
    pilot_attempt: int | None = None,
) -> int:
    from vllm import SamplingParams

    row_map = candidate_by_key(candidates)
    tasks: list[tuple[Candidate, int, str, bool, int]] = []
    written = 0
    cumulative_generation_seconds = 0.0
    cumulative_output_tokens = 0
    total_missing = sum(
        1
        for key, indices in rollout_indices.items()
        for index in indices
        if (*key, index) not in output_seen
    )
    completed = len(output_seen)

    def emit(batch: list[tuple[Candidate, int, str, bool, int]]) -> None:
        nonlocal written, completed, cumulative_generation_seconds, cumulative_output_tokens
        if not batch:
            return
        prompts = [item[2] for item in batch]
        params = [
            SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=caps[item[0].domain],
                seed=rollout_seed(item[0].domain, item[0].row_id, item[1]),
                skip_special_tokens=True,
            )
            for item in batch
        ]
        generation_started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
        generation_seconds = time.perf_counter() - generation_started
        cumulative_generation_seconds += generation_seconds
        if len(outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} requests")
        result_rows: list[dict[str, Any]] = []
        for task, generated in zip(batch, outputs):
            candidate, rollout_index, prompt, prefix_open, prompt_tokens = task
            if not generated.outputs:
                raise RuntimeError(f"No completion returned for {candidate.domain}/{candidate.row_id}")
            completion = generated.outputs[0].text
            parsed = parse_protocol(prompt, completion)
            answer_extractable, extracted_answer, answer_correct = parse_answer(
                candidate, parsed["final_region"]
            )
            think_text = parsed["think_body"] or ""
            think_tokens = len(tokenizer(think_text, add_special_tokens=False)["input_ids"]) if think_text else 0
            output_tokens = len(generated.outputs[0].token_ids)
            finish_reason = generated.outputs[0].finish_reason
            max_new_tokens = caps[candidate.domain]
            truncated = finish_reason == "length" or output_tokens >= max_new_tokens
            result: dict[str, Any] = {
                "id": candidate.row_id,
                "sample_id": candidate.row_id,
                "source": candidate.row.get("source"),
                "domain": candidate.domain,
                "answer_type": candidate.row.get("answer_type"),
                "canonical_answer": candidate.row.get("canonical_answer"),
                "phase": phase,
                "rollout_index": rollout_index,
                "seed": rollout_seed(candidate.domain, candidate.row_id, rollout_index),
                "max_new_tokens": max_new_tokens,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "generated_text": completion,
                "raw_output": completion,
                "final_region": parsed["final_region"],
                "protocol_valid": parsed["protocol_valid"],
                "protocol_parse_note": parsed["parse_note"],
                "think_mode": parsed["think_mode"],
                "think_tokens": think_tokens,
                "answer_extractable": answer_extractable,
                "extracted_answer": extracted_answer,
                "answer_correct": answer_correct,
                "finish_reason": finish_reason,
                "truncated": truncated,
                "template_prefix_open": prefix_open,
            }
            if pilot_attempt is not None:
                result["pilot_attempt"] = pilot_attempt
            result_rows.append(result)
        append_jsonl(output_path, result_rows)
        for result in result_rows:
            key = (str(result["domain"]), str(result["id"]), int(result["rollout_index"]))
            if key in output_seen:
                raise ValueError(f"Duplicate generated rollout key: {key}")
            output_seen.add(key)
        written += len(result_rows)
        cumulative_output_tokens += sum(row["output_tokens"] for row in result_rows)
        completed += len(result_rows)
        status_update(
            status_path,
            {
                "phase": phase,
                "state": "running",
                "completed_rollouts": completed,
                "new_rollouts_this_run": written,
                "scheduled_missing_rollouts": total_missing,
                "batch_size": len(batch),
                "batch_generation_seconds": generation_seconds,
                "batch_output_tokens": sum(row["output_tokens"] for row in result_rows),
                "batch_output_tokens_per_second": sum(row["output_tokens"] for row in result_rows) / generation_seconds,
                "cumulative_generation_seconds": cumulative_generation_seconds,
                "cumulative_output_tokens": cumulative_output_tokens,
                "cumulative_output_tokens_per_second": cumulative_output_tokens / cumulative_generation_seconds,
            },
        )

    prompt_cache: dict[tuple[str, str], tuple[str, bool, int]] = {}
    for candidate in candidates:
        key = (candidate.domain, candidate.row_id)
        indices = rollout_indices.get(key, [])
        if not indices:
            continue
        if key not in prompt_cache:
            prompt, prefix_open = render_prompt(tokenizer, candidate)
            n_prompt_tokens = prompt_token_count(tokenizer, prompt)
            if n_prompt_tokens + caps[candidate.domain] > args.max_model_len:
                raise ValueError(
                    f"Prompt+generation exceeds max_model_len for {candidate.domain}/{candidate.row_id}: "
                    f"{n_prompt_tokens}+{caps[candidate.domain]}>{args.max_model_len}"
                )
            prompt_cache[key] = (prompt, prefix_open, n_prompt_tokens)
        prompt, prefix_open, n_prompt_tokens = prompt_cache[key]
        for rollout_index in indices:
            if (candidate.domain, candidate.row_id, rollout_index) in output_seen:
                continue
            tasks.append((candidate, rollout_index, prompt, prefix_open, n_prompt_tokens))
            if len(tasks) >= args.batch_size:
                emit(tasks)
                tasks = []
    emit(tasks)
    return written


def assigned_worker(domain: str, row_id: str, worker_count: int) -> int:
    return int.from_bytes(stable_digest(SEED, "gpu_worker", domain, row_id)[:8], "big") % worker_count


def run_worker(args: argparse.Namespace) -> None:
    candidates, _ = load_candidates(args.root)
    task_rows = read_jsonl(args.worker_task_file)
    worker_candidates: list[Candidate] = []
    rollout_indices: dict[tuple[str, str], list[int]] = {}
    candidate_map = candidate_by_key(candidates)
    for task in task_rows:
        key = (str(task["domain"]), str(task["id"]))
        if assigned_worker(*key, args.worker_count) != args.worker_rank:
            continue
        candidate = candidate_map.get(key)
        if candidate is None:
            raise ValueError(f"Task file references a non-candidate: {key}")
        worker_candidates.append(candidate)
        rollout_indices[key] = [int(index) for index in task["rollout_indices"]]

    output_path = args.worker_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    worker_rows, output_seen = load_rollout_index(output_path)
    phase = args.worker_phase
    caps = json.loads(args.worker_caps_json)
    if phase in {"initial", "retest"}:
        central_path = args.output_dir / ("initial_rollouts.jsonl" if phase == "initial" else "retest_rollouts.jsonl")
        task_keys = {
            (domain, row_id, index)
            for (domain, row_id), indices in rollout_indices.items()
            for index in indices
        }
        worker_by_key = {
            (str(row["domain"]), str(row["id"]), int(row["rollout_index"])): row
            for row in worker_rows
        }
        central_additions = []
        for row in iter_jsonl(central_path, recover_tail=True):
            key = (str(row["domain"]), str(row["id"]), int(row["rollout_index"]))
            if key not in task_keys:
                continue
            existing = worker_by_key.get(key)
            if existing is not None:
                if existing != row:
                    raise ValueError(f"Conflicting worker/central checkpoint: {key}")
                continue
            if key not in output_seen:
                central_additions.append(row)
                output_seen.add(key)
                worker_by_key[key] = row
        append_jsonl(output_path, central_additions)
        worker_rows.extend(central_additions)
    if args.worker_reuse_pilot:
        pilot_rows = read_jsonl(args.output_dir / "pilot_rollouts.jsonl")
        pilot_index = {
            (str(row["domain"]), str(row["id"]), int(row["rollout_index"])): row
            for row in pilot_rows
        }
        additions: list[dict[str, Any]] = []
        for candidate in worker_candidates:
            for index in rollout_indices[(candidate.domain, candidate.row_id)]:
                key = (candidate.domain, candidate.row_id, index)
                source = pilot_index.get(key)
                if source is None or int(source["max_new_tokens"]) != int(caps[candidate.domain]):
                    continue
                if key in output_seen:
                    continue
                reused = rollout_aliases(source)
                reused["phase"] = "initial"
                reused["reused_from_pilot"] = True
                additions.append(reused)
                output_seen.add(key)
        append_jsonl(output_path, additions)
        worker_rows.extend(additions)

    status_path = args.worker_output_path.with_suffix(".status.json")
    missing = sum(
        1
        for key, indices in rollout_indices.items()
        for index in indices
        if (*key, index) not in output_seen
    )
    status_update(
        status_path,
        {
            "phase": phase,
            "worker_rank": args.worker_rank,
            "worker_count": args.worker_count,
            "state": "starting" if missing else "complete",
            "scheduled_rollouts": sum(len(v) for v in rollout_indices.values()),
            "missing_rollouts": missing,
        },
    )
    if missing:
        from transformers import AutoTokenizer

        llm = create_llm(args)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
        run_generation_batches(
            llm=llm,
            tokenizer=tokenizer,
            candidates=worker_candidates,
            rollout_indices=rollout_indices,
            output_path=output_path,
            output_seen=output_seen,
            phase=phase,
            caps={key: int(value) for key, value in caps.items()},
            args=args,
            status_path=status_path,
            pilot_attempt=args.worker_attempt if phase == "pilot" else None,
        )
    status_update(
        status_path,
        {
            "phase": phase,
            "worker_rank": args.worker_rank,
            "worker_count": args.worker_count,
            "state": "complete",
            "scheduled_rollouts": sum(len(v) for v in rollout_indices.values()),
            "completed_rollouts": sum(
                (domain, row_id, index) in output_seen
                for (domain, row_id), indices in rollout_indices.items()
                for index in indices
            ),
            "checkpoint_rollouts": len(read_jsonl(output_path, recover_tail=True)),
        },
    )


def run_worker_pool(
    args: argparse.Namespace,
    *,
    phase: str,
    candidates: list[Candidate],
    rollout_indices: dict[tuple[str, str], list[int]],
    caps: dict[str, int],
    attempt: int = 1,
    reuse_pilot: bool = False,
) -> list[dict[str, Any]]:
    task_rows = [
        {"domain": candidate.domain, "id": candidate.row_id, "rollout_indices": rollout_indices[(candidate.domain, candidate.row_id)]}
        for candidate in candidates
        if rollout_indices.get((candidate.domain, candidate.row_id))
    ]
    task_payload = jsonl_bytes(task_rows)
    selection_hash = sha256_bytes(task_payload)[:12]
    task_dir = args.output_dir / "worker_tasks"
    checkpoint_dir = args.output_dir / "worker_checkpoints"
    log_dir = args.output_dir / "worker_logs"
    for path in (task_dir, checkpoint_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    task_path = task_dir / f"{phase}_a{attempt}_{selection_hash}.jsonl"
    atomic_write_bytes(task_path, task_payload)
    caps_json = json.dumps(caps, sort_keys=True)
    ranks = sorted({assigned_worker(row["domain"], row["id"], args.gpus) for row in task_rows})
    processes: dict[int, subprocess.Popen[Any]] = {}
    log_handles: dict[int, Any] = {}
    worker_output_paths = {
        rank: checkpoint_dir / f"{phase}_a{attempt}_{selection_hash}_gpu{rank}.jsonl" for rank in ranks
    }
    script_path = Path(__file__).resolve()
    for rank in ranks:
        output_path = worker_output_paths[rank]
        log_path = log_dir / f"{phase}_a{attempt}_{selection_hash}_gpu{rank}.log"
        command = [
            sys.executable,
            str(script_path),
            "worker",
            "--root", str(args.root),
            "--output-dir", str(args.output_dir),
            "--model-path", str(args.model_path),
            "--gpus", str(args.gpus),
            "--max-model-len", str(args.max_model_len),
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--batch-size", str(args.batch_size),
            "--worker-rank", str(rank),
            "--worker-count", str(args.gpus),
            "--worker-task-file", str(task_path),
            "--worker-output-path", str(output_path),
            "--worker-phase", phase,
            "--worker-attempt", str(attempt),
            "--worker-caps-json", caps_json,
        ]
        if reuse_pilot:
            command.append("--worker-reuse-pilot")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTHONUNBUFFERED"] = "1"
        env["VLLM_LOGGING_LEVEL"] = "INFO"
        handle = log_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(args.root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            handle.close()
            raise
        processes[rank] = process
        log_handles[rank] = handle

    status_path = args.output_dir / "status.json"
    last_report = 0.0
    failure: tuple[int, int] | None = None
    try:
        while any(process.poll() is None for process in processes.values()):
            for rank, process in processes.items():
                code = process.poll()
                if code not in (None, 0):
                    failure = (rank, code)
                    break
            if failure:
                break
            now = time.monotonic()
            if now - last_report >= 10:
                worker_statuses = []
                for rank in ranks:
                    path = worker_output_paths[rank].with_suffix(".status.json")
                    if path.exists():
                        try:
                            worker_statuses.append(json.loads(path.read_text(encoding="utf-8")))
                        except json.JSONDecodeError:
                            pass
                completed = sum(int(row.get("completed_rollouts", 0)) for row in worker_statuses)
                scheduled = sum(int(row.get("scheduled_rollouts", 0)) for row in worker_statuses)
                status_update(
                    status_path,
                    {
                        "phase": phase,
                        "state": "running",
                        "workers": len(ranks),
                        "worker_processes_exited": sum(process.poll() == 0 for process in processes.values()),
                        "completed_rollouts": completed,
                        "scheduled_rollouts": scheduled or sum(len(v) for v in rollout_indices.values()),
                    },
                )
                last_report = now
            time.sleep(2)
        if failure:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
            rank, code = failure
            log_path = log_dir / f"{phase}_a{attempt}_{selection_hash}_gpu{rank}.log"
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:] if log_path.exists() else ""
            raise RuntimeError(f"GPU worker {rank} failed with exit code {code}:\n{tail}")
        failed = {rank: proc.returncode for rank, proc in processes.items() if proc.returncode != 0}
        if failed:
            raise RuntimeError(f"GPU worker exit failure: {failed}")
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
        raise
    finally:
        for handle in log_handles.values():
            handle.close()

    central_path = {
        "pilot": args.output_dir / "pilot_attempts.jsonl",
        "initial": args.output_dir / "initial_rollouts.jsonl",
        "retest": args.output_dir / "retest_rollouts.jsonl",
    }[phase]
    if phase == "pilot":
        existing_rows, _ = load_pilot_attempts(central_path)
        existing_pilot_by_key = {
            (row["domain"], row["id"], int(row["rollout_index"]), int(row["pilot_attempt"])): row
            for row in existing_rows
        }
        merged_new: list[dict[str, Any]] = []
        for rank in ranks:
            shard_rows = read_jsonl(worker_output_paths[rank], recover_tail=True)
            for row in shard_rows:
                key = (row["domain"], row["id"], int(row["rollout_index"]), int(row.get("pilot_attempt", -1)))
                existing = existing_pilot_by_key.get(key)
                if existing is not None:
                    if existing != row:
                        raise ValueError(f"Conflicting duplicate pilot rollout: {key}")
                    continue
                existing_pilot_by_key[key] = row
                merged_new.append(row)
        append_jsonl(central_path, merged_new)
    else:
        existing_rows, existing_keys = load_rollout_index(central_path)
        existing_by_key = {
            (row["domain"], row["id"], int(row["rollout_index"])): row for row in existing_rows
        }
        merged_new = []
        for rank in ranks:
            shard_rows = read_jsonl(worker_output_paths[rank], recover_tail=True)
            for row in shard_rows:
                key = (row["domain"], row["id"], int(row["rollout_index"]))
                if key in existing_keys:
                    if existing_by_key[key] != row:
                        raise ValueError(f"Conflicting duplicate central/shard rollout: {key}")
                    continue
                existing_keys.add(key)
                existing_by_key[key] = row
                merged_new.append(row)
        append_jsonl(central_path, merged_new)

    # The merged JSONL is the durable checkpoint; remove per-worker duplicates.
    for path in worker_output_paths.values():
        path.unlink(missing_ok=True)
        path.with_suffix(".status.json").unlink(missing_ok=True)
    task_path.unlink(missing_ok=True)
    if phase == "pilot":
        merged_rows, _ = load_pilot_attempts(central_path)
    else:
        merged_rows, _ = load_rollout_index(central_path)
    status_update(
        status_path,
        {
            "phase": phase,
            "state": "checkpointed",
            "new_rollouts_merged": len(merged_new),
            "checkpoint_rollouts": len(merged_rows),
            "workers": len(ranks),
        },
    )
    return merged_rows


def deterministic_pilot(candidates: list[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    for domain in ("knowledge", "math"):
        rows = [candidate for candidate in candidates if candidate.domain == domain]
        ranked = sorted(
            rows,
            key=lambda item: (stable_digest(SEED, "pilot", domain, item.row_id), item.row_id),
        )
        selected.extend(ranked[:PILOT_PER_DOMAIN])
    return selected


def pilot_paths(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        output_dir / "pilot_attempts.jsonl",
        output_dir / "pilot_rollouts.jsonl",
        output_dir / "pilot_runtime.json",
        output_dir / "pilot_manifest.json",
    )


def is_clear_think_truncation(row: dict[str, Any]) -> bool:
    if not row.get("truncated") or row.get("think_mode") != "think":
        return False
    note = row.get("protocol_parse_note")
    return note in {"template_prefix_unclosed", "template_prefix_structure_error", "ambiguous_think_boundary"}


def run_pilot(args: argparse.Namespace) -> None:
    self_test()
    candidates, input_manifest = load_candidates(args.root)
    attempts_path, final_path, runtime_path, manifest_path = pilot_paths(args.output_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("gate_passed") is True:
            print(f"Pilot already passed: {manifest_path}", flush=True)
            return
    model_info = model_manifest(args.model_path, args.output_dir / "model_manifest.json")
    packages = package_manifest()
    pilot_candidates = deterministic_pilot(candidates)
    atomic_write_json(
        args.output_dir / "pilot_selection.json",
        {
            "seed": SEED,
            "records_per_domain": PILOT_PER_DOMAIN,
            "ids": {domain: [row.row_id for row in pilot_candidates if row.domain == domain] for domain in ("knowledge", "math")},
        },
    )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {
        "caps": {"knowledge": 3072, "math": 3072},
        "attempts": {"knowledge": 1, "math": 1},
    }
    all_attempts, attempt_keys = load_pilot_attempts(attempts_path)
    status_path = args.output_dir / "status.json"

    while True:
        current_candidates = [row for row in pilot_candidates if runtime["attempts"][row.domain] > 0]
        rollout_indices = {
            (row.domain, row.row_id): list(range(ROLLOUTS_PER_QUESTION)) for row in current_candidates
        }
        # Attempt number is tracked by domain; run one domain at a time so cap changes remain explicit.
        for domain in ("knowledge", "math"):
            domain_rows = [row for row in current_candidates if row.domain == domain]
            attempt = int(runtime["attempts"][domain])
            cap = int(runtime["caps"][domain])
            missing_keys = {
                (row.domain, row.row_id, index)
                for row in domain_rows
                for index in range(ROLLOUTS_PER_QUESTION)
                if (row.domain, row.row_id, index, attempt) not in attempt_keys
            }
            if not missing_keys:
                continue
            domain_indices = {
                (row.domain, row.row_id): list(range(ROLLOUTS_PER_QUESTION)) for row in domain_rows
            }
            run_worker_pool(
                args,
                phase="pilot",
                candidates=domain_rows,
                rollout_indices=domain_indices,
                caps={"knowledge": cap, "math": cap},
                attempt=attempt,
            )
            all_attempts, attempt_keys = load_pilot_attempts(attempts_path)
        all_attempts, attempt_keys = load_pilot_attempts(attempts_path)
        missing_by_domain = {
            domain: [
                (row.row_id, index)
                for row in pilot_candidates
                if row.domain == domain
                for index in range(ROLLOUTS_PER_QUESTION)
                if (domain, row.row_id, index, int(runtime["attempts"][domain])) not in attempt_keys
            ]
            for domain in ("knowledge", "math")
        }
        if any(missing_by_domain.values()):
            atomic_write_json(runtime_path, runtime)
            raise RuntimeError(f"Pilot checkpoint incomplete: { {k: len(v) for k, v in missing_by_domain.items()} }")
        escalated = False
        for domain in ("knowledge", "math"):
            cap = int(runtime["caps"][domain])
            attempt = int(runtime["attempts"][domain])
            current = [
                row for row in all_attempts
                if row.get("domain") == domain and int(row.get("pilot_attempt", -1)) == attempt
            ]
            if len(current) != PILOT_PER_DOMAIN * ROLLOUTS_PER_QUESTION:
                raise RuntimeError(f"Pilot {domain} count mismatch at cap {cap}: {len(current)}")
            if cap == 3072 and any(is_clear_think_truncation(row) for row in current):
                runtime["caps"][domain] = 4096
                runtime["attempts"][domain] = attempt + 1
                escalated = True
        atomic_write_json(runtime_path, runtime)
        if not escalated:
            break

    final_attempt = {domain: int(runtime["attempts"][domain]) for domain in ("knowledge", "math")}
    final_rows = [
        row for row in all_attempts if int(row.get("pilot_attempt", -1)) == final_attempt[str(row["domain"])]
    ]
    final_keys = [(row["domain"], row["id"], int(row["rollout_index"])) for row in final_rows]
    if len(final_rows) != 2 * PILOT_PER_DOMAIN * ROLLOUTS_PER_QUESTION or len(set(final_keys)) != len(final_keys):
        raise RuntimeError(f"Pilot final contract failed: rows={len(final_rows)} unique={len(set(final_keys))}")
    atomic_write_bytes(final_path, jsonl_bytes(final_rows))
    by_domain = collections.Counter(row["domain"] for row in final_rows)
    if by_domain != {"knowledge": 2000, "math": 2000}:
        raise RuntimeError(f"Pilot per-domain count failure: {dict(by_domain)}")
    parser_test_ok = True
    gate = bool(parser_test_ok and len(set(final_keys)) == len(final_rows) and by_domain == {"knowledge": 2000, "math": 2000})
    pilot_summary = {
        "stage": "pilot",
        "records_per_domain": PILOT_PER_DOMAIN,
        "rollouts_per_record": ROLLOUTS_PER_QUESTION,
        "domains": {
            domain: metric_summary([row for row in final_rows if row.get("domain") == domain])
            for domain in ("knowledge", "math")
        },
    }
    atomic_write_json(args.output_dir / "pilot_summary.json", pilot_summary)
    manifest = {
        "stage": "pilot",
        "gate_passed": gate,
        "engineering_gate": {
            "rollouts_complete": True,
            "unique_rollout_keys": True,
            "parser_boundary_tests": "PASS",
        },
        "seed": SEED,
        "prompt_version": PROMPT_VERSION,
        "enable_thinking": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "max_new_tokens_by_domain": runtime["caps"],
        "pilot_attempts_by_domain": runtime["attempts"],
        "records_per_domain": PILOT_PER_DOMAIN,
        "rollouts_per_record": ROLLOUTS_PER_QUESTION,
        "rollout_counts": dict(by_domain),
        "prompt_text": {"knowledge_suffix": PROMPT_KNOWLEDGE_SUFFIX, "math_suffix": PROMPT_MATH_SUFFIX},
        "truncated_rollouts": dict(collections.Counter(
            row["domain"] for row in final_rows if row.get("truncated")
        )),
        "clear_think_truncations": dict(collections.Counter(
            row["domain"] for row in final_rows if is_clear_think_truncation(row)
        )),
        "input_files": input_manifest,
        "model": model_info,
        "packages": packages,
        "args": {"gpus": args.gpus, "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs},
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "batch_size": args.batch_size,
        "runtime": inference_runtime(args),
        "pilot_summary": pilot_summary,
        "pilot_rollouts_sha256": sha256_file(final_path),
    }
    atomic_write_json(manifest_path, manifest)
    status_update(status_path, {"phase": "pilot", "state": "passed" if gate else "failed", "rollouts": len(final_rows)})
    if not gate:
        raise RuntimeError("Pilot engineering gate failed")
    print(f"PILOT_GATE=PASS manifest={manifest_path}", flush=True)


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        return {"rollouts": 0}
    correct = sum(bool(row.get("answer_correct")) for row in rows)
    effective = sum(bool(row.get("answer_correct")) and bool(row.get("protocol_valid")) for row in rows)
    valid = sum(bool(row.get("protocol_valid")) for row in rows)
    extractable = sum(bool(row.get("answer_extractable")) for row in rows)
    think = sum(row.get("think_mode") == "think" for row in rows)
    no_think = sum(row.get("think_mode") == "no_think" for row in rows)
    truncated = sum(bool(row.get("truncated")) for row in rows)
    think_values = [int(row.get("think_tokens", 0)) for row in rows]
    correct_think_values = [int(row.get("think_tokens", 0)) for row in rows if row.get("answer_correct")]
    return {
        "rollouts": total,
        "answer_correct_count": correct,
        "correct_count": correct,
        "answer_solve_rate": correct / total,
        "accuracy": correct / total,
        "effective_success_rate": effective / total,
        "valid_rate": valid / total,
        "protocol_valid_rate": valid / total,
        "answer_extractable_count": extractable,
        "answer_extractable_rate": extractable / total,
        "think_count": think,
        "nothink_count": no_think,
        "think_rate": think / total,
        "nothink_rate": no_think / total,
        "truncated_count": truncated,
        "truncation_rate": truncated / total,
        "truncated_rate": truncated / total,
        "finish_reason_counts": dict(collections.Counter(str(row.get("finish_reason")) for row in rows)),
        "avg_think_tokens": statistics.mean(think_values),
        "median_think_tokens": statistics.median(think_values),
        "p90_think_tokens": percentile(think_values, 0.90),
        "avg_correct_think_tokens": statistics.mean(correct_think_values) if correct_think_values else None,
    }


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stratum_for(candidate: Candidate) -> str:
    if candidate.domain == "knowledge":
        return str(len(candidate.row["options_original"]))
    return str(candidate.row["answer_type"])


def stratum_sort_key(label: str) -> tuple[int, Any]:
    if label.isdecimal():
        return (0, int(label))
    order = {"integer": 0, "decimal": 1, "fraction": 2}
    return (1, order.get(label, 99), label)


def proportional_allocation(counts: collections.Counter[str], sample_size: int) -> dict[str, int]:
    population = sum(counts.values())
    if sample_size > population:
        raise ValueError(f"Cannot allocate {sample_size} from {population}")
    result: dict[str, int] = {}
    remainder: dict[str, int] = {}
    for label, count in counts.items():
        result[label], remainder[label] = divmod(count * sample_size, population)
    need = sample_size - sum(result.values())
    for label in sorted(counts, key=lambda item: (-remainder[item], stratum_sort_key(item))):
        if need <= 0:
            break
        result[label] += 1
        need -= 1
    return result


def choose_difficulty_quotas(bucket_counts: collections.Counter[str], sample_size: int) -> dict[str, int]:
    targets = {bucket: int(round(sample_size * TARGET_BUCKET_RATIOS[bucket])) for bucket in BUCKET_ORDER}
    # Largest-remainder correction keeps the target total exact.
    delta = sample_size - sum(targets.values())
    for bucket in sorted(BUCKET_ORDER, key=lambda item: (-(sample_size * TARGET_BUCKET_RATIOS[item] % 1), BUCKET_ORDER.index(item)))[:delta]:
        targets[bucket] += 1
    quotas = {bucket: min(targets[bucket], bucket_counts[bucket]) for bucket in BUCKET_ORDER}
    remaining = sample_size - sum(quotas.values())
    # Keep B1-B3 dominant, then use B4; B0 may grow only to approximately 10%.
    for bucket in ("B2", "B1", "B3", "B4", "B0"):
        if remaining <= 0:
            break
        cap = min(bucket_counts[bucket], 800 if bucket == "B0" else bucket_counts[bucket])
        add = min(remaining, max(0, cap - quotas[bucket]))
        quotas[bucket] += add
        remaining -= add
    if remaining:
        raise ValueError(
            f"Cannot fill {sample_size} while keeping B0<=800; bucket population={dict(bucket_counts)}"
        )
    return quotas


def select_domain(
    domain: str,
    candidates: list[Candidate],
    summaries: dict[tuple[str, str], dict[str, Any]],
    base_seed: int = SEED,
) -> tuple[list[Candidate], dict[str, Any]]:
    rows = [candidate for candidate in candidates if candidate.domain == domain]
    grouped: dict[str, list[Candidate]] = collections.defaultdict(list)
    for candidate in rows:
        grouped[summaries[(domain, candidate.row_id)]["difficulty_bucket"]].append(candidate)
    bucket_counts = collections.Counter({bucket: len(grouped[bucket]) for bucket in BUCKET_ORDER})
    quotas = choose_difficulty_quotas(bucket_counts, FINAL_PER_DOMAIN)
    selected: list[Candidate] = []
    selected_strata: collections.Counter[str] = collections.Counter()
    for bucket in BUCKET_ORDER:
        group = grouped[bucket]
        quota = quotas[bucket]
        if not quota:
            continue
        strata: dict[str, list[Candidate]] = collections.defaultdict(list)
        for candidate in group:
            strata[stratum_for(candidate)].append(candidate)
        category_targets = proportional_allocation(
            collections.Counter({label: len(values) for label, values in strata.items()}), quota
        )
        for label, category_quota in sorted(category_targets.items(), key=lambda pair: stratum_sort_key(pair[0])):
            ranked = sorted(
                strata[label],
                key=lambda candidate: (
                    stable_digest(base_seed, "final", domain, bucket, label, candidate.row_id),
                    candidate.row_id,
                ),
            )
            chosen = ranked[:category_quota]
            selected.extend(chosen)
            selected_strata[label] += len(chosen)
    if len(selected) != FINAL_PER_DOMAIN or len({candidate.row_id for candidate in selected}) != FINAL_PER_DOMAIN:
        raise AssertionError(f"{domain} final sample count/ID check failed")
    final_bucket_counts = collections.Counter(summaries[(domain, candidate.row_id)]["difficulty_bucket"] for candidate in selected)
    report = {
        "candidate_bucket_counts": {bucket: bucket_counts[bucket] for bucket in BUCKET_ORDER},
        "target_bucket_counts": {bucket: int(round(FINAL_PER_DOMAIN * TARGET_BUCKET_RATIOS[bucket])) for bucket in BUCKET_ORDER},
        "final_bucket_quotas": {bucket: quotas[bucket] for bucket in BUCKET_ORDER},
        "final_bucket_counts": {bucket: final_bucket_counts[bucket] for bucket in BUCKET_ORDER},
        "final_strata_counts": dict(sorted(selected_strata.items(), key=lambda item: stratum_sort_key(item[0]))),
    }
    return selected, report


def summaries_from_rollouts(
    candidates: list[Candidate],
    initial_rows: list[dict[str, Any]],
    retest_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    candidate_map = candidate_by_key(candidates)
    rollouts: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in initial_rows + retest_rows:
        key = (str(row.get("domain")), str(row.get("id")))
        if key not in candidate_map:
            raise ValueError(f"Rollout references non-candidate ID: {key}")
        rollouts[key].append(row)
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate.domain, candidate.row_id)
        sample_rollouts = sorted(rollouts.get(key, []), key=lambda row: int(row["rollout_index"]))
        if len(sample_rollouts) not in {4, 8}:
            raise ValueError(f"Expected 4 or 8 rollouts for {key}; got {len(sample_rollouts)}")
        indices = [int(row["rollout_index"]) for row in sample_rollouts]
        if len(indices) != len(set(indices)) or set(indices) != set(range(len(indices))):
            raise ValueError(f"Unexpected rollout indexes for {key}: {indices}")
        correct = sum(bool(row.get("answer_correct")) for row in sample_rollouts)
        total = len(sample_rollouts)
        initial = [row for row in sample_rollouts if int(row["rollout_index"]) < 4]
        initial_correct = sum(bool(row.get("answer_correct")) for row in initial)
        per_question = {
            "id": candidate.row_id,
            "source": candidate.row.get("source"),
            "domain": candidate.domain,
            "answer_type": candidate.row.get("answer_type"),
            "options_count": len(candidate.row.get("options_original", {})) if candidate.domain == "knowledge" else None,
            "rollouts": total,
            "correct_rollouts": correct,
            "answer_solve_rate": correct / total,
            "initial_correct_rollouts": initial_correct,
            "initial_bucket": bucket_for(initial_correct, 4),
            "difficulty_bucket": bucket_for(correct, total),
            "metrics": metric_summary(sample_rollouts),
            "retested": total == 8,
        }
        summaries[key] = per_question
    return summaries, rollouts


def write_question_summaries(
    output_dir: Path,
    candidates: list[Candidate],
    summaries: dict[tuple[str, str], dict[str, Any]],
    selected_ids: set[tuple[str, str]] | None = None,
) -> None:
    selected_ids = selected_ids or set()
    for domain, name in (("knowledge", "knowledge_probe_summary.jsonl"), ("math", "math_probe_summary.jsonl")):
        rows = []
        for candidate in candidates:
            if candidate.domain != domain:
                continue
            row = dict(summaries[(domain, candidate.row_id)])
            row["selected_for_final"] = (domain, candidate.row_id) in selected_ids
            rows.append(row)
        atomic_write_bytes(output_dir / name, jsonl_bytes(rows))


def aggregate_domain_rollouts(
    domain: str,
    summaries: dict[tuple[str, str], dict[str, Any]],
    rollouts: dict[tuple[str, str], list[dict[str, Any]]],
    selected_ids: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    keys = [key for key in summaries if key[0] == domain and (selected_ids is None or key in selected_ids)]
    flat = [row for key in keys for row in rollouts[key]]
    buckets = collections.Counter(summaries[key]["difficulty_bucket"] for key in keys)
    return {
        "questions": len(keys),
        "difficulty_bucket_counts": {bucket: buckets[bucket] for bucket in BUCKET_ORDER},
        **metric_summary(flat),
    }


def final_paths(root: Path) -> dict[str, Path]:
    return {
        "knowledge": root / "Nemotron-RL-knowledge-mcqa" / "processed_data" / "knowledge_rl_train_final.jsonl",
        "math": root / "AceReason-Math" / "processed_data" / "math_rl_train_final.jsonl",
        "merged": root / "processed_data" / "rl_train_final.jsonl",
    }


def finalize_outputs(
    args: argparse.Namespace,
    candidates: list[Candidate],
    initial_rows: list[dict[str, Any]],
    retest_rows: list[dict[str, Any]],
    pilot_manifest: dict[str, Any],
    input_manifest: dict[str, Any],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    summaries, rollouts = summaries_from_rollouts(candidates, initial_rows, retest_rows)
    selected_by_domain: dict[str, list[Candidate]] = {}
    selection_report: dict[str, Any] = {}
    for domain in ("knowledge", "math"):
        selected_by_domain[domain], selection_report[domain] = select_domain(domain, candidates, summaries)
    selected_ids = {
        (candidate.domain, candidate.row_id)
        for rows in selected_by_domain.values()
        for candidate in rows
    }
    write_question_summaries(args.output_dir, candidates, summaries, selected_ids)

    selection_payload = {
        "seed": SEED,
        "target_per_domain": FINAL_PER_DOMAIN,
        "selected_ids": {
            domain: [candidate.row_id for candidate in selected_by_domain[domain]]
            for domain in ("knowledge", "math")
        },
        "report": selection_report,
    }
    atomic_write_json(args.output_dir / "final_selection.json", selection_payload)

    paths = final_paths(args.root)
    for domain, key in (("knowledge", "knowledge"), ("math", "math")):
        chosen = {(candidate.domain, candidate.row_id) for candidate in selected_by_domain[domain]}
        content = b"".join(
            candidate.raw_line
            for candidate in candidates
            if (candidate.domain, candidate.row_id) in chosen
        )
        atomic_write_bytes(paths[key], content, refuse_different=True)
    merged = paths["knowledge"].read_bytes() + paths["math"].read_bytes()
    atomic_write_bytes(paths["merged"], merged, refuse_different=True)

    all_rows_by_key = {
        (row["domain"], row["id"], int(row["rollout_index"])): row
        for row in initial_rows + retest_rows
    }
    ordered_all = []
    ordered_selected = []
    for candidate in candidates:
        n = summaries[(candidate.domain, candidate.row_id)]["rollouts"]
        for index in range(n):
            row = all_rows_by_key[(candidate.domain, candidate.row_id, index)]
            ordered_all.append(row)
            if (candidate.domain, candidate.row_id) in selected_ids:
                ordered_selected.append(row)
    atomic_write_bytes(args.output_dir / "rollout_detail.jsonl", jsonl_bytes(ordered_all))

    recheck_transitions = collections.Counter()
    for key, summary in summaries.items():
        if summary["retested"]:
            recheck_transitions[f"{summary['initial_bucket']}->{summary['difficulty_bucket']}"] += 1
    report = {
        "seed": SEED,
        "prompt_version": PROMPT_VERSION,
        "enable_thinking": True,
        "difficulty_definition": "answer_correct_count / actual_rollouts",
        "valid_rate_definition": "protocol_valid / rollouts",
        "effective_success_rate_definition": "answer_correct and protocol_valid / rollouts",
        "difficulty_buckets": {
            "4_rollouts": {"B0": "0/4", "B1": "1/4", "B2": "2/4", "B3": "3/4", "B4": "4/4"},
            "8_rollouts": {"B0": "0/8", "B1": "1-2/8", "B2": "3-5/8", "B3": "6-7/8", "B4": "8/8"},
        },
        "domains": {
            domain: {
                "all_candidates": aggregate_domain_rollouts(domain, summaries, rollouts),
                "final_selected": aggregate_domain_rollouts(domain, summaries, rollouts, selected_ids),
                "selection": selection_report[domain],
            }
            for domain in ("knowledge", "math")
        },
        "b0_b4_recheck_bucket_transitions": dict(sorted(recheck_transitions.items())),
        "retested_questions": sum(recheck_transitions.values()),
        "final_selected_rollouts": metric_summary(ordered_selected),
        "input_files": input_manifest,
        "final_files": {key: {"path": str(path), "records": 16_000 if key == "merged" else FINAL_PER_DOMAIN, "sha256": sha256_file(path)} for key, path in paths.items()},
    }
    atomic_write_json(args.output_dir / "probe_report.json", report)
    manifest = {
        "stage": "complete",
        "seed": SEED,
        "prompt_version": PROMPT_VERSION,
        "enable_thinking": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "prompt_text": {"knowledge_suffix": PROMPT_KNOWLEDGE_SUFFIX, "math_suffix": PROMPT_MATH_SUFFIX},
        "max_new_tokens_by_domain": pilot_manifest["max_new_tokens_by_domain"],
        "rollouts_per_candidate": ROLLOUTS_PER_QUESTION,
        "candidate_counts": {"knowledge": 11_754, "math": 12_512},
        "final_counts": {"knowledge": FINAL_PER_DOMAIN, "math": FINAL_PER_DOMAIN, "merged": 2 * FINAL_PER_DOMAIN},
        "pilot_manifest": str(args.output_dir / "pilot_manifest.json"),
        "input_files": input_manifest,
        "model": model_info,
        "packages": package_manifest(),
        "runtime": inference_runtime(args),
        "pilot_runtime": pilot_manifest.get("runtime", {
            **pilot_manifest.get("args", {}),
            "gpu_memory_utilization": pilot_manifest.get("gpu_memory_utilization"),
            "batch_size": pilot_manifest.get("batch_size"),
        }),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "selection": selection_report,
        "recheck_bucket_transitions": dict(sorted(recheck_transitions.items())),
        "outputs": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()},
    }
    atomic_write_json(args.output_dir / "probe_manifest.json", manifest)
    return report


def check_initial_rollouts(
    candidates: list[Candidate], rows: list[dict[str, Any]],
    tokenizer: Any, *, complete: bool,
) -> dict[str, Any]:
    candidate_map = candidate_by_key(candidates)
    prompt_hashes: dict[tuple[str, str], str] = {}
    keys: set[tuple[str, str, int]] = set()
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    domains: collections.Counter[str] = collections.Counter()
    for row in rows:
        sample = (str(row.get("domain")), str(row.get("id")))
        candidate = candidate_map.get(sample)
        index = row.get("rollout_index")
        if candidate is None or type(index) is not int or index not in range(4):
            raise ValueError(f"Non-candidate or invalid first-round rollout: {sample}/{index}")
        key = (*sample, index)
        if key in keys:
            raise ValueError(f"Duplicate first-round rollout key: {key}")
        keys.add(key)
        counts[sample] += 1
        domains[sample[0]] += 1
        if int(row.get("max_new_tokens", -1)) != 4096:
            raise ValueError(f"Non-4096 rollout: {key}")
        if row.get("seed") != rollout_seed(*key):
            raise ValueError(f"Rollout seed mismatch: {key}")
        if row.get("canonical_answer") != candidate.row["canonical_answer"]:
            raise ValueError(f"Canonical answer mismatch: {key}")
        normalized = rollout_aliases(row)
        if not isinstance(normalized["raw_output"], str):
            raise ValueError(f"Non-text output: {key}")
        for field in ("protocol_valid", "answer_extractable", "answer_correct", "truncated"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"Invalid boolean {field}: {key}")
        if row.get("think_mode") not in {"think", "no_think"}:
            raise ValueError(f"Invalid Think mode: {key}")
        if type(row.get("think_tokens")) is not int or row["think_tokens"] < 0:
            raise ValueError(f"Invalid thinking tokens: {key}")
        if row.get("finish_reason") not in {"stop", "length"}:
            raise ValueError(f"Unexpected finish reason: {key}")
        if sample not in prompt_hashes:
            prompt, _ = render_prompt(tokenizer, candidate)
            if prompt_token_count(tokenizer, prompt) + 4096 > 8192:
                raise ValueError(f"Candidate exceeds the approved context budget: {sample}")
            prompt_hashes[sample] = sha256_bytes(prompt.encode("utf-8"))
        if row.get("prompt_sha256") != prompt_hashes[sample]:
            raise ValueError(f"Rendered prompt mismatch: {key}")
    if complete and (
        len(rows) != len(candidates) * 4
        or set(counts) != set(candidate_map)
        or any(count != 4 for count in counts.values())
    ):
        raise ValueError("First-round coverage must be exactly four rollouts for every candidate")
    return {
        "rollouts": len(rows), "unique_rollout_keys": len(keys),
        "domain_rollouts": dict(domains), "questions_with_four_rollouts": sum(n == 4 for n in counts.values()),
        "missing_rollouts": len(candidates) * 4 - len(rows),
        "duplicate_rollout_keys": 0, "non_candidate_rollouts": 0,
        "non_4096_rollouts": 0, "validation_files_opened": 0,
        "input_boundary_enforced": True, "complete": complete,
    }


def probe_question_summary(candidate: Candidate, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metric_summary(rows)
    correct = metrics["correct_count"]
    return {
        "id": candidate.row_id, "sample_id": candidate.row_id,
        "domain": candidate.domain, "source": candidate.row.get("source"),
        "answer_type": candidate.row.get("answer_type"),
        "options_count": len(candidate.row.get("options_bo", {})) if candidate.domain == "knowledge" else None,
        "rollouts": 4, "correct_count": correct, "correct_rollouts": correct,
        "difficulty_bucket": bucket_for(correct, 4),
        **{name: metrics[name] for name in (
            "answer_solve_rate", "answer_extractable_rate", "protocol_valid_rate", "truncated_rate",
            "nothink_rate", "avg_think_tokens", "median_think_tokens", "avg_correct_think_tokens",
        )},
        "metrics": metrics,
    }


def probe_group_report(
    question_summaries: list[dict[str, Any]], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    buckets = {summary["id"]: summary["difficulty_bucket"] for summary in question_summaries}
    grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKET_ORDER}
    for row in rows:
        grouped[buckets[row["id"]]].append(row)
    counts = collections.Counter(buckets.values())
    total = len(question_summaries)
    return {
        "questions": total, **metric_summary(rows),
        "difficulty_bucket_counts": {b: counts[b] for b in BUCKET_ORDER},
        "difficulty_bucket_proportions": {b: counts[b] / total if total else 0 for b in BUCKET_ORDER},
        "by_difficulty_bucket": {
            b: {"questions": counts[b], "question_proportion": counts[b] / total if total else 0,
                **metric_summary(grouped[b])} for b in BUCKET_ORDER
        },
    }


def probe_report_markdown(report: dict[str, Any]) -> str:
    lines = ["# 全量首轮 RL Probe", "", "只包含 G=4 首轮结果；没有复核、抽样或训练。", ""]
    for domain in ("knowledge", "math", "combined"):
        group = report[domain]
        lines.extend([
            f"## {domain}", "",
            f"题数 {group['questions']:,}；rollout {group['rollouts']:,}。",
            f"答案正确率 {group['answer_solve_rate']:.2%}；可提取率 {group['answer_extractable_rate']:.2%}；"
            f"协议有效率 {group['protocol_valid_rate']:.2%}；截断率 {group['truncated_rate']:.2%}。",
            f"Think {group['think_count']:,}（{group['think_rate']:.2%}）；"
            f"NoThink {group['nothink_count']:,}（{group['nothink_rate']:.2%}）。",
            f"Thinking tokens mean {group['avg_think_tokens']:.2f}；"
            f"median {group['median_think_tokens']:.2f}；P90 {group['p90_think_tokens']:.2f}。", "",
            "| 桶 | 题数 | 占比 | Think | NoThink | tokens mean | median | P90 | 协议有效 | 可提取 | 截断 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for bucket in BUCKET_ORDER:
            part = group["by_difficulty_bucket"][bucket]
            if not part["questions"]:
                lines.append(f"| {bucket} | 0 | 0% | — | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {bucket} | {part['questions']:,} | {part['question_proportion']:.2%} | "
                f"{part['think_rate']:.2%} | {part['nothink_rate']:.2%} | {part['avg_think_tokens']:.2f} | "
                f"{part['median_think_tokens']:.2f} | {part['p90_think_tokens']:.2f} | "
                f"{part['protocol_valid_rate']:.2%} | {part['answer_extractable_rate']:.2%} | {part['truncated_rate']:.2%} |"
            )
        lines.append("")
    lines.extend([
        "## 验收", "", "每题严格四条，rollout key 无重复；所有记录输出预算为 4096。",
        "validation 和 pilot_attempts 文件由输入边界守卫禁止打开。",
        "P90 使用排序后的线性插值；thinking tokens 统计包含 NoThink 的零值。",
        "各难度桶仅由 answer_correct 决定。", "", "## 结果文件", "",
    ])
    lines.extend(f"- `{name}`：`{value}`" for name, value in report["output_paths"].items())
    return "\n".join(lines) + "\n"


def finalize_probe_only(
    args: argparse.Namespace, candidates: list[Candidate], rows: list[dict[str, Any]],
    run_manifest: dict[str, Any], tokenizer: Any,
) -> dict[str, Any]:
    checks = check_initial_rollouts(candidates, rows, tokenizer, complete=True)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["id"])].append(rollout_aliases(row))
    ordered: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        part = sorted(grouped[(candidate.domain, candidate.row_id)], key=lambda row: row["rollout_index"])
        ordered.extend(part)
        summaries.append(probe_question_summary(candidate, part))
    paths = {
        "rollouts": args.output_dir / "probe_rollouts.jsonl",
        "knowledge_rollouts": args.output_dir / "knowledge_probe_rollouts.jsonl",
        "math_rollouts": args.output_dir / "math_probe_rollouts.jsonl",
        "summary": args.output_dir / "probe_summary.jsonl",
        "knowledge_summary": args.output_dir / "knowledge_probe_summary.jsonl",
        "math_summary": args.output_dir / "math_probe_summary.jsonl",
        "report": args.output_dir / "probe_report.json",
        "report_markdown": args.output_dir / "probe_report.md",
    }
    atomic_write_bytes(paths["rollouts"], jsonl_bytes(ordered))
    atomic_write_bytes(paths["summary"], jsonl_bytes(summaries))
    report: dict[str, Any] = {
        "stage": "probe_only", "status": "complete", "seed": SEED,
        "verification": checks, "output_paths": {name: str(path) for name, path in paths.items()},
        "difficulty_definition": "B0=0/4; B1=1/4; B2=2/4; B3=3/4; B4=4/4; answer_correct only",
        "p90_method": "linear interpolation; NoThink zeros included",
    }
    for domain in ("knowledge", "math"):
        domain_rows = [row for row in ordered if row["domain"] == domain]
        domain_summaries = [row for row in summaries if row["domain"] == domain]
        atomic_write_bytes(paths[f"{domain}_rollouts"], jsonl_bytes(domain_rows))
        atomic_write_bytes(paths[f"{domain}_summary"], jsonl_bytes(domain_summaries))
        report[domain] = probe_group_report(domain_summaries, domain_rows)
    report["combined"] = probe_group_report(summaries, ordered)
    report["pilot_reused_rollouts"] = sum(bool(row.get("reused_from_pilot")) for row in ordered)
    if report["pilot_reused_rollouts"] != 4000:
        raise ValueError("Expected exactly 4,000 final Pilot rollouts to be reused")
    atomic_write_json(paths["report"], report)
    atomic_write_bytes(paths["report_markdown"], probe_report_markdown(report).encode("utf-8"))
    manifest = {
        **run_manifest, "status": "complete", "verification": checks,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
    }
    atomic_write_json(args.output_dir / "probe_manifest.json", manifest)
    status_update(args.output_dir / "status.json", {
        "phase": "probe_only_complete", "state": "passed", "completed_rollouts": len(rows),
        "expected_rollouts": len(candidates) * 4, "domain_rollouts": checks["domain_rollouts"],
    })
    return report


def run_probe_only(args: argparse.Namespace) -> None:
    self_test()
    candidates, inputs = load_candidates(args.root)
    pilot_manifest_path = args.output_dir / "pilot_manifest.json"
    pilot_path = args.output_dir / "pilot_rollouts.jsonl"
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    if not pilot_manifest.get("gate_passed") or pilot_manifest.get("input_files") != inputs:
        raise ValueError("Final Pilot gate/input manifest mismatch")
    caps = {"knowledge": 4096, "math": 4096}
    if pilot_manifest.get("max_new_tokens_by_domain") != caps:
        raise ValueError("Probe-only requires the final 4096-token Pilot")
    model_info = model_manifest(args.model_path, args.output_dir / "model_manifest.json")
    if pilot_manifest.get("model") != model_info:
        raise ValueError("Checkpoint/tokenizer differ from final Pilot")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    pilot_rows = read_jsonl(pilot_path)
    check_initial_rollouts(candidates, pilot_rows, tokenizer, complete=False)
    pilot_counts = collections.Counter(row["domain"] for row in pilot_rows)
    if pilot_counts != {"knowledge": 2000, "math": 2000}:
        raise ValueError(f"Malformed final Pilot: {dict(pilot_counts)}")
    pilot_sample_counts = collections.Counter((row["domain"], row["id"]) for row in pilot_rows)
    if len(pilot_sample_counts) != 1000 or any(count != 4 for count in pilot_sample_counts.values()):
        raise ValueError("Final Pilot must have 500 questions/domain with exactly four rollouts")
    contract = {
        "stage": "probe_only", "input_files": inputs, "model": model_info,
        "seed": SEED, "enable_thinking": True, "prompt_version": PROMPT_VERSION,
        "prompt_text": {"knowledge_suffix": PROMPT_KNOWLEDGE_SUFFIX, "math_suffix": PROMPT_MATH_SUFFIX},
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 4096},
        "runtime": inference_runtime(args), "rollouts_per_candidate": 4,
        "pilot_rollouts_sha256": sha256_file(pilot_path),
        "pilot_manifest_sha256": sha256_file(pilot_manifest_path),
        "expected_rollouts": {"knowledge": 47016, "math": 50048, "combined": 97064},
        "input_boundary": {"validation_read": False, "pilot_attempts_read": False, "audit_guard": True},
        "excluded_stages": ["extreme_bucket_retest", "difficulty_sampling", "LCoT2Tree", "structure_cost", "training"],
    }
    run_path = args.output_dir / "probe_only_run_manifest.json"
    if run_path.exists():
        original = json.loads(run_path.read_text(encoding="utf-8"))
        if any(original.get(key) != value for key, value in contract.items()):
            raise ValueError("Probe-only resume contract changed")
        run_manifest = original
    else:
        run_manifest = {
            **contract, "status": "running", "script_sha256": sha256_file(Path(__file__)),
            "packages": package_manifest(),
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_json(run_path, run_manifest)
    initial_path = args.output_dir / "initial_rollouts.jsonl"
    rows, seen = load_rollout_index(initial_path)
    check_initial_rollouts(candidates, rows, tokenizer, complete=False)
    reused: list[dict[str, Any]] = []
    existing = {(row["domain"], row["id"], row["rollout_index"]): row for row in rows}
    for source in pilot_rows:
        key = (source["domain"], source["id"], source["rollout_index"])
        row = rollout_aliases(source)
        row.update(phase="initial", reused_from_pilot=True)
        if key in seen:
            if rollout_aliases(existing[key]) != row:
                raise ValueError(f"Checkpoint conflicts with reusable final Pilot: {key}")
            continue
        reused.append(row)
        seen.add(key)
    append_jsonl(initial_path, reused)
    rows.extend(reused)
    atomic_write_json(args.output_dir / "probe_manifest.json", {
        **run_manifest, "status": "running", "pilot_reused_rollouts": 4000,
        "initial_checkpoint_rollouts": len(rows), "remaining_rollouts_at_launch": 97064 - len(rows),
    })
    if len(rows) != 97064:
        run_worker_pool(
            args, phase="initial", candidates=candidates,
            rollout_indices={(candidate.domain, candidate.row_id): list(range(4)) for candidate in candidates},
            caps=caps, reuse_pilot=True,
        )
    rows, _ = load_rollout_index(initial_path)
    _, current_inputs = load_candidates(args.root)
    if current_inputs != inputs:
        raise ValueError("Candidate inputs changed during Probe")
    if sha256_file(pilot_path) != contract["pilot_rollouts_sha256"]:
        raise ValueError("Final Pilot changed during Probe")
    if sha256_file(pilot_manifest_path) != contract["pilot_manifest_sha256"]:
        raise ValueError("Pilot manifest changed during Probe")
    if model_manifest(args.model_path, args.output_dir / "model_manifest.json") != model_info:
        raise ValueError("Checkpoint/tokenizer changed during Probe")
    report = finalize_probe_only(args, candidates, rows, run_manifest, tokenizer)
    print(json.dumps({"stage": "probe_only", "status": "complete", "verification": report["verification"]}, ensure_ascii=False), flush=True)


def verify_probe_only(args: argparse.Namespace) -> None:
    candidates, inputs = load_candidates(args.root)
    manifest = json.loads((args.output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "probe_only" or manifest.get("status") != "complete":
        raise ValueError("Probe-only results are not complete")
    if manifest.get("input_files") != inputs:
        raise ValueError("Input hashes changed")
    if model_manifest(args.model_path, args.output_dir / "model_manifest.json") != manifest.get("model"):
        raise ValueError("Checkpoint/tokenizer no longer match the Probe manifest")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    output = manifest["outputs"]
    for artifact in output.values():
        if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"Result hash mismatch: {artifact['path']}")
    rows = read_jsonl(Path(output["rollouts"]["path"]))
    checks = check_initial_rollouts(candidates, rows, tokenizer, complete=True)
    by_sample: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if "sample_id" not in row or "raw_output" not in row:
            raise ValueError("Required output aliases are missing")
        by_sample[(row["domain"], row["id"])].append(row)
    pilot_path = args.output_dir / "pilot_rollouts.jsonl"
    if sha256_file(pilot_path) != manifest["pilot_rollouts_sha256"]:
        raise ValueError("Final Pilot hash changed")
    if sha256_file(args.output_dir / "pilot_manifest.json") != manifest["pilot_manifest_sha256"]:
        raise ValueError("Pilot manifest hash changed")
    expected_pilot = {}
    for source in read_jsonl(pilot_path):
        normalized = rollout_aliases(source)
        normalized.update(phase="initial", reused_from_pilot=True)
        expected_pilot[(source["domain"], source["id"], source["rollout_index"])] = normalized
    actual_pilot = {
        (row["domain"], row["id"], row["rollout_index"]): row
        for row in rows if row.get("reused_from_pilot")
    }
    if len(expected_pilot) != 4000 or expected_pilot != actual_pilot:
        raise ValueError("Reusable 4096 Pilot completions are not preserved exactly")
    summaries = read_jsonl(Path(output["summary"]["path"]))
    if len(summaries) != len(candidates) or len({row["id"] for row in summaries}) != len(candidates):
        raise ValueError("Summary coverage/uniqueness mismatch")
    expected_summaries = [probe_question_summary(c, by_sample[(c.domain, c.row_id)]) for c in candidates]
    if summaries != expected_summaries:
        raise ValueError("Question summary does not match rollouts")
    for domain in ("knowledge", "math"):
        if read_jsonl(Path(output[f"{domain}_rollouts"]["path"])) != [r for r in rows if r["domain"] == domain]:
            raise ValueError(f"{domain} rollout export mismatch")
        if read_jsonl(Path(output[f"{domain}_summary"]["path"])) != [r for r in summaries if r["domain"] == domain]:
            raise ValueError(f"{domain} summary export mismatch")
    report = json.loads(Path(output["report"]["path"]).read_text(encoding="utf-8"))
    for domain in ("knowledge", "math", "combined"):
        selected_summaries = summaries if domain == "combined" else [s for s in summaries if s["domain"] == domain]
        selected_rows = rows if domain == "combined" else [r for r in rows if r["domain"] == domain]
        if report[domain] != probe_group_report(selected_summaries, selected_rows):
            raise ValueError(f"{domain} report mismatch")
    atomic_write_json(args.output_dir / "probe_verification.json", {"verification": "PASS", **checks})
    print(json.dumps({"verification": "PASS", **checks}, ensure_ascii=False), flush=True)


def run_full(args: argparse.Namespace) -> None:
    self_test()
    candidates, input_manifest = load_candidates(args.root)
    attempts_path, pilot_rollouts_path, _, pilot_manifest_path = pilot_paths(args.output_dir)
    if not pilot_manifest_path.is_file():
        raise FileNotFoundError(f"Pilot manifest missing: {pilot_manifest_path}")
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    if pilot_manifest.get("gate_passed") is not True:
        raise RuntimeError("Pilot engineering gate did not pass")
    if pilot_manifest.get("input_files") != input_manifest:
        raise RuntimeError("Candidate inputs differ from the inputs used by Pilot")
    model_info = model_manifest(args.model_path, args.output_dir / "model_manifest.json")
    if pilot_manifest.get("model") != model_info:
        raise RuntimeError("Model/tokenizer files changed after Pilot")

    initial_path = args.output_dir / "initial_rollouts.jsonl"
    retest_path = args.output_dir / "retest_rollouts.jsonl"
    initial_rows, initial_seen = load_rollout_index(initial_path)
    retest_rows, retest_seen = load_rollout_index(retest_path)
    pilot_rows = read_jsonl(pilot_rollouts_path)
    selected_pilot_ids = {
        (row["domain"], row["id"]) for row in pilot_rows
    }
    if len(selected_pilot_ids) != 2 * PILOT_PER_DOMAIN:
        raise RuntimeError(f"Pilot sample selection malformed: {len(selected_pilot_ids)} IDs")
    caps = {domain: int(pilot_manifest["max_new_tokens_by_domain"][domain]) for domain in ("knowledge", "math")}

    # Reuse the exact Pilot completions for the same candidate IDs and rollout seeds.
    pilot_by_key = {
        (row["domain"], row["id"], int(row["rollout_index"])): row for row in pilot_rows
    }
    reused = []
    for candidate in candidates:
        for index in range(ROLLOUTS_PER_QUESTION):
            key = (candidate.domain, candidate.row_id, index)
            pilot_row = pilot_by_key.get(key)
            if pilot_row is None or key in initial_seen:
                continue
            if int(pilot_row["max_new_tokens"]) != caps[candidate.domain]:
                continue
            reused_row = dict(pilot_row)
            reused_row["phase"] = "initial"
            reused_row["reused_from_pilot"] = True
            reused.append(reused_row)
            initial_seen.add(key)
    append_jsonl(initial_path, reused)
    initial_rows.extend(reused)

    status_path = args.output_dir / "status.json"
    full_indices = {
        (candidate.domain, candidate.row_id): list(range(ROLLOUTS_PER_QUESTION)) for candidate in candidates
    }
    run_worker_pool(
        args,
        phase="initial",
        candidates=candidates,
        rollout_indices=full_indices,
        caps=caps,
        attempt=1,
        reuse_pilot=True,
    )
    initial_rows, initial_seen = load_rollout_index(initial_path)
    expected_initial = sum(1 for candidate in candidates) * ROLLOUTS_PER_QUESTION
    if len(initial_rows) != expected_initial:
        raise RuntimeError(f"Initial full probe incomplete: {len(initial_rows)} != {expected_initial}")

    # Finish any four-rollout retest partially checkpointed by an interrupted run
    # before computing per-question summaries (which require exactly 4 or 8 rows).
    retest_rows, retest_seen = load_rollout_index(retest_path)
    retested_keys = {(str(row["domain"]), str(row["id"])) for row in retest_rows}
    retest_counts = collections.Counter((str(row["domain"]), str(row["id"])) for row in retest_rows)
    incomplete_retests = [
        candidate
        for candidate in candidates
        if (candidate.domain, candidate.row_id) in retested_keys
        and retest_counts[(candidate.domain, candidate.row_id)] < 4
    ]
    if incomplete_retests:
        run_worker_pool(
            args,
            phase="retest",
            candidates=incomplete_retests,
            rollout_indices={
                (candidate.domain, candidate.row_id): list(range(4, 8))
                for candidate in incomplete_retests
            },
            caps=caps,
            attempt=1,
        )

    iteration = 0
    while True:
        retest_rows, retest_seen = load_rollout_index(retest_path)
        summaries, _ = summaries_from_rollouts(candidates, initial_rows, retest_rows)
        selected_by_domain: dict[str, list[Candidate]] = {}
        selection_report: dict[str, Any] = {}
        for domain in ("knowledge", "math"):
            selected_by_domain[domain], selection_report[domain] = select_domain(domain, candidates, summaries)
        selected_keys = {
            (candidate.domain, candidate.row_id)
            for domain_rows in selected_by_domain.values()
            for candidate in domain_rows
        }
        pending = []
        for candidate in candidates:
            key = (candidate.domain, candidate.row_id)
            summary = summaries[key]
            if key not in selected_keys or summary["rollouts"] == 8:
                continue
            if summary["initial_bucket"] in {"B0", "B4"}:
                pending.append(candidate)
        atomic_write_json(
            args.output_dir / "selection_progress.json",
            {
                "iteration": iteration,
                "selected_counts": {domain: len(selected_by_domain[domain]) for domain in ("knowledge", "math")},
                "pending_extreme_retests": {domain: sum(row.domain == domain for row in pending) for domain in ("knowledge", "math")},
                "bucket_counts": selection_report,
            },
        )
        if not pending:
            report = finalize_outputs(
                args,
                candidates,
                initial_rows,
                retest_rows,
                pilot_manifest,
                input_manifest,
                model_info,
            )
            status_update(status_path, {"phase": "complete", "state": "passed", "selected": {"knowledge": 8000, "math": 8000}})
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            return
        iteration += 1
        pending_indices = {
            (candidate.domain, candidate.row_id): list(range(4, 8)) for candidate in pending
        }
        run_worker_pool(
            args,
            phase="retest",
            candidates=pending,
            rollout_indices=pending_indices,
            caps=caps,
            attempt=iteration,
        )
        initial_rows, initial_seen = load_rollout_index(initial_path)


def verify_local(args: argparse.Namespace) -> None:
    candidates, input_manifest = load_candidates(args.root)
    output_dir = args.output_dir
    initial_rows, initial_keys = load_rollout_index(output_dir / "initial_rollouts.jsonl")
    retest_rows, retest_keys = load_rollout_index(output_dir / "retest_rollouts.jsonl")
    detail_rows, detail_keys = load_rollout_index(output_dir / "rollout_detail.jsonl")
    if len(initial_rows) != len(candidates) * 4:
        raise AssertionError(f"Initial rollout count mismatch: {len(initial_rows)}")
    if not initial_keys.isdisjoint(retest_keys):
        raise AssertionError("Initial/retest rollout keys overlap")
    candidate_keys = {(candidate.domain, candidate.row_id) for candidate in candidates}
    expected_detail_count = len(initial_rows) + len(retest_rows)
    if len(detail_rows) != expected_detail_count or len(detail_keys) != expected_detail_count:
        raise AssertionError("Merged rollout detail count/uniqueness mismatch")
    by_key = candidate_by_key(candidates)
    selected: dict[str, set[str]] = {}
    for domain, key in (("knowledge", "knowledge"), ("math", "math")):
        path = final_paths(args.root)[key]
        rows = read_jsonl(path)
        if len(rows) != FINAL_PER_DOMAIN:
            raise AssertionError(f"{domain} final count mismatch: {len(rows)}")
        ids = [str(row.get("id")) for row in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Duplicate IDs in {domain} final file")
        selected[domain] = set(ids)
        for row in rows:
            candidate = by_key.get((domain, str(row["id"])))
            if candidate is None:
                raise AssertionError(f"Output ID not in candidates: {domain}/{row['id']}")
            expected = json.loads(candidate.raw_line.decode("utf-8"))
            if row != expected:
                raise AssertionError(f"Final row fields changed: {domain}/{row['id']}")
    merged = read_jsonl(final_paths(args.root)["merged"])
    if len(merged) != 2 * FINAL_PER_DOMAIN:
        raise AssertionError(f"Merged final count mismatch: {len(merged)}")
    merged_ids = [str(row.get("id")) for row in merged]
    if len(merged_ids) != len(set(merged_ids)):
        raise AssertionError("Merged final IDs are not unique")
    if set(merged_ids) != selected["knowledge"] | selected["math"]:
        raise AssertionError("Merged final ID union mismatch")
    if selected["knowledge"] & selected["math"]:
        raise AssertionError("Knowledge/Math final ID overlap")
    manifest = json.loads((output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("input_files") != input_manifest:
        raise AssertionError("Input hashes differ from manifest")
    print(
        json.dumps(
            {
                "verification": "PASS",
                "candidate_counts": dict(collections.Counter(candidate.domain for candidate in candidates)),
                "initial_rollouts": len(initial_rows),
                "retest_rollouts": len(retest_rows),
                "rollout_detail_unique": len(detail_keys),
                "final_counts": {"knowledge": len(selected["knowledge"]), "math": len(selected["math"]), "merged": len(merged)},
                "validation_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "pilot", "probe-only", "verify-probe", "full", "verify", "worker"))
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_DEFAULT))
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=96)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--worker-rank", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--worker-task-file", type=Path)
    parser.add_argument("--worker-output-path", type=Path)
    parser.add_argument("--worker-phase", choices=("pilot", "initial", "retest"))
    parser.add_argument("--worker-attempt", type=int, default=1)
    parser.add_argument("--worker-caps-json", default="{}")
    parser.add_argument("--worker-reuse-pilot", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.gpus <= 4:
        parser.error("--gpus must be between 1 and 4")
    if min(args.max_model_len, args.max_num_seqs, args.max_num_batched_tokens) <= 0:
        parser.error("model length, sequence concurrency and token budget must be positive")
    if args.output_dir is None:
        args.output_dir = args.root / "processed_data" / "rl_probe"
    if args.batch_size <= 0 or args.batch_size % 4:
        parser.error("--batch-size must be a positive multiple of 4")
    if args.mode == "worker":
        required = (args.worker_rank, args.worker_count, args.worker_task_file, args.worker_output_path, args.worker_phase)
        if any(value is None for value in required):
            parser.error("worker mode requires worker rank/count/task/output/phase arguments")
    return args


def main() -> None:
    args = parse_args()
    if args.mode in {"probe-only", "verify-probe"} or (args.mode == "worker" and args.worker_phase == "initial"):
        enforce_probe_input_boundary()
    if args.mode == "self-test":
        self_test()
    elif args.mode == "pilot":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_pilot(args)
    elif args.mode == "full":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_full(args)
    elif args.mode == "probe-only":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_probe_only(args)
    elif args.mode == "verify-probe":
        verify_probe_only(args)
    elif args.mode == "worker":
        run_worker(args)
    else:
        verify_local(args)


if __name__ == "__main__":
    main()

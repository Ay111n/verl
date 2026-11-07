"""Per-domain answer metrics and actual-GRPO-group diagnostics."""
from __future__ import annotations
from collections import defaultdict
import math
import statistics


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def _flatten(row):
    return {**row, **row.get("reward_extra_info", {})}


def group_metrics(records, num_generations=8):
    groups = defaultdict(list)
    for row in records:
        row = _flatten(row)
        if not row.get("group_id"):
            raise ValueError("Missing GRPO group_id")
        groups[row["group_id"]].append(row)
    histogram = [0] * (num_generations + 1)
    details = []
    zero_count, observed_adv_groups = 0, 0
    for gid, rows in groups.items():
        indices = [row.get("rollout_index") for row in rows]
        if len(rows) != num_generations or set(indices) != set(range(num_generations)):
            raise ValueError(f"Missing/duplicate responses in group {gid}")
        if any(row["correctness"] not in (0, 1, False, True) for row in rows):
            raise ValueError("Correctness must be binary")
        correct = sum(int(row["correctness"]) for row in rows)
        if not 0 <= correct <= num_generations:
            raise ValueError("Correctness must be binary")
        histogram[correct] += 1
        rewards = [float(row["total_reward"]) for row in rows]
        actual_zero = None
        if all(row.get("advantages") is not None for row in rows):
            values = []
            for row in rows:
                adv = row["advantages"]
                adv = adv if isinstance(adv, (list, tuple)) else [adv]
                mask = row.get("response_mask", [1] * len(adv))
                if len(mask) != len(adv):
                    raise ValueError("Advantage/mask lengths differ")
                values.extend(float(v) for v, valid in zip(adv, mask) if valid)
            if not values or any(not math.isfinite(v) for v in values):
                raise ValueError("Actual advantages must contain finite valid response values")
            actual_zero = all(v == 0.0 for v in values)
            observed_adv_groups += 1
            zero_count += int(actual_zero)
        details.append({"group_id": gid, "correct_count": correct,
                        "reward_mean": statistics.mean(rewards), "reward_std": statistics.pstdev(rewards),
                        "actual_zero_advantage": actual_zero})
    return {"groups": details, "correct_count_histogram": histogram,
            "actual_advantage_observed_groups": observed_adv_groups,
            "zero_advantage_group_rate": zero_count / observed_adv_groups
            if observed_adv_groups == len(groups) and groups else None}


def summarise_records(records, num_generations=8):
    rows = [_flatten(row) for row in records]
    result = {"responses": len(rows)}
    if not rows:
        return result
    for domain in ("knowledge", "math"):
        subset = [row for row in rows if row.get("domain") == domain]
        result[f"{domain}/responses"] = len(subset)
        result[f"{domain}/accuracy"] = statistics.mean(row["correctness"] for row in subset) if subset else None
    for key in ("correctness", "answer_extractable", "thinking_closed", "thinking_unclosed",
                "no_reliable_final", "empty_final", "extraction_failed", "truncated", "protocol_valid"):
        values = [int(row[key]) for row in rows if row.get(key) is not None]
        result[f"{key}/rate"] = statistics.mean(values) if values else None
    known_no_answer = [row["no_answer"] for row in rows if row.get("no_answer") is not None]
    result["no_answer/known_count"] = len(known_no_answer)
    result["no_answer/rate_known"] = statistics.mean(known_no_answer) if known_no_answer else None
    for key in ("thinking_tokens", "output_tokens", "total_reward", "component_r0"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[f"{key}/mean"] = statistics.mean(values) if values else None
        result[f"{key}/std"] = statistics.pstdev(values) if values else None
        if key in ("thinking_tokens", "output_tokens"):
            result[f"{key}/median"] = statistics.median(values) if values else None
            result[f"{key}/p95"] = percentile(values, .95)
    grouped = group_metrics(rows, num_generations=num_generations)
    result["grpo/groups"] = len(grouped["groups"])
    result["grpo/zero_advantage_group_rate"] = grouped["zero_advantage_group_rate"]
    result["grpo/actual_advantage_observed_groups"] = grouped["actual_advantage_observed_groups"]
    for count, value in enumerate(grouped["correct_count_histogram"]):
        result[f"grpo/correct_count_{count}"] = value
    return result

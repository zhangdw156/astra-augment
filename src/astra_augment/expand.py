"""Truncation-based SFT data augmentation for tool-calling conversations."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .utils import find_indices, immediate_response_failed, read_jsonl


def _tail_indices(indices: list[int], ratio: float) -> list[int]:
    """Return the last `ratio` fraction of indices."""
    count = max(1, math.ceil(len(indices) * ratio))
    return indices[-count:]


def expand_record(
    record: dict[str, Any],
    ratio: float,
    mode: str,
) -> list[dict[str, Any]]:
    """Generate augmented samples from a single conversation record."""
    messages = record.get("messages", [])
    if len(messages) < 3:
        return []

    indices = find_indices(messages, mode)

    if mode == "tool_call":
        if not indices:
            return []
        targets = _tail_indices(indices, ratio)
        results = []
        for idx in targets:
            if not immediate_response_failed(messages, idx):
                results.append({"messages": messages[: idx + 1]})
        return results

    elif mode == "response":
        if len(indices) < 2:
            return []
        candidates = indices[:-1]
        targets = _tail_indices(candidates, ratio)
        return [{"messages": messages[: idx + 1]} for idx in targets]

    return []


def expand(
    input_path: Path,
    output_path: Path,
    ratio: float,
    mode: str,
    format: str = "qwen3",
) -> int:
    """Read JSONL, generate augmented samples, write to output. Returns count."""
    if format != "qwen3":
        raise ValueError(f"Unsupported format: {format!r}. Only 'qwen3' is supported.")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    count = 0
    with open(output_path, "w") as fout:
        for record in read_jsonl(input_path):
            for aug in expand_record(record, ratio, mode):
                fout.write(json.dumps(aug, ensure_ascii=False) + "\n")
                count += 1
    return count

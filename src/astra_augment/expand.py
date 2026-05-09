"""Truncation-based SFT data augmentation for tool-calling conversations."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .utils import is_assistant, is_tool_call


def _has_failed_response_after(messages: list[dict[str, Any]], idx: int) -> bool:
    """Check if any tool_response after idx contains a failure."""
    for msg in messages[idx + 1 :]:
        content = msg.get("content", "")
        if "<tool_response>" not in content:
            continue
        start = content.find("<tool_response>") + len("<tool_response>")
        end = content.find("</tool_response>")
        if start < 0 or end < 0:
            continue
        body = content[start:end].strip()
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("success") is False:
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    return False


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

    if mode == "tool_call":
        tc_indices = [i for i, m in enumerate(messages) if is_tool_call(m)]
        if not tc_indices:
            return []
        targets = _tail_indices(tc_indices, ratio)
        results = []
        for idx in targets:
            if not _has_failed_response_after(messages, idx):
                results.append({"messages": messages[: idx + 1]})
        return results

    elif mode == "response":
        asst_indices = [i for i, m in enumerate(messages) if is_assistant(m)]
        if not asst_indices:
            return []
        # exclude the very last assistant (that's the original full conversation)
        if len(asst_indices) < 2:
            return []
        candidates = asst_indices[:-1]
        targets = _tail_indices(candidates, ratio)
        return [{"messages": messages[: idx + 1]} for idx in targets]

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'tool_call' or 'response'.")


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
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for aug in expand_record(record, ratio, mode):
                fout.write(json.dumps(aug, ensure_ascii=False) + "\n")
                count += 1
    return count

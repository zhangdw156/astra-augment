"""Extract a specific tool-call or response position from conversations."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .utils import find_indices, immediate_response_failed, read_jsonl


def slice_record(
    record: dict[str, Any],
    last: int,
    mode: str,
) -> dict[str, Any] | None:
    """Extract a single truncated sample at a specific position.

    Args:
        record: {"messages": [...]}
        last: positive int, 1 = last position, 2 = second-to-last, etc.
        mode: "tool_call" or "response"

    Returns:
        Truncated record, or None if position is invalid or tool call failed.
    """
    messages = record.get("messages", [])
    if len(messages) < 3:
        return None

    indices = find_indices(messages, mode)

    if not indices or last > len(indices):
        return None

    idx = indices[-last]

    if mode == "tool_call" and immediate_response_failed(messages, idx):
        return None

    return {"messages": messages[: idx + 1]}


def slice_at(
    input_path: Path,
    output_path: Path,
    last: int,
    mode: str,
    format: str = "qwen3",
) -> tuple[int, int]:
    """Extract a specific position from each conversation. Returns (kept, total)."""
    if format != "qwen3":
        raise ValueError(f"Unsupported format: {format!r}. Only 'qwen3' is supported.")
    if last < 1:
        raise ValueError(f"last must be >= 1, got {last}")

    kept = 0
    total = 0
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fout:
            for record in read_jsonl(input_path):
                total += 1
                result = slice_record(record, last, mode)
                if result is not None:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    kept += 1
        Path(tmp).replace(output_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return kept, total

"""Extract a specific tool-call or response position from conversations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import is_assistant, is_tool_call


def _immediate_response_failed(messages: list[dict[str, Any]], idx: int) -> bool:
    """Check if the tool_response immediately following idx is a failure."""
    for msg in messages[idx + 1 :]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant"):
            return False
        if "<tool_response>" in content:
            start = content.find("<tool_response>") + len("<tool_response>")
            end = content.find("</tool_response>")
            if 0 <= start < end:
                try:
                    parsed = json.loads(content[start:end].strip())
                    return isinstance(parsed, dict) and parsed.get("success") is False
                except (json.JSONDecodeError, TypeError):
                    pass
            return False
    return False


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

    if mode == "tool_call":
        indices = [i for i, m in enumerate(messages) if is_tool_call(m)]
    elif mode == "response":
        indices = [i for i, m in enumerate(messages) if is_assistant(m)]
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'tool_call' or 'response'.")

    if not indices or last > len(indices):
        return None

    idx = indices[-last]

    if mode == "tool_call" and _immediate_response_failed(messages, idx):
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
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            result = slice_record(record, last, mode)
            if result is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                kept += 1
    return kept, total

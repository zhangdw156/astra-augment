"""Shared helpers for message inspection and JSONL I/O."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def is_tool_call(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "assistant" and "<tool_call>" in msg.get("content", "")


def is_response(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "assistant" and "<tool_call>" not in msg.get("content", "")


def find_indices(messages: list[dict[str, Any]], mode: str) -> list[int]:
    if mode == "tool_call":
        return [i for i, m in enumerate(messages) if is_tool_call(m)]
    elif mode == "response":
        return [i for i, m in enumerate(messages) if is_response(m)]
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'tool_call' or 'response'.")


def immediate_response_failed(messages: list[dict[str, Any]], idx: int) -> bool:
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


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

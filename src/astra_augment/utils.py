"""Shared helpers for message inspection and JSONL I/O."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def is_tool_call(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "assistant" and "<tool_call>" in msg.get("content", "")


def is_response(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    return (
        msg.get("role") == "assistant"
        and isinstance(content, str)
        and bool(content)
        and "<tool_call>" not in content
    )


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
            start_pos = content.find("<tool_response>")
            end = content.find("</tool_response>")
            if start_pos >= 0 and end > start_pos:
                start = start_pos + len("<tool_response>")
                try:
                    parsed = json.loads(content[start:end].strip())
                    return isinstance(parsed, dict) and parsed.get("success") is False
                except (json.JSONDecodeError, TypeError):
                    pass
            return False
    return False


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e

"""Shared helpers for message inspection."""
from __future__ import annotations

from typing import Any


def is_tool_call(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "assistant" and "<tool_call>" in msg.get("content", "")


def is_response(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "assistant" and "<tool_call>" not in msg.get("content", "")

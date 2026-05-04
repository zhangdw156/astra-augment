"""Tests for astra_augment.expand."""

import json

import pytest

from astra_augment.expand import expand, expand_record


def _make_record(messages):
    return {"messages": messages}


def _msg(role, content):
    return {"role": role, "content": content}


class TestExpandRecordToolCall:
    def test_basic_tool_call_truncation(self):
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "<tool_call>{}</tool_call>"),
            _msg("tool", '<tool_response>{"success": true}</tool_response>'),
            _msg("assistant", "done"),
        ])
        results = expand_record(record, ratio=1.0, mode="tool_call")
        assert len(results) == 1
        assert len(results[0]["messages"]) == 2

    def test_skips_failed_response(self):
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "<tool_call>{}</tool_call>"),
            _msg("tool", '<tool_response>{"success": false}</tool_response>'),
            _msg("assistant", "retry"),
        ])
        results = expand_record(record, ratio=1.0, mode="tool_call")
        assert len(results) == 0

    def test_no_tool_calls(self):
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _msg("user", "bye"),
        ])
        results = expand_record(record, ratio=1.0, mode="tool_call")
        assert results == []

    def test_too_few_messages(self):
        record = _make_record([_msg("user", "hi"), _msg("assistant", "hello")])
        assert expand_record(record, ratio=1.0, mode="tool_call") == []

    def test_ratio_selects_tail(self):
        record = _make_record([
            _msg("user", "q1"),
            _msg("assistant", "<tool_call>1</tool_call>"),
            _msg("tool", '<tool_response>{"success": true}</tool_response>'),
            _msg("user", "q2"),
            _msg("assistant", "<tool_call>2</tool_call>"),
            _msg("tool", '<tool_response>{"success": true}</tool_response>'),
            _msg("user", "q3"),
            _msg("assistant", "<tool_call>3</tool_call>"),
            _msg("tool", '<tool_response>{"success": true}</tool_response>'),
            _msg("assistant", "done"),
        ])
        results = expand_record(record, ratio=0.34, mode="tool_call")
        # ceil(3 * 0.34) = ceil(1.02) = 2, so last 2 tool calls selected
        assert len(results) == 2
        assert results[-1]["messages"][-1]["content"] == "<tool_call>3</tool_call>"


class TestExpandRecordResponse:
    def test_basic_response_truncation(self):
        record = _make_record([
            _msg("user", "q1"),
            _msg("assistant", "a1"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
            _msg("user", "q3"),
            _msg("assistant", "a3"),
        ])
        results = expand_record(record, ratio=0.5, mode="response")
        assert len(results) == 1
        assert results[0]["messages"][-1]["content"] == "a2"

    def test_single_assistant_returns_empty(self):
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _msg("user", "bye"),
        ])
        results = expand_record(record, ratio=1.0, mode="response")
        assert results == []

    def test_unknown_mode_raises(self):
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _msg("user", "bye"),
        ])
        with pytest.raises(ValueError, match="Unknown mode"):
            expand_record(record, ratio=1.0, mode="invalid")


class TestExpand:
    def test_end_to_end(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"

        records = [
            {"messages": [
                _msg("user", "q1"),
                _msg("assistant", "<tool_call>x</tool_call>"),
                _msg("tool", '<tool_response>{"success": true}</tool_response>'),
                _msg("assistant", "done"),
            ]},
        ]
        input_path.write_text("\n".join(json.dumps(r) for r in records))

        count = expand(input_path, output_path, ratio=1.0, mode="tool_call")
        assert count == 1

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1
        result = json.loads(lines[0])
        assert len(result["messages"]) == 2

    def test_invalid_ratio(self, tmp_path):
        with pytest.raises(ValueError, match="ratio must be in"):
            expand(tmp_path / "in.jsonl", tmp_path / "out.jsonl", ratio=0.0, mode="tool_call")

    def test_unsupported_format(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported format"):
            expand(
                tmp_path / "in.jsonl", tmp_path / "out.jsonl",
                ratio=0.5, mode="tool_call", format="llama",
            )

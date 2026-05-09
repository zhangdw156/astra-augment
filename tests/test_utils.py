"""Tests for astra_augment.utils."""

import pytest

from astra_augment.utils import (
    find_indices,
    immediate_response_failed,
    is_response,
    is_tool_call,
    read_jsonl,
)


def _msg(role, content):
    return {"role": role, "content": content}


# ── is_tool_call ───────────────────────────────────────────────────


class TestIsToolCall:
    def test_true_for_tool_call(self):
        assert is_tool_call(_msg("assistant", "<tool_call>{}</tool_call>"))

    def test_false_for_plain_assistant(self):
        assert not is_tool_call(_msg("assistant", "hello"))

    def test_false_for_user(self):
        assert not is_tool_call(_msg("user", "<tool_call>{}</tool_call>"))

    def test_false_for_missing_content(self):
        assert not is_tool_call({"role": "assistant"})

    def test_false_for_empty_content(self):
        assert not is_tool_call(_msg("assistant", ""))


# ── is_response ────────────────────────────────────────────────────


class TestIsResponse:
    def test_true_for_plain_assistant(self):
        assert is_response(_msg("assistant", "hello"))

    def test_false_for_tool_call(self):
        assert not is_response(_msg("assistant", "<tool_call>{}</tool_call>"))

    def test_false_for_user(self):
        assert not is_response(_msg("user", "hello"))

    def test_false_for_missing_content(self):
        assert not is_response({"role": "assistant"})

    def test_false_for_empty_content(self):
        assert not is_response(_msg("assistant", ""))

    def test_false_for_none_content(self):
        assert not is_response({"role": "assistant", "content": None})

    def test_false_for_non_string_content(self):
        assert not is_response({"role": "assistant", "content": 123})


# ── find_indices ───────────────────────────────────────────────────


class TestFindIndices:
    def test_tool_call_mode(self):
        messages = [
            _msg("user", "q"),
            _msg("assistant", "<tool_call>x</tool_call>"),
            _msg("tool", "resp"),
            _msg("assistant", "done"),
        ]
        assert find_indices(messages, "tool_call") == [1]

    def test_response_mode(self):
        messages = [
            _msg("user", "q"),
            _msg("assistant", "a1"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
        ]
        assert find_indices(messages, "response") == [1, 3]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            find_indices([], "bad")

    def test_empty_messages(self):
        assert find_indices([], "tool_call") == []


# ── immediate_response_failed ──────────────────────────────────────


class TestImmediateResponseFailed:
    def test_success_response(self):
        messages = [
            _msg("assistant", "<tool_call>x</tool_call>"),
            _msg("tool", '<tool_response>{"success": true}</tool_response>'),
        ]
        assert not immediate_response_failed(messages, 0)

    def test_failed_response(self):
        messages = [
            _msg("assistant", "<tool_call>x</tool_call>"),
            _msg("tool", '<tool_response>{"success": false}</tool_response>'),
        ]
        assert immediate_response_failed(messages, 0)

    def test_missing_closing_tag(self):
        messages = [
            _msg("assistant", "<tool_call>x</tool_call>"),
            _msg("tool", '<tool_response>{"success": false}'),
        ]
        assert not immediate_response_failed(messages, 0)

    def test_missing_opening_tag(self):
        messages = [
            _msg("assistant", "<tool_call>x</tool_call>"),
            _msg("tool", '{"success": false}</tool_response>'),
        ]
        assert not immediate_response_failed(messages, 0)

    def test_malformed_json_inside_tags(self):
        messages = [
            _msg("assistant", "<tool_call>x</tool_call>"),
            _msg("tool", "<tool_response>not json</tool_response>"),
        ]
        assert not immediate_response_failed(messages, 0)

    def test_stops_at_user_message(self):
        messages = [
            _msg("assistant", "<tool_call>x</tool_call>"),
            _msg("user", "interrupt"),
            _msg("tool", '<tool_response>{"success": false}</tool_response>'),
        ]
        assert not immediate_response_failed(messages, 0)

    def test_no_following_messages(self):
        messages = [_msg("assistant", "<tool_call>x</tool_call>")]
        assert not immediate_response_failed(messages, 0)


# ── read_jsonl ─────────────────────────────────────────────────────


class TestReadJsonl:
    def test_valid_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n')
        records = list(read_jsonl(path))
        assert records == [{"a": 1}, {"b": 2}]

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n')
        records = list(read_jsonl(path))
        assert records == [{"a": 1}, {"b": 2}]

    def test_invalid_json_reports_line_number(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
        with pytest.raises(ValueError, match=r":2:.*invalid JSON"):
            list(read_jsonl(path))

    def test_invalid_json_on_line_3_with_blank(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n\nBAD\n')
        with pytest.raises(ValueError, match=r":3:.*invalid JSON"):
            list(read_jsonl(path))

    def test_empty_file(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text("")
        assert list(read_jsonl(path)) == []

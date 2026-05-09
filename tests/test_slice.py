"""Tests for astra_augment.slice."""

import json

import pytest
from click.testing import CliRunner

from astra_augment.cli import main
from astra_augment.slice import slice_at, slice_record


def _make_record(messages):
    return {"messages": messages}


def _msg(role, content):
    return {"role": role, "content": content}


MULTI_TC_RECORD = _make_record([
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


# ── CLI integration tests ───────────────────────────────────────────


class TestSliceCLI:
    def test_basic_slice(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        input_path.write_text(json.dumps(MULTI_TC_RECORD) + "\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "slice", str(input_path),
            "-o", str(output_path),
            "--last", "2",
            "--mode", "tool_call",
        ])
        assert result.exit_code == 0
        assert "1/1" in result.output

        output = json.loads(output_path.read_text().strip())
        assert output["messages"][-1]["content"] == "<tool_call>2</tool_call>"

    def test_slice_last_3(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        input_path.write_text(json.dumps(MULTI_TC_RECORD) + "\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "slice", str(input_path),
            "-o", str(output_path),
            "--last", "3",
            "--mode", "tool_call",
        ])
        assert result.exit_code == 0
        output = json.loads(output_path.read_text().strip())
        assert output["messages"][-1]["content"] == "<tool_call>1</tool_call>"

    def test_slice_response_mode(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        record = _make_record([
            _msg("user", "q1"),
            _msg("assistant", "a1"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
        ])
        input_path.write_text(json.dumps(record) + "\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "slice", str(input_path),
            "-o", str(output_path),
            "--last", "2",
            "--mode", "response",
        ])
        assert result.exit_code == 0
        output = json.loads(output_path.read_text().strip())
        assert output["messages"][-1]["content"] == "a1"

    def test_slice_invalid_last(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        input_path.write_text(json.dumps(MULTI_TC_RECORD) + "\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "slice", str(input_path),
            "-o", str(tmp_path / "out.jsonl"),
            "--last", "0",
            "--mode", "tool_call",
        ])
        assert result.exit_code != 0

    def test_slice_skips_failed_tool_response(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "<tool_call>bad</tool_call>"),
            _msg("tool", '<tool_response>{"success": false}</tool_response>'),
            _msg("assistant", "done"),
        ])
        input_path.write_text(json.dumps(record) + "\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "slice", str(input_path),
            "-o", str(output_path),
            "--last", "1",
            "--mode", "tool_call",
        ])
        assert result.exit_code == 0
        assert "0/1" in result.output

    def test_slice_multiple_records(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        rec2 = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "<tool_call>only</tool_call>"),
            _msg("tool", '<tool_response>{"success": true}</tool_response>'),
            _msg("assistant", "done"),
        ])
        input_path.write_text(json.dumps(MULTI_TC_RECORD) + "\n" + json.dumps(rec2) + "\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "slice", str(input_path),
            "-o", str(output_path),
            "--last", "2",
            "--mode", "tool_call",
        ])
        assert result.exit_code == 0
        assert "1/2" in result.output


# ── slice_record unit tests ─────────────────────────────────────────


class TestSliceRecordToolCall:
    def test_last_1(self):
        result = slice_record(MULTI_TC_RECORD, last=1, mode="tool_call")
        assert result is not None
        assert result["messages"][-1]["content"] == "<tool_call>3</tool_call>"

    def test_last_2(self):
        result = slice_record(MULTI_TC_RECORD, last=2, mode="tool_call")
        assert result is not None
        assert result["messages"][-1]["content"] == "<tool_call>2</tool_call>"

    def test_last_3(self):
        result = slice_record(MULTI_TC_RECORD, last=3, mode="tool_call")
        assert result is not None
        assert result["messages"][-1]["content"] == "<tool_call>1</tool_call>"

    def test_position_out_of_range(self):
        assert slice_record(MULTI_TC_RECORD, last=5, mode="tool_call") is None

    def test_skips_failed_response(self):
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "<tool_call>ok</tool_call>"),
            _msg("tool", '<tool_response>{"success": true}</tool_response>'),
            _msg("user", "again"),
            _msg("assistant", "<tool_call>bad</tool_call>"),
            _msg("tool", '<tool_response>{"success": false}</tool_response>'),
            _msg("assistant", "retry"),
        ])
        assert slice_record(record, last=1, mode="tool_call") is None
        result = slice_record(record, last=2, mode="tool_call")
        assert result is not None
        assert result["messages"][-1]["content"] == "<tool_call>ok</tool_call>"

    def test_only_checks_immediate_response(self):
        record = _make_record([
            _msg("user", "q1"),
            _msg("assistant", "<tool_call>first</tool_call>"),
            _msg("tool", '<tool_response>{"success": true}</tool_response>'),
            _msg("user", "q2"),
            _msg("assistant", "<tool_call>second</tool_call>"),
            _msg("tool", '<tool_response>{"success": false}</tool_response>'),
            _msg("assistant", "done"),
        ])
        result = slice_record(record, last=2, mode="tool_call")
        assert result is not None
        assert result["messages"][-1]["content"] == "<tool_call>first</tool_call>"

    def test_no_tool_calls(self):
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _msg("user", "bye"),
        ])
        assert slice_record(record, last=1, mode="tool_call") is None

    def test_too_few_messages(self):
        record = _make_record([_msg("user", "hi"), _msg("assistant", "hello")])
        assert slice_record(record, last=1, mode="tool_call") is None


class TestSliceRecordResponse:
    def test_last_1(self):
        record = _make_record([
            _msg("user", "q1"),
            _msg("assistant", "a1"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
            _msg("user", "q3"),
            _msg("assistant", "a3"),
        ])
        result = slice_record(record, last=1, mode="response")
        assert result is not None
        assert result["messages"][-1]["content"] == "a3"

    def test_last_2(self):
        record = _make_record([
            _msg("user", "q1"),
            _msg("assistant", "a1"),
            _msg("user", "q2"),
            _msg("assistant", "a2"),
            _msg("user", "q3"),
            _msg("assistant", "a3"),
        ])
        result = slice_record(record, last=2, mode="response")
        assert result is not None
        assert result["messages"][-1]["content"] == "a2"

    def test_unknown_mode_raises(self):
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _msg("user", "bye"),
        ])
        with pytest.raises(ValueError, match="Unknown mode"):
            slice_record(record, last=1, mode="invalid")


class TestSliceAt:
    def test_end_to_end(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        input_path.write_text(json.dumps(MULTI_TC_RECORD) + "\n")

        kept, total = slice_at(input_path, output_path, last=2, mode="tool_call")
        assert total == 1
        assert kept == 1
        result = json.loads(output_path.read_text().strip())
        assert result["messages"][-1]["content"] == "<tool_call>2</tool_call>"

    def test_skips_insufficient_records(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        record = _make_record([
            _msg("user", "hi"),
            _msg("assistant", "<tool_call>x</tool_call>"),
            _msg("tool", '<tool_response>{"success": true}</tool_response>'),
            _msg("assistant", "done"),
        ])
        input_path.write_text(json.dumps(record) + "\n")

        kept, total = slice_at(input_path, output_path, last=3, mode="tool_call")
        assert total == 1
        assert kept == 0

    def test_invalid_last_zero(self, tmp_path):
        with pytest.raises(ValueError, match="last must be >= 1"):
            slice_at(tmp_path / "in.jsonl", tmp_path / "out.jsonl", last=0, mode="tool_call")

    def test_unsupported_format(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported format"):
            slice_at(
                tmp_path / "in.jsonl", tmp_path / "out.jsonl",
                last=1, mode="tool_call", format="llama",
            )

    def test_no_partial_output_on_bad_input(self, tmp_path):
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        input_path.write_text(json.dumps(MULTI_TC_RECORD) + "\nBAD JSON\n")
        with pytest.raises(ValueError, match="invalid JSON"):
            slice_at(input_path, output_path, last=1, mode="tool_call")
        assert not output_path.exists()

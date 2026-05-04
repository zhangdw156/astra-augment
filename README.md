# astra-augment

[![PyPI](https://img.shields.io/pypi/v/astra-augment)](https://pypi.org/project/astra-augment/)
[![Python](https://img.shields.io/pypi/pyversions/astra-augment)](https://pypi.org/project/astra-augment/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SFT data augmentation via conversation truncation for tool-calling trajectories.

Part of the [Astra](https://github.com/zhangdw156/Astra) ecosystem — a data factory for high-quality multi-turn tool-calling conversation trajectories.

## Install

```bash
pip install astra-augment
```

## Usage

```bash
# Expand last 20% of tool calls into separate training samples
astra-augment expand input.jsonl -o output.jsonl --ratio 0.2 --mode tool_call

# Expand last 40% of assistant responses
astra-augment expand input.jsonl -o output.jsonl --ratio 0.4 --mode response
```

### Modes

- **`tool_call`**: Truncates conversations at assistant tool-call positions (skips those followed by failed responses)
- **`response`**: Truncates at assistant response positions (excludes the final one to avoid duplicating the original)

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--ratio` | Fraction of tail positions to expand (0, 1] |
| `--mode` | Truncation mode: `tool_call` or `response` |
| `--format` | Dataset format (default: `qwen3`) |
| `-o` | Output JSONL path |

## Development

```bash
# Clone and set up
git clone https://github.com/zhangdw156/astra-augment.git
cd astra-augment
uv sync --all-groups

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## License

MIT

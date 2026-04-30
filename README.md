# astra-augment

SFT data augmentation via conversation truncation for tool-calling trajectories.

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

"""CLI for astra-augment."""

from __future__ import annotations

from pathlib import Path

import click

from . import __version__


@click.group()
@click.version_option(version=__version__, prog_name="astra-augment")
def main() -> None:
    """SFT data augmentation via conversation truncation."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--ratio", required=True, type=float, help="Fraction of tail positions to expand (0, 1]."
)
@click.option(
    "--mode", required=True, type=click.Choice(["tool_call", "response"]), help="Truncation mode."
)
@click.option(
    "--format", "fmt", default="qwen3", type=click.Choice(["qwen3"]), help="Dataset format."
)
def expand(input_path: Path, output_path: Path, ratio: float, mode: str, fmt: str) -> None:
    """Generate augmented samples by truncating conversations."""
    from .expand import expand as do_expand

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = do_expand(input_path, output_path, ratio, mode, format=fmt)
    click.echo(f"Generated {count} augmented samples → {output_path}")

#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

VERSION_FILE = Path("VERSION")


def load_version(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Version file not found: {path}")
    version = path.read_text(encoding="utf-8").strip()
    if not version:
        raise RuntimeError(f"Empty version in {path}")
    return version


def write_outputs(values: dict[str, str]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            for key, value in values.items():
                handle.write(f"{key}={value}\n")
        return

    for key, value in values.items():
        print(f"{key}={value}")


def main() -> int:
    try:
        version = load_version(VERSION_FILE)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    outputs = {
        "version": version,
        "tag": f"v{version}",
        "is_release": "true" if ".dev" not in version else "false",
    }
    write_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

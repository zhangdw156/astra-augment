#!/usr/bin/env python3
"""Compute the next development version after a release and update VERSION.

Given a release version like 0.7.0, bumps minor → 0.8.0.dev0.
Skips if the dev branch already has an equal or newer version.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

VERSION_FILE = Path("VERSION")


def parse_base(v: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in v.split(".dev")[0].split("."))  # type: ignore[return-value]


def next_dev_version(release: str) -> str:
    # TODO: only supports minor bumps; extend for major/patch releases
    major, minor, _ = parse_base(release)
    return f"{major}.{minor + 1}.0.dev0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_version")
    args = parser.parse_args()

    release = args.release_version.strip()
    if ".dev" in release:
        print(f"Not a release version: {release}", file=sys.stderr)
        return 1

    new_version = next_dev_version(release)
    current = VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else ""

    if parse_base(current) >= parse_base(new_version):
        print(f"VERSION {current} is already at or ahead of {new_version}, skipping.")
        return 0

    VERSION_FILE.write_text(new_version + "\n", encoding="utf-8")
    print(f"Bumped VERSION: {current} -> {new_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

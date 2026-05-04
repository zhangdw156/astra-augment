# CLAUDE.md

## Setup and common commands

- Python: >=3.10
- Package/dependency manager: `uv`
- Install: `uv sync`
- Install with dev deps: `uv sync --all-groups`
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_expand.py`
- Lint: `uv run ruff check .`
- Build: `uv build`

## CLI entry point

```bash
astra-augment expand input.jsonl -o output.jsonl --ratio 0.2 --mode tool_call
```

## Architecture

This is a small, focused CLI tool with three source files:

- `src/astra_augment/__init__.py` — version via importlib.metadata
- `src/astra_augment/cli.py` — Click CLI with `expand` subcommand
- `src/astra_augment/expand.py` — core truncation logic (`expand_record()`, `expand()`)

## Version management

Single `VERSION` file at repo root. Format: `X.Y.Z.devN` for development, `X.Y.Z` for release.

Release workflow: merge to `main` with a non-`.dev` version triggers `.github/workflows/release-on-main.yml`, which creates a GitHub release and publishes to PyPI.

### Dev version bumps

On `dev`, versions use `X.Y.Z.devN`. To bump: increment only `N`.

## Issue-Driven Development (IDD)

Same workflow as the parent Astra project:

1. Create an issue first
2. Branch from `dev`: `git checkout -b <type>/<issue>-<desc>`
3. Commit messages: Conventional Commits with scope
4. PR targeting `dev`, body includes `Closes #<issue>`
5. Squash merge, delete feature branch

### Branch model

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases |
| `dev` | Active development |

## Coding conventions

- Conventional Commits for messages
- ruff for linting (line-length 100)
- No comments unless the WHY is non-obvious

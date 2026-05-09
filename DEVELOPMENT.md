# Development

This project follows **Issue-Driven Development (IDD)** — every change starts with an issue.

## Branch Model

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases |
| `dev`  | Active development |

## Workflow

1. **Create an issue** describing the bug, feature, or task.
2. **Create a branch** from `dev`:
   ```bash
   git checkout dev && git pull
   git checkout -b <type>/<issue>-<short-desc>
   ```
   Branch type examples: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`.
3. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(expand): add dry-run option
   fix(cli): handle empty input file gracefully
   docs: update DEVELOPMENT.md
   ```
4. **Open a PR** targeting `dev`. The PR body must include `Closes #<issue>`.
5. **Squash merge**, then delete the feature branch.

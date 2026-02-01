---
description: Format and lint the CS2 Analyzer codebase with ruff
---

Format and lint the codebase using ruff.

## Commands

**Format + lint with auto-fix:**
```cmd
ruff format src/ tests/ && ruff check --fix src/ tests/
```

**Check only (no changes):**
```cmd
ruff format --check src/ tests/ && ruff check src/ tests/
```

## What Ruff Checks

Based on pyproject.toml configuration:
- **E**: pycodestyle errors
- **F**: pyflakes
- **W**: pycodestyle warnings
- **I**: isort (import sorting)
- **UP**: pyupgrade
- **B**: flake8-bugbear
- **C4**: flake8-comprehensions

## If Issues Remain After --fix

Some issues can't be auto-fixed:
1. Read each remaining issue
2. Fix manually following the suggestion
3. Re-run to verify clean

## Rules

- Never commit with linting errors
- Line length: 100 characters max
- Target: Python 3.11+
- No wildcard imports
- Type hints on all functions

## Pre-commit Integration

If pre-commit is enabled, ruff runs automatically on commit.
To run pre-commit manually:
```cmd
pre-commit run --all-files
```

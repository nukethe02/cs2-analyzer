---
description: Check and fix failing CI tests automatically
---

# Fix Failing CI Tests

Check GitHub Actions status and fix failures.

## Step 1: Check CI Status

```cmd
gh run list --limit 5
```

Or check specific workflow:
```cmd
gh run list --workflow=ci.yml --limit 5
```

## Step 2: Get Failure Details

If there's a failure, get the logs:
```cmd
gh run view {run_id} --log-failed
```

Or view in browser:
```cmd
gh run view {run_id} --web
```

## Step 3: Identify the Issue

Common CI failure categories:

### Lint Failures (ruff)
```
Error: ruff check failed
```
**Fix:**
```cmd
ruff format src/ tests/ && ruff check --fix src/ tests/
```

### Type Check Failures (mypy)
```
Error: mypy found issues
```
**Fix:** Add type hints or fix type errors in the flagged files.

### Test Failures (pytest)
```
FAILED tests/test_something.py::test_name
```
**Fix:**
1. Run locally: `set PYTHONPATH=src && pytest tests/test_something.py::test_name -v`
2. Read the assertion error
3. Fix the code or test

### Security Failures (bandit)
```
High severity issue found
```
**Fix:** Address the security concern flagged by bandit.

### Build Failures
```
Error: Package build failed
```
**Fix:** Check pyproject.toml syntax and dependencies.

## Step 4: Fix Locally

1. Make the fix
2. Verify locally:
```cmd
ruff format src/ tests/ && ruff check --fix src/ tests/
set PYTHONPATH=src && pytest tests/ -v --tb=short
```

## Step 5: Push and Verify

```cmd
git add <fixed-files>
git commit -m "fix(ci): resolve <issue>"
git push
```

Then verify CI passes:
```cmd
gh run watch
```

## CI Jobs Reference

Based on `.github/workflows/ci.yml`:

| Job | What it checks |
|-----|----------------|
| lint | ruff format/check, mypy |
| test | pytest on Ubuntu/macOS/Windows, Python 3.11-3.13 |
| security | bandit, safety |
| build | Package builds correctly |
| docs | README renders |

## Quick Fixes

**Import sorting:**
```cmd
ruff check --fix --select I src/ tests/
```

**Format only:**
```cmd
ruff format src/ tests/
```

**Run single failing test:**
```cmd
set PYTHONPATH=src && pytest tests/test_file.py::test_name -v -x
```

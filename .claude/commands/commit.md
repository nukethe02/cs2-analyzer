---
description: Create a conventional commit for CS2 Analyzer
---

Create a conventional commit following project standards.

## Workflow

1. **Check status:**
```cmd
git status
```

2. **Review changes:**
```cmd
git diff
git diff --staged
```

3. **Stage specific files** (never use `git add .`):
```cmd
git add <specific-files>
```

4. **Commit with conventional format:**
```cmd
git commit -m "<type>(<scope>): <description>"
```

## Commit Types

| Type | Use For |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes nor adds |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance, dependencies |

## Scopes for CS2 Analyzer

| Scope | Directory/Area |
|-------|----------------|
| `api` | api.py, endpoints |
| `cli` | cli.py |
| `ai` | ai/ directory |
| `analysis` | analysis/ directory |
| `infra` | infra/ directory |
| `viz` | visualization/ directory |
| `core` | core/ directory |
| `integrations` | integrations/ directory |
| `ui` | static/, frontend |

## Example Commits

```
feat(api): add player comparison endpoint
fix(analysis): handle NaN values in TTD calculation
refactor(core): extract safe accessors to utils
test(metrics): add edge case tests for CP
chore(deps): update demoparser2 to 0.8.0
```

## Before Committing

- [ ] Tests pass: `/run-tests`
- [ ] Code formatted: `/format`
- [ ] No sensitive data (env vars, keys)
- [ ] Meaningful commit message

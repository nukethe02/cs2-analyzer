---
description: Create a pull request for CS2 Analyzer
---

# Create Pull Request

Create a well-structured PR for CS2 Analyzer.

## Pre-PR Checklist

- [ ] Tests pass: `/run-tests`
- [ ] Code formatted: `/format`
- [ ] Self-reviewed: `/grill-me`
- [ ] Behavior verified: `/prove-it-works`
- [ ] Branch pushed to remote

## Create PR

### 1. Push Branch

```cmd
git push -u origin <branch-name>
```

### 2. Create PR with gh CLI

```cmd
gh pr create --title "<type>(<scope>): <description>" --body "## Summary
- First change
- Second change

## Test Plan
- [ ] Unit tests pass
- [ ] Manual testing with real demo
- [ ] Edge cases verified

## Screenshots (if UI changes)
[Add screenshots here]

## Related Issues
Fixes #123

---
Generated with Claude Code"
```

### 3. Or Create Interactively

```cmd
gh pr create
```

Follow the prompts.

## PR Title Format

Follow conventional commits:
```
<type>(<scope>): <description>
```

**Types:** feat, fix, docs, style, refactor, perf, test, chore

**Scopes:** api, cli, ai, analysis, infra, viz, core, integrations, ui

**Examples:**
- `feat(api): add player comparison endpoint`
- `fix(analysis): handle NaN in TTD calculation`
- `refactor(core): simplify demo parsing`

## PR Body Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Specific change 1
- Specific change 2

## Test Plan
How was this tested?
- [ ] Unit tests added/updated
- [ ] Manual testing steps:
  1. Step 1
  2. Step 2

## Screenshots (if applicable)

## Performance Impact
- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance regression (justified because...)

## Breaking Changes
- [ ] No breaking changes
- [ ] Breaking change: [describe]

## Checklist
- [ ] Code follows project style
- [ ] Tests pass locally
- [ ] Documentation updated (if needed)
- [ ] CLAUDE.md updated (if needed)
```

## After Creating PR

1. **Check CI status:**
```cmd
gh pr checks
```

2. **View PR:**
```cmd
gh pr view --web
```

3. **Add reviewers (if applicable):**
```cmd
gh pr edit --add-reviewer <username>
```

## Common Issues

### CI Failing
Use `/fix-ci` to diagnose and fix.

### Merge Conflicts
```cmd
git fetch origin main
git rebase origin/main
# Resolve conflicts
git push --force-with-lease
```

### PR Too Large
Consider splitting into smaller PRs:
- One for refactoring
- One for new feature
- One for tests

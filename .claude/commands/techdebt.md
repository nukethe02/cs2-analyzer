---
description: Scan for technical debt - run at end of each session
---

Scan CS2 Analyzer for technical debt. Run at the end of each coding session.

## Scan Categories

### 1. TODO/FIXME Comments

```cmd
findstr /s /i "TODO FIXME HACK XXX BUG" src\opensight\*.py
```

**Known TODOs in this codebase:**
- `cli.py:425` - TODO: Implement utility metrics display
- `cli.py:432` - TODO: Implement trade metrics display
- `cli.py:439` - TODO: Implement opening duel metrics display
- `ai/coaching.py:886` - TODO: Implement recalculate_stats()

### 2. Debug Artifacts (should NOT be in production)

```cmd
findstr /s /i "\[DEBUG\]" src\opensight\*.py
```

**Known debug artifacts:**
- `api.py:861-870` - Timeline debug logging with logger.info
- `api.py:942-948` - Download endpoint debug logging
- `cache.py:686-695` - Round timeline debug logging

These should be converted to proper `logger.debug()` calls.

### 3. Duplicated Code

Look for:
- Similar functions doing the same thing
- Copy-pasted blocks
- Repeated validation logic

### 4. Large Functions (>50 lines)

Candidates for extraction/refactoring:
- Check `analytics.py` (7,882 lines total)
- Check `api.py` (80KB)

### 5. Missing Tests

Check coverage:
```cmd
set PYTHONPATH=src && pytest tests/ --cov=opensight --cov-report=term-missing
```

### 6. Outdated Comments

Comments that don't match the code behavior.

## Output Format

```
## Technical Debt Report

### Priority 1 (Fix This Sprint)
- [file:line] Description

### Priority 2 (Fix When Touching File)
- [file:line] Description

### Priority 3 (Backlog)
- [file:line] Description
```

## Action Items

After scanning:
1. Add critical items to your task list
2. Fix small issues immediately if time permits
3. Create issues for larger refactoring work
4. Update this list with any new debt discovered

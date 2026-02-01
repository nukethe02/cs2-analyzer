---
description: Staff Engineer planning mode for complex implementations
---

You are a Staff Engineer planning this implementation. Before writing ANY code, create a comprehensive plan.

## Planning Framework

### 1. Scope Analysis

**Files to be modified:**
- List every file that will change
- Flag high-risk files:
  - `api.py` (80KB - high complexity)
  - `analytics.py` (7,882 lines)
  - Any file with >500 lines

**Cross-module dependencies:**
- What modules does this touch?
- Import chain implications?
- Shared dataclasses affected?

### 2. Risk Assessment

**Security implications:**
- [ ] User input handling?
- [ ] Authentication/authorization?
- [ ] Data exposure risks?
- [ ] Rate limiting needed?

**Performance impact:**
- [ ] Hot path affected?
- [ ] Database queries added?
- [ ] Memory usage increase?
- [ ] Caching implications?

**Breaking changes:**
- [ ] API contract changes?
- [ ] Database schema changes?
- [ ] Configuration changes?
- [ ] Backward compatibility?

### 3. Test Strategy

**Existing tests:**
- What tests cover this area?
- Will they break?

**New tests needed:**
- Unit tests for new functions
- Integration tests for workflows
- Edge case coverage

**How to verify:**
- Manual testing steps
- Demo files to test with
- Expected outcomes

### 4. Rollback Plan

- How do we undo if it breaks?
- Feature flag needed?
- Database migration reversible?
- Git revert sufficient?

## Output Template

```markdown
## Implementation Plan: [Feature Name]

### Scope
- Files: [list]
- Risk Level: LOW / MEDIUM / HIGH
- Estimated Changes: [X files, ~Y lines]

### Approach
[Describe the implementation approach]

### Steps
1. [First step with file:function specifics]
2. [Second step]
...

### Tests
- [ ] [Test 1]
- [ ] [Test 2]

### Rollback
[How to undo]

### Open Questions
- [Any uncertainties to resolve]
```

## When to Use This

- New feature implementation
- Changes touching 3+ files
- Any change to api.py or analytics.py
- Security-related changes
- Database schema changes

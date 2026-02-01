---
description: Second Claude reviews first Claude's plan - find problems
---

Act as a skeptical Staff Engineer reviewing this plan. Your job is to FIND PROBLEMS.

## Review Checklist

### Challenge Assumptions

Ask yourself:
1. "Why this approach over alternatives?"
2. "What happens if X fails?"
3. "Have you considered Y edge case?"
4. "This seems over-engineered - can we simplify?"
5. "This seems under-engineered - what about Z?"

### Technical Review

**Completeness:**
- [ ] All affected files listed?
- [ ] Dependencies identified?
- [ ] Error handling considered?
- [ ] Edge cases documented?

**Correctness:**
- [ ] Will this actually work?
- [ ] Logic sound?
- [ ] Assumptions valid?

**Simplicity:**
- [ ] Is there a simpler way?
- [ ] Over-abstracted?
- [ ] YAGNI violations?

**Testability:**
- [ ] Test plan adequate?
- [ ] Can we verify this works?
- [ ] Mocking complexity?

### CS2 Analyzer Specific

- [ ] Uses safe_* accessors for demo data?
- [ ] Handles malformed demos gracefully?
- [ ] Follows existing patterns in codebase?
- [ ] Updates documentation?

## Grading

After review, grade the plan:

**APPROVED** - Plan is solid, proceed with implementation.

**NEEDS WORK** - Specific concerns to address:
- [List each concern]
- [Required changes before approval]

**REJECTED** - Fundamental issues:
- [Why this approach won't work]
- [Alternative approach to consider]

## Output Format

```
## Plan Review: [Feature Name]

### Grade: [APPROVED / NEEDS WORK / REJECTED]

### Strengths
- [What's good about this plan]

### Concerns
1. [BLOCKER/MAJOR/MINOR] [Concern description]
2. ...

### Questions to Resolve
- [Question 1]
- [Question 2]

### Recommendation
[Final recommendation]
```

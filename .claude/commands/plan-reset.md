---
description: Scrap current approach and implement the elegant solution
---

# Plan Reset - The Elegant Solution

Knowing everything you know now about this problem, SCRAP the current approach entirely.

## When to Use This

- Current fix feels hacky
- Too many edge cases being patched
- Code is getting harder to understand
- You're fighting the architecture
- Sunk cost is driving decisions

## Reset Process

### Step 1: Step Back

Ask yourself:
1. **What is the ACTUAL problem we're solving?**
   - Not the symptoms
   - Not the current implementation's limitations
   - The root problem

2. **What are the constraints?**
   - What MUST remain unchanged?
   - What are real constraints vs assumed ones?

3. **What does success look like?**
   - How would we know this is solved?
   - What's the simplest test case?

### Step 2: Fresh Perspective

Imagine you're explaining this to a new team member:
- "We need to [actual goal]"
- "Currently it does [current behavior]"
- "The ideal solution would [desired outcome]"

### Step 3: Consider Alternatives

List at least 3 different approaches:
1. The current approach (for comparison)
2. A completely different architecture
3. A minimal/simple approach

For each, ask:
- How much code?
- How testable?
- How maintainable?
- What could go wrong?

### Step 4: Implement Fresh

Pick the best approach and implement from scratch:
- Don't copy-paste from the old solution
- Write new tests first
- Keep it simple

## The 10x Engineer Question

> "What would a 10x engineer do differently?"

Usually:
- Solve a simpler version of the problem
- Use existing libraries/patterns
- Avoid premature optimization
- Make it work, make it right, make it fast (in that order)

## Red Flags to Watch For

Current approach might need reset if:
- [ ] More than 3 edge cases being handled
- [ ] Nested if/else deeper than 2 levels
- [ ] Function longer than 50 lines
- [ ] You're adding comments to explain tricky logic
- [ ] Tests are hard to write
- [ ] Similar code appearing in multiple places

## Example Reset

**Before (hacky):**
```python
def process_data(data):
    if data is None:
        return default
    if isinstance(data, list):
        if len(data) == 0:
            return default
        # Handle list case with 50 more lines...
    elif isinstance(data, dict):
        # Handle dict case with 50 more lines...
    # ... more special cases
```

**After (elegant):**
```python
def process_data(data):
    normalized = normalize_input(data)  # Single source of truth
    return transform(normalized)        # Simple transformation
```

## Commit Strategy

When resetting:
1. Create a new branch
2. Implement the fresh solution
3. Run all tests
4. Compare behavior with original
5. If better, merge and delete old code
6. Don't keep "backup" code commented out

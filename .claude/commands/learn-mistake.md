---
description: Learn from mistake and update CLAUDE.md
---

# Learn From Mistake

I just made a mistake and you corrected me. Before we continue, let's capture this lesson.

## Process

### Step 1: Identify Root Cause

What was the actual mistake?
- Not a symptom, the root cause
- Why did I make this mistake?
- What information was I missing?

### Step 2: Formulate Rule

Write a concise rule to prevent this in future:
- Specific and actionable
- Easy to remember
- Includes the "why"

Format:
```
[Category]: [Rule] - [Brief explanation why]
```

### Step 3: Update CLAUDE.md

Add the rule to the "Lessons Learned" section:

```markdown
## Lessons Learned (Claude Updates This Section)

### Common Mistakes
- [ ] [Category]: [Rule] (added YYYY-MM-DD)
```

### Step 4: Confirm

Read back the rule to confirm it captures the lesson.

## Example Lessons

### Demo Parsing
```markdown
- [ ] Parser: Always use safe_int/safe_float for demo values - raw values can be NaN (added 2024-01-15)
- [ ] Parser: Check DataFrame is not empty before accessing columns - empty demos exist (added 2024-01-16)
```

### API Development
```markdown
- [ ] API: Rate limit decorator must come BEFORE @app.route - order matters (added 2024-01-17)
- [ ] API: Always validate Steam IDs with validate_steam_id() - format varies (added 2024-01-18)
```

### Testing
```markdown
- [ ] Test: Windows uses 'set PYTHONPATH=src' not 'export' - cross-platform (added 2024-01-19)
- [ ] Test: Mock external APIs in tests - network calls flaky (added 2024-01-20)
```

### Git/Workflow
```markdown
- [ ] Git: Never use 'git add .' - review files individually (added 2024-01-21)
- [ ] Git: Check for .env files before committing - secrets (added 2024-01-22)
```

## Categories

Use these categories for consistency:
- **Parser**: Demo parsing issues
- **API**: API development
- **Analysis**: Metric calculations
- **Test**: Testing issues
- **Git**: Version control
- **Deploy**: Deployment issues
- **Security**: Security concerns
- **Performance**: Performance issues

## Why This Matters

> "After every correction, end with: 'Update your CLAUDE.md so you don't make that mistake again.'"
> — Boris Cherny

Claude is "eerily good at writing rules for itself." By capturing mistakes as rules:
1. Same mistake won't happen twice
2. Rules compound over time
3. CLAUDE.md becomes project-specific knowledge base

---
description: Verify changes work - show evidence, not assurances
---

Prove that my changes work correctly. Don't just run tests - VERIFY behavior with evidence.

## Verification Process

### 1. Before State

Capture the behavior on main branch:
```cmd
git stash
git checkout main
```

Run the relevant code path and capture output:
- What does the function return?
- What does the API respond?
- What gets logged?

### 2. After State

Switch to feature branch:
```cmd
git checkout -
git stash pop
```

Run the same code path:
- What does it return now?
- What's different?

### 3. Diff Analysis

Compare the outputs:
- Are changes expected?
- Any unexpected side effects?
- Regression detected?

### 4. Edge Cases

Test boundary conditions:
- Empty input
- None/null values
- NaN values (common in demo data)
- Maximum values
- Invalid input

### 5. Integration Check

Does it work with the rest of the system?
- API endpoint still works?
- CLI command works?
- Frontend displays correctly?

## Evidence Requirements

For each claim, provide:

```markdown
## Proof: [What we're proving]

### Test Command
[Actual command run]

### Output
[Actual output - copy/paste]

### Expected vs Actual
- Expected: [what should happen]
- Actual: [what happened]
- Verdict: PASS / FAIL
```

## CS2 Analyzer Specific Tests

### For Metric Changes
```python
# Test with real demo data structure
test_data = {"kills": [...], "deaths": [...]}
result = calculate_metric(test_data)
print(f"Result: {result}")
```

### For API Changes
```cmd
curl -X POST http://localhost:7860/your-endpoint -H "Content-Type: application/json" -d "{...}"
```

### For Parser Changes
```cmd
set PYTHONPATH=src && python -c "from opensight.core.parser import DemoParser; p = DemoParser(); print(p.parse('test.dem'))"
```

## Rules

- Show actual output, not paraphrased descriptions
- Run commands, don't guess outcomes
- Test both happy path AND error paths
- If you can't prove it, say so explicitly

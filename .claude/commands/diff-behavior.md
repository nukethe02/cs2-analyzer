---
description: Compare behavior between main and feature branch
---

# Diff Behavior Between Branches

Compare actual behavior between main and your feature branch to catch regressions.

## Process

### Step 1: Capture Main Branch Behavior

```cmd
git stash
git checkout main
```

Run the code path you're testing:
```cmd
set PYTHONPATH=src && python -c "
from opensight.analysis.analytics import DemoAnalyzer
# Your test code here
print(result)
" > main_output.txt
```

Or for API:
```cmd
curl http://localhost:7860/your-endpoint > main_output.json
```

### Step 2: Capture Feature Branch Behavior

```cmd
git checkout -
git stash pop
```

Run the same code:
```cmd
set PYTHONPATH=src && python -c "
from opensight.analysis.analytics import DemoAnalyzer
# Same test code
print(result)
" > feature_output.txt
```

### Step 3: Compare Outputs

**Text diff:**
```cmd
fc main_output.txt feature_output.txt
```

**JSON diff (if JSON output):**
```cmd
python -c "
import json
main = json.load(open('main_output.json'))
feature = json.load(open('feature_output.json'))
# Compare specific fields
"
```

### Step 4: Analyze Differences

For each difference:
- [ ] Is this expected?
- [ ] Is this an improvement?
- [ ] Is this a regression?
- [ ] Does it break any contract?

## What to Compare

### For Metric Changes
- Same input → same output (or better)?
- Edge cases handled?
- NaN handling consistent?

### For API Changes
- Response schema unchanged?
- Status codes correct?
- Error messages appropriate?

### For Parser Changes
- Same demo → same data?
- Missing data handled?
- Performance acceptable?

## Automated Comparison Script

Create `scripts/diff-behavior.py`:
```python
import subprocess
import json
import sys

def capture_output(branch, command):
    subprocess.run(["git", "checkout", branch], capture_output=True)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

def main():
    command = sys.argv[1]
    main_out = capture_output("main", command)
    feature_out = capture_output("-", command)

    if main_out == feature_out:
        print("No behavior change detected")
    else:
        print("BEHAVIOR CHANGED:")
        print("Main:", main_out[:500])
        print("Feature:", feature_out[:500])
```

## Common Regressions to Check

1. **Metric values changed unexpectedly**
   - TTD, CP, HLTV Rating should be stable for same input

2. **Missing data in response**
   - Fields that existed before should still exist

3. **Error handling changed**
   - Same invalid input should produce same error

4. **Performance regression**
   - Time the operation on both branches

## Cleanup

After comparing:
```cmd
del main_output.txt feature_output.txt
```

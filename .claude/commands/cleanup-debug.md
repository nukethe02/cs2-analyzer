---
description: Remove debug logging artifacts from production code
---

# Cleanup Debug Artifacts

Remove debug logging that shouldn't be in production.

## Known Debug Artifacts

### api.py

**Lines 861-870** - Timeline debug logging:
```python
# DEBUG: Log timeline data before storing result
logger.info(f"[DEBUG] API: round_timeline has {len(timeline)} rounds")
```

**Lines 942-948** - Download endpoint debug logging:
```python
# DEBUG: Log what we're sending to frontend
logger.info(f"[DEBUG] Download endpoint: Sending {len(timeline)} rounds")
```

### cache.py

**Lines 686-695** - Round timeline debug logging:
```python
# DEBUG: Log timeline details
logger.info(f"[DEBUG] round_timeline length: {len(round_timeline)}")
```

## How to Fix

### Option 1: Remove entirely
If the logging isn't needed, delete the lines.

### Option 2: Convert to proper debug level
Change from:
```python
logger.info(f"[DEBUG] Something: {value}")
```

To:
```python
logger.debug(f"Something: {value}")
```

This way it only shows when debug logging is enabled.

## Cleanup Process

1. **Find all debug artifacts:**
```cmd
findstr /s /i "\[DEBUG\]" src\opensight\*.py
```

2. **Review each occurrence:**
   - Is this logging valuable for debugging?
   - Does it contain sensitive data?
   - Should it be debug level or removed?

3. **Make changes:**
   - Remove unnecessary logging
   - Convert to logger.debug() if needed
   - Ensure no sensitive data logged

4. **Test:**
```cmd
set PYTHONPATH=src && pytest tests/test_api.py -v
```

5. **Verify no new debug artifacts:**
```cmd
findstr /s /i "\[DEBUG\]" src\opensight\*.py
```

## Guidelines

- Production logs should be INFO level and above
- DEBUG level for development troubleshooting only
- Never log:
  - API keys
  - Full Steam IDs (truncate to first 8 chars)
  - User passwords/tokens
  - Full file paths with usernames
- Always log:
  - Security events (auth failures)
  - Error conditions
  - Important state changes

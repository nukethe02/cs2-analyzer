---
description: Run the CS2 Analyzer test suite with proper PYTHONPATH
---

Run the CS2 Analyzer test suite.

## Windows Commands

**All tests:**
```cmd
set PYTHONPATH=src && pytest tests/ -v --tb=short
```

**Specific module:**
```cmd
set PYTHONPATH=src && pytest tests/test_{module}.py -v
```

**With coverage:**
```cmd
set PYTHONPATH=src && pytest tests/ --cov=opensight --cov-report=term-missing
```

## Test File Reference

| File | Tests |
|------|-------|
| test_api.py | API endpoints, security headers |
| test_analytics.py | DemoAnalyzer, metrics engine |
| test_hltv_rating.py | HLTV 2.0 formula accuracy |
| test_metrics.py | TTD, CP, utility calculations |
| test_replay.py | 2D replay generation |
| test_sharecode.py | Share code encode/decode |
| test_your_match.py | Personal dashboard features |
| test_combat.py | Combat metrics |
| test_economy.py | Economy detection |
| test_parser.py | Demo parsing |

## If Tests Fail

1. Read the failure message carefully
2. Check if it's a real bug or test environment issue
3. Fix the code, not the test (unless test is wrong)
4. Re-run to verify fix
5. Never commit with failing tests

## Common Issues

- **Module not found**: Ensure `PYTHONPATH=src` is set
- **Import errors**: Run `pip install -e ".[dev]"` first
- **Slow tests**: Use `-x` flag to stop on first failure

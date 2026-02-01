---
description: Security audit for CS2 Analyzer changes - OWASP Top 10
---

# Security Audit

Check changes against OWASP Top 10 and CS2 Analyzer specific concerns.

## OWASP Top 10 Checklist

### 1. Injection
- [ ] SQL injection in database queries?
- [ ] Command injection in subprocess calls?
- [ ] XSS in any user-displayed content?
- [ ] Path traversal in file operations?

**Check for:**
```python
# BAD - SQL injection
cursor.execute(f"SELECT * FROM users WHERE id = {user_input}")

# GOOD - Parameterized
cursor.execute("SELECT * FROM users WHERE id = ?", (user_input,))
```

### 2. Broken Authentication
- [ ] Authentication bypass possible?
- [ ] Session management issues?
- [ ] Credential exposure?

### 3. Sensitive Data Exposure
- [ ] Secrets in logs?
- [ ] API keys exposed?
- [ ] PII in responses?

**Check for:**
```python
# BAD
logger.info(f"User {user_id} with API key {api_key}")

# GOOD
logger.info(f"User {user_id[:8]}... authenticated")
```

### 4. XML External Entities (XXE)
- [ ] Any XML parsing?
- [ ] Using defusedxml?

### 5. Broken Access Control
- [ ] Can users access others' data?
- [ ] Steam ID isolation?
- [ ] Demo file access control?

### 6. Security Misconfiguration
- [ ] Debug mode disabled in production?
- [ ] Verbose error messages to users?
- [ ] Default credentials?

### 7. Cross-Site Scripting (XSS)
- [ ] User input echoed without escaping?
- [ ] HTML content properly sanitized?

**Check static/index.html for:**
```javascript
// Must use escapeHtml() for user content
element.innerHTML = escapeHtml(userInput);
```

### 8. Insecure Deserialization
- [ ] Using pickle.loads on untrusted data?
- [ ] yaml.load without safe_load?
- [ ] JSON parsing of untrusted input?

### 9. Vulnerable Dependencies
Run:
```cmd
safety check
pip-audit
```

### 10. Insufficient Logging
- [ ] Security events logged?
- [ ] No sensitive data in logs?
- [ ] Audit trail for important actions?

## CS2 Analyzer Specific

### Input Validation
- [ ] `validate_steam_id()` used for all Steam IDs?
- [ ] `validate_demo_id()` used for demo IDs?
- [ ] `validate_job_id()` used for job IDs?

### File Handling
- [ ] Only .dem and .dem.gz accepted?
- [ ] File size limit enforced (500MB)?
- [ ] Temp files cleaned up in finally block?

### Rate Limiting
- [ ] Resource-intensive endpoints rate limited?
- [ ] `/analyze` - 5/min
- [ ] `/replay/generate` - 3/min

### Security Headers
Check `security_headers_middleware()`:
- [ ] CSP frame-ancestors set?
- [ ] X-Content-Type-Options: nosniff?
- [ ] X-Frame-Options set?

## Quick Security Scan

```cmd
bandit -r src/opensight -ll
```

## Output Format

```markdown
## Security Audit Report

### Critical Issues
- [CRITICAL] [Description] - [File:Line]

### High Priority
- [HIGH] [Description] - [File:Line]

### Medium Priority
- [MEDIUM] [Description] - [File:Line]

### Low Priority
- [LOW] [Description] - [File:Line]

### Passed Checks
- [PASS] All inputs validated
- [PASS] No hardcoded secrets
...
```

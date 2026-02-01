---
description: Deploy CS2 Analyzer to Hugging Face Spaces
---

# Deploy to Hugging Face Spaces

Deploy the CS2 Analyzer to HuggingFace Spaces.

## Pre-flight Checklist

- [ ] All tests pass: `/run-tests`
- [ ] Code formatted: `/format`
- [ ] No debug artifacts: `/cleanup-debug`
- [ ] Committed to main branch
- [ ] No sensitive data in code

## Deploy Process

### 1. Ensure HF Remote is Set Up

```cmd
git remote -v
```

If `hf` remote not present:
```cmd
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
```

### 2. Push to HuggingFace

```cmd
git push hf main
```

Or if using GitHub Actions (deploy-hf.yml):
The deployment happens automatically on push to main.

### 3. Monitor Build

1. Go to HuggingFace Spaces dashboard
2. Check the "Logs" tab
3. Watch for build completion

### 4. Verify Deployment

**Health check:**
```cmd
curl https://YOUR_SPACE.hf.space/health
```

**Readiness check:**
```cmd
curl https://YOUR_SPACE.hf.space/readiness
```

**Quick API test:**
```cmd
curl https://YOUR_SPACE.hf.space/about
```

## Rollback if Needed

### Quick Rollback
```cmd
git revert HEAD
git push hf main
```

### Rollback to Specific Commit
```cmd
git push hf <commit-hash>:main --force
```

## Environment Variables

Required secrets in HF Spaces settings:
- `OPENAI_API_KEY` (for AI coaching features)
- `HF_TOKEN` (for HF API access)

Optional:
- `OPENSIGHT_LOG_LEVEL` (default: INFO)
- `OPENSIGHT_CACHE_DIR` (default: /tmp/opensight/cache)

## Dockerfile Reference

The deployment uses `Dockerfile`:
- Multi-stage build
- Port 7860
- Non-root user (appuser)
- Health check endpoint
- uvicorn with 1 worker

## Common Issues

### Build Fails - demoparser2
demoparser2 requires Rust compilation. The Dockerfile handles this.

### Memory Issues
HF Spaces free tier has limited memory. Large demos may fail.

### Rate Limits
HF Spaces has connection limits. Rate limiting in app helps.

### Missing Secrets
Check HF Spaces settings for required environment variables.

## Deployment Logs Location

1. HF Spaces Dashboard > Your Space > Logs
2. Or use HF CLI: `huggingface-cli repo logs <space>`

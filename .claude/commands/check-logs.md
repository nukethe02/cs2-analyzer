---
description: Debug deployment issues by checking logs
---

# Check Logs for Debugging

Debug CS2 Analyzer deployment issues.

## HuggingFace Spaces

### View Build Logs
1. Go to your Space on huggingface.co
2. Click "Logs" tab
3. Look for build errors

### View Runtime Logs
Same location - switch to "Container logs"

### Common Patterns to Search

```
ERROR
Exception
Traceback
ImportError
ModuleNotFoundError
```

## Docker (Local)

### Find Container ID
```cmd
docker ps
docker ps -a  # Include stopped
```

### View Logs
```cmd
docker logs <container_id>
docker logs <container_id> --tail 100
docker logs <container_id> -f  # Follow
```

### Search for Errors
```cmd
docker logs <container_id> 2>&1 | findstr /i "error exception traceback"
```

### Enter Container for Debugging
```cmd
docker exec -it <container_id> /bin/bash
```

## Health Check Debugging

### Check Health Endpoint
```cmd
curl http://localhost:7860/health
```

Expected response:
```json
{"status": "healthy"}
```

### Check Readiness
```cmd
curl http://localhost:7860/readiness
```

### Detailed Status
```cmd
curl http://localhost:7860/cache/stats
```

## Common Issues and Solutions

### 1. ImportError: demoparser2
**Cause:** Rust not available or compilation failed
**Solution:** Check Dockerfile has Rust build stage

### 2. SQLite Database Locked
**Cause:** Multiple workers accessing same DB
**Solution:** Ensure single worker or use different DB path

### 3. Out of Memory
**Cause:** Large demo file or memory leak
**Solution:** Check demo size limits, review memory usage

### 4. Permission Denied
**Cause:** Writing to read-only path
**Solution:** Check OPENSIGHT_CACHE_DIR is writable

### 5. API Key Missing
**Cause:** OPENAI_API_KEY not set
**Solution:** Add to HF Spaces secrets

### 6. Rate Limited
**Cause:** Too many requests
**Solution:** Check rate limiting configuration

## Log Analysis Checklist

1. [ ] Find the first error in the logs
2. [ ] Identify the full stack trace
3. [ ] Check what request triggered it
4. [ ] Look for patterns (same error repeated?)
5. [ ] Check if it's reproducible locally

## Reproduce Locally

```cmd
docker build -t opensight .
docker run -p 7860:7860 opensight
curl http://localhost:7860/health
```

## Environment Debugging

Check what environment variables are set:
```cmd
docker exec <container_id> env | findstr OPENSIGHT
```

---
description: Guide for adding a new API endpoint to CS2 Analyzer
---

# Adding a New API Endpoint to CS2 Analyzer

Follow the established pattern in api.py.

## Pre-Implementation Questions

1. **What endpoint?** (path, e.g., `/api/your-feature`)
2. **HTTP method?** (GET, POST, PUT, DELETE)
3. **What does it accept?** (request body, query params)
4. **What does it return?** (response schema)
5. **Resource intensive?** (needs rate limiting?)

## Implementation Checklist

### Step 1: Define Request/Response Models

```python
from pydantic import BaseModel, Field

class YourRequest(BaseModel):
    """Request model for your endpoint."""
    param1: str = Field(..., description="Description")
    param2: Optional[int] = Field(None, description="Optional param")

class YourResponse(BaseModel):
    """Response model for your endpoint."""
    result: str
    data: dict
```

### Step 2: Add Validation (if needed)

Use existing validators or create new ones:

```python
def validate_your_param(value: str) -> str:
    """Validate your parameter."""
    if not value or len(value) < 1:
        raise HTTPException(status_code=400, detail="Invalid param")
    # Add validation logic
    return value
```

### Step 3: Define Route

```python
@app.post("/api/your-endpoint")
@rate_limit("5/minute")  # Add if resource-intensive
async def your_endpoint(request: YourRequest) -> YourResponse:
    """
    Your endpoint description.

    Args:
        request: The request body

    Returns:
        YourResponse with result

    Raises:
        HTTPException: On invalid input or processing error
    """
    try:
        # Validate inputs
        validated_param = validate_your_param(request.param1)

        # Process
        result = process_your_request(validated_param)

        return YourResponse(result="success", data=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in your_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal error")
```

### Step 4: Add Security Headers (if returning HTML)

Check `security_headers_middleware()` in api.py if your endpoint returns HTML content.

### Step 5: Write Tests (`tests/test_api.py`)

```python
def test_your_endpoint_success(test_client):
    """Test successful request."""
    response = test_client.post(
        "/api/your-endpoint",
        json={"param1": "valid", "param2": 123}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "success"

def test_your_endpoint_invalid_input(test_client):
    """Test invalid input handling."""
    response = test_client.post(
        "/api/your-endpoint",
        json={"param1": ""}  # Invalid
    )
    assert response.status_code == 400

def test_your_endpoint_rate_limit(test_client):
    """Test rate limiting."""
    # Make 6 requests quickly
    for i in range(6):
        response = test_client.post(
            "/api/your-endpoint",
            json={"param1": "valid"}
        )
    # 6th should be rate limited
    assert response.status_code == 429
```

### Step 6: Update Documentation

1. **CLAUDE.md** - Add to API Endpoints table:
```markdown
| POST | `/api/your-endpoint` | Description |
```

2. **`/about` endpoint** - Add to the response if applicable

### Step 7: Verify

```cmd
ruff format src/ tests/ && ruff check --fix src/ tests/
set PYTHONPATH=src && pytest tests/test_api.py -v -k "your_endpoint"
```

## Existing Patterns for Reference

| Endpoint | Method | Rate Limit | Validation |
|----------|--------|------------|------------|
| /analyze | POST | 5/min | validate_demo_id |
| /decode | POST | None | Input validation |
| /api/your-match/{demo_id}/{steam_id} | GET | None | Both params validated |
| /replay/generate | POST | 3/min | Complex validation |

## Common Pitfalls

- [ ] Rate limit decorator MUST come BEFORE @app.route
- [ ] Always wrap in try/except
- [ ] Return proper HTTPException, not raw errors
- [ ] Add test for both success and failure cases
- [ ] Update test_api.py security header tests if adding HTML endpoint

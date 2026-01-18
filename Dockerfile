FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pandas numpy || true
RUN pip install --no-cache-dir demoparser2 || echo "demoparser2 failed, continuing"
RUN pip install --no-cache-dir watchdog rich typer || true

# Copy source code
COPY src/ ./src/

# Expose port 7860
EXPOSE 7860

# Set Python path so imports work
ENV PYTHONPATH="/app/src"

# Run the app
CMD ["python", "-m", "uvicorn", "opensight.api:app", "--host", "0.0.0.0", "--port", "7860"]

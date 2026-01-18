FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Rust for demoparser2
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir fastapi uvicorn python-multipart pandas numpy

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir -e . || echo "Package install failed, continuing anyway"

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Set Python path
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Run the app
CMD ["uvicorn", "opensight.api:app", "--host", "0.0.0.0", "--port", "7860"]

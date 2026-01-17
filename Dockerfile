FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the app
CMD ["uvicorn", "opensight.api:app", "--host", "0.0.0.0", "--port", "7860"]

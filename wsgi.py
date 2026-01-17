"""WSGI entry point for Render deployment."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from opensight.api import app

# For gunicorn with uvicorn workers
# Run with: gunicorn wsgi:app -k uvicorn.workers.UvicornWorker

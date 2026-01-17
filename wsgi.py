"""WSGI entry point for Render deployment."""

from opensight.api import app

# For gunicorn with uvicorn workers
# Run with: gunicorn wsgi:app -k uvicorn.workers.UvicornWorker

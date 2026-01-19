"""
WSGI entry point for production deployment.

This file is used by WSGI servers like Gunicorn to run the application.
"""

from opensight.web.app import create_app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    app.run()

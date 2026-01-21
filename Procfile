# Flask deployment (Render.com, Heroku)
# Sets PYTHONPATH for consistent imports with opensight.* module path
web: PYTHONPATH=src gunicorn "opensight.web.app:create_app()" --bind 0.0.0.0:$PORT

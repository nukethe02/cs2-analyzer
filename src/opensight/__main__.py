"""
OpenSight CLI Entry Point

Allows running the package as a module: python -m opensight
"""

import sys


def main():
    """Main entry point for the CLI."""
    try:
        from opensight.cli import app

        app()
    except ImportError as e:
        # Fallback if typer/rich not installed
        print(f"Error: Missing dependencies for CLI. {e}")
        print("Install with: pip install typer rich")
        sys.exit(1)


if __name__ == "__main__":
    main()

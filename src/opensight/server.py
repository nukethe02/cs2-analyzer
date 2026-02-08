"""
OpenSight Web Server Entry Point

Provides the `opensight-web` command to start the FastAPI server.

Usage:
    opensight-web                    # Start on default port 7860
    opensight-web --port 8000        # Start on custom port
    opensight-web --host 127.0.0.1   # Bind to localhost only
    opensight-web --reload           # Enable auto-reload for development
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the OpenSight web server."""
    parser = argparse.ArgumentParser(
        description="OpenSight CS2 Demo Analyzer - Web Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    opensight-web                     Start server on http://0.0.0.0:7860
    opensight-web --port 8000         Start on port 8000
    opensight-web --host 127.0.0.1    Bind to localhost only
    opensight-web --reload            Enable auto-reload (development)
        """,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to (default: 7860)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, demo parsing is CPU-heavy)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install 'uvicorn[standard]'")
        sys.exit(1)

    logger.info("Starting OpenSight web server on http://%s:%s", args.host, args.port)
    logger.info("Press Ctrl+C to stop")

    uvicorn.run(
        "opensight.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

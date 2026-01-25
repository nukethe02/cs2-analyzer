"""
Flask Application Factory for OpenSight Web

Creates and configures the Flask application with all routes,
security settings, and error handlers.
"""

import logging
import os
import secrets
from datetime import datetime
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

# Configuration
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
ALLOWED_EXTENSIONS = {"dem"}
UPLOAD_FOLDER = Path("/tmp/opensight_uploads")


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(config: dict | None = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured Flask application
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # Security configuration
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(32))
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

    # Session security
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_ENV") == "production"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # Apply custom config
    if config:
        app.config.update(config)

    # Ensure upload folder exists
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # Register routes
    register_routes(app)

    # Register error handlers
    register_error_handlers(app)

    return app


def register_routes(app: Flask) -> None:
    """Register all application routes."""

    @app.route("/")
    def index():
        """Home page with dashboard."""
        return render_template("index.html")

    @app.route("/analyze", methods=["GET", "POST"])
    def analyze():
        """Demo file analysis page."""
        if request.method == "POST":
            return handle_demo_upload()
        return render_template("analyze.html")

    @app.route("/tactical", methods=["POST"])
    def tactical_analysis():
        """Tactical demo review page."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        try:
            # Import here to avoid circular imports
            from opensight.analysis.tactical_service import TacticalAnalysisService
            from opensight.core.parser import DemoParser

            # Parse demo
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{filename}"
            filepath = Path(UPLOAD_FOLDER) / safe_filename

            filepath.parent.mkdir(parents=True, exist_ok=True)
            file.save(str(filepath))
            
            try:
                parser = DemoParser(filepath)
                demo_data = parser.parse()

                # Tactical analysis
                service = TacticalAnalysisService(demo_data)
                summary = service.analyze()

                # Render tactical view
                return render_template(
                    "tactical.html",
                    demo_info={
                        "filename": file.filename,
                        "map": demo_data.map_name,
                        "duration": demo_data.duration_seconds,
                        "rounds": len(getattr(demo_data, "round_starts", [])),
                    },
                    tactical_summary={
                        "key_insights": summary.key_insights,
                    },
                    t_stats=summary.t_stats,
                    ct_stats=summary.ct_stats,
                    t_executes=summary.t_executes,
                    buy_patterns=summary.buy_patterns,
                    key_players=summary.key_players,
                    round_plays=summary.round_plays,
                    players=list(summary.player_analysis.values()),
                    t_strengths=summary.t_strengths,
                    t_weaknesses=summary.t_weaknesses,
                    ct_strengths=summary.ct_strengths,
                    ct_weaknesses=summary.ct_weaknesses,
                    t_win_rate=summary.t_win_rate,
                    ct_win_rate=summary.ct_win_rate,
                    execution_success=[("Smoke Exec", 78), ("Flash Entry", 65), ("Anti-Eco", 45)],
                    team_recommendations=summary.team_recommendations,
                    individual_recommendations=summary.individual_recommendations,
                    practice_drills=summary.practice_drills,
                )
            finally:
                # Cleanup temp file
                try:
                    filepath.unlink()
                except:
                    pass

        except Exception as e:
            logger.error(f"Tactical analysis error: {e}", exc_info=True)
            return render_template(
                "error.html",
                error=f"Analysis failed: {str(e)}"
            ), 500

    @app.route("/decode", methods=["GET", "POST"])
    def decode():
        """Share code decoder page."""
        result = None
        error = None
        share_code = ""

        if request.method == "POST":
            share_code = request.form.get("share_code", "").strip()
            if share_code:
                try:
                    from opensight.integrations.sharecode import decode_sharecode

                    info = decode_sharecode(share_code)
                    result = {
                        "match_id": info.match_id,
                        "outcome_id": info.outcome_id,
                        "token": info.token,
                    }
                except Exception as e:
                    error = str(e)

        return render_template("decode.html", result=result, error=error, share_code=share_code)

    @app.route("/team")
    def team():
        """Team stats dashboard."""
        return render_template("team.html")

    @app.route("/about")
    def about():
        """About page."""
        return render_template("about.html")

    @app.route("/api/analyze", methods=["POST"])
    def api_analyze():
        """API endpoint for demo analysis."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Only .dem files allowed"}), 400

        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{filename}"
            filepath = UPLOAD_FOLDER / safe_filename
            file.save(str(filepath))

            # Analyze the demo
            from opensight.analysis.metrics import calculate_comprehensive_metrics
            from opensight.core.parser import DemoParser

            parser = DemoParser(filepath)
            demo_data = parser.parse()
            metrics = calculate_comprehensive_metrics(demo_data)

            # Build response
            result = {
                "success": True,
                "demo_info": {
                    "map": demo_data.map_name,
                    "duration_seconds": demo_data.duration_seconds,
                    "rounds": len(getattr(demo_data, "round_starts", [])),
                    "players": len(demo_data.player_names),
                },
                "players": [],
            }

            for _steam_id, pm in sorted(
                metrics.items(), key=lambda x: x[1].overall_rating(), reverse=True
            ):
                player_data = {
                    "name": pm.player_name,
                    "team": pm.team,
                    "rating": round(pm.overall_rating(), 1),
                }

                if pm.engagement:
                    player_data.update(
                        {
                            "kills": pm.engagement.total_kills,
                            "deaths": pm.engagement.total_deaths,
                            "kd": round(
                                pm.engagement.total_kills / max(pm.engagement.total_deaths, 1), 2
                            ),
                            "hs_percent": round(pm.engagement.headshot_percentage, 1),
                            "dpr": round(pm.engagement.damage_per_round, 1),
                        }
                    )

                if pm.crosshair_placement:
                    player_data["cp_score"] = round(pm.crosshair_placement.placement_score, 1)

                if pm.opening_duels:
                    player_data.update(
                        {
                            "opening_kills": pm.opening_duels.opening_kills,
                            "opening_deaths": pm.opening_duels.opening_deaths,
                        }
                    )

                if pm.trades:
                    player_data["trades"] = pm.trades.trades_completed

                result["players"].append(player_data)

            # Clean up file
            filepath.unlink(missing_ok=True)

            return jsonify(result)

        except Exception as e:
            logger.exception("Error analyzing demo")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/decode", methods=["POST"])
    def api_decode():
        """API endpoint for share code decoding."""
        data = request.get_json()
        if not data or "share_code" not in data:
            return jsonify({"error": "No share code provided"}), 400

        try:
            from opensight.integrations.sharecode import decode_sharecode

            info = decode_sharecode(data["share_code"])
            return jsonify(
                {
                    "success": True,
                    "match_id": info.match_id,
                    "outcome_id": info.outcome_id,
                    "token": info.token,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/health")
    def health():
        """Health check endpoint for deployment."""
        return jsonify({"status": "healthy", "version": "1.0.0"})


def handle_demo_upload():
    """Handle demo file upload and analysis."""
    if "demo_file" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("analyze"))

    file = request.files["demo_file"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("analyze"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Only .dem files are allowed.", "error")
        return redirect(url_for("analyze"))

    try:
        # Save and process
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        filepath = UPLOAD_FOLDER / safe_filename
        file.save(str(filepath))

        # Analyze
        from opensight.analysis.metrics import calculate_comprehensive_metrics
        from opensight.core.parser import DemoParser

        parser = DemoParser(filepath)
        demo_data = parser.parse()
        metrics = calculate_comprehensive_metrics(demo_data)

        # Store in session for display
        session["analysis_result"] = {
            "map": demo_data.map_name,
            "duration": demo_data.duration_seconds,
            "rounds": len(demo_data.round_starts),
            "filename": filename,
        }

        # Build player data
        players = []
        for _steam_id, pm in sorted(
            metrics.items(), key=lambda x: x[1].overall_rating(), reverse=True
        ):
            player = {
                "name": pm.player_name,
                "team": pm.team,
                "rating": round(pm.overall_rating(), 1),
                "kills": pm.engagement.total_kills if pm.engagement else 0,
                "deaths": pm.engagement.total_deaths if pm.engagement else 0,
                "hs_percent": round(pm.engagement.headshot_percentage, 1) if pm.engagement else 0,
                "dpr": round(pm.engagement.damage_per_round, 1) if pm.engagement else 0,
            }
            players.append(player)

        session["analysis_players"] = players

        # Cleanup
        filepath.unlink(missing_ok=True)

        flash("Demo analyzed successfully!", "success")
        return render_template(
            "results.html",
            demo_info=session["analysis_result"],
            players=session["analysis_players"],
        )

    except Exception as e:
        logger.exception("Error processing demo")
        flash(f"Error analyzing demo: {str(e)}", "error")
        return redirect(url_for("analyze"))


def register_error_handlers(app: Flask) -> None:
    """Register error handlers."""

    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", error_code=404, error_message="Page not found"), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template(
            "error.html", error_code=500, error_message="Internal server error"
        ), 500

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(e):
        flash("File too large. Maximum size is 500MB.", "error")
        return redirect(url_for("analyze"))


# Development server
if __name__ == "__main__":
    app = create_app()
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)

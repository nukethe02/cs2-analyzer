"""
OpenSight Web API - Robust CS2 Demo Analyzer

FastAPI application for CS2 demo analysis with professional-grade metrics.

Provides:
- Demo analysis with HLTV 2.0 Rating, KAST%, TTD, Crosshair Placement
- Simple synchronous processing for reliability
- Clean error handling with full logging

Endpoints:
- GET /         HTML interface with drop-zone
- GET /health   Health check
- POST /analyze Upload and analyze demo file
"""

import logging
import tempfile
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

__version__ = "0.4.0"

# Security constants
MAX_FILE_SIZE_MB = 500
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = (".dem", ".dem.gz")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="OpenSight API",
    description="CS2 demo analyzer - professional-grade metrics powered by awpy",
    version=__version__,
)

# Enable CORS for all origins (development mode)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts for JSON serialization."""
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Try to convert to string for unknown types
        return str(obj)


def player_stats_to_dict(player: Any) -> dict:
    """Convert PlayerMatchStats to a JSON-serializable dict."""
    return {
        "steam_id": str(player.steam_id),
        "name": player.name,
        "team": player.team,
        "kills": player.kills,
        "deaths": player.deaths,
        "assists": player.assists,
        "headshots": player.headshots,
        "total_damage": player.total_damage,
        "rounds_played": player.rounds_played,
        "kd_ratio": player.kd_ratio,
        "kd_diff": player.kd_diff,
        "adr": player.adr,
        "headshot_percentage": player.headshot_percentage,
        "kast_percentage": player.kast_percentage,
        "hltv_rating": player.hltv_rating,
        "impact_rating": player.impact_rating,
        "survival_rate": player.survival_rate,
        "kills_per_round": player.kills_per_round,
        "deaths_per_round": player.deaths_per_round,
        # TTD (Time to Damage)
        "ttd_median_ms": round(player.ttd_median_ms, 1) if player.ttd_median_ms else None,
        "ttd_mean_ms": round(player.ttd_mean_ms, 1) if player.ttd_mean_ms else None,
        "ttd_samples": len(player.ttd_values),
        "prefire_count": player.prefire_count,
        # Crosshair Placement
        "cp_median_error_deg": round(player.cp_median_error_deg, 1) if player.cp_median_error_deg else None,
        "cp_mean_error_deg": round(player.cp_mean_error_deg, 1) if player.cp_mean_error_deg else None,
        "cp_samples": len(player.cp_values),
        # Opening duels
        "opening_duel_wins": player.opening_duels.wins,
        "opening_duel_losses": player.opening_duels.losses,
        "opening_duel_attempts": player.opening_duels.attempts,
        "opening_duel_win_rate": player.opening_duels.win_rate,
        # Trades
        "kills_traded": player.trades.kills_traded,
        "deaths_traded": player.trades.deaths_traded,
        # Clutches
        "clutch_situations": player.clutches.total_situations,
        "clutch_wins": player.clutches.total_wins,
        "clutch_win_rate": player.clutches.win_rate,
        # Multi-kills
        "rounds_with_2k": player.multi_kills.rounds_with_2k,
        "rounds_with_3k": player.multi_kills.rounds_with_3k,
        "rounds_with_4k": player.multi_kills.rounds_with_4k,
        "rounds_with_5k": player.multi_kills.rounds_with_5k,
        # Utility
        "flashbangs_thrown": player.utility.flashbangs_thrown,
        "enemies_flashed": player.utility.enemies_flashed,
        "flash_assists": player.utility.flash_assists,
        "he_thrown": player.utility.he_thrown,
        "he_damage": player.utility.he_damage,
        # Weapon breakdown
        "weapon_kills": player.weapon_kills,
    }


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CS2 Demo Analyzer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e4e4e4;
            padding: 2rem;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #ffd700, #ff8c00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 2rem;
        }
        .drop-zone {
            border: 3px dashed #4a5568;
            border-radius: 1rem;
            padding: 4rem 2rem;
            text-align: center;
            background: rgba(255,255,255,0.02);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #ffd700;
            background: rgba(255,215,0,0.05);
        }
        .drop-zone p { font-size: 1.2rem; margin-bottom: 1rem; }
        .drop-zone small { color: #666; }
        #file-input { display: none; }
        .status {
            margin-top: 1.5rem;
            padding: 1rem;
            border-radius: 0.5rem;
            display: none;
        }
        .status.loading { display: block; background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
        .status.error { display: block; background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .status.success { display: block; background: rgba(34, 197, 94, 0.2); color: #4ade80; }
        #results {
            margin-top: 2rem;
            display: none;
        }
        #results.visible { display: block; }
        .result-card {
            background: rgba(255,255,255,0.05);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .result-card h2 { color: #ffd700; margin-bottom: 1rem; font-size: 1.3rem; }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }
        .stat-item { text-align: center; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #fff; }
        .stat-label { font-size: 0.85rem; color: #888; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        th { color: #ffd700; font-weight: 600; }
        tr:hover { background: rgba(255,255,255,0.03); }
        .rating { font-weight: bold; }
        .rating.high { color: #4ade80; }
        .rating.medium { color: #fbbf24; }
        .rating.low { color: #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <h1>CS2 Demo Analyzer</h1>
        <p class="subtitle">Powered by awpy - Professional-grade CS2 analytics</p>

        <div class="drop-zone" id="drop-zone">
            <p>Drop your .dem file here or click to select</p>
            <small>Supports .dem and .dem.gz files up to 500MB</small>
            <input type="file" id="file-input" accept=".dem,.dem.gz">
        </div>

        <div class="status" id="status"></div>

        <div id="results">
            <div class="result-card" id="match-info"></div>
            <div class="result-card" id="player-stats"></div>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const status = document.getElementById('status');
        const results = document.getElementById('results');
        const matchInfo = document.getElementById('match-info');
        const playerStats = document.getElementById('player-stats');

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

        async function handleFile(file) {
            if (!file.name.match(/\\.dem(\\.gz)?$/i)) {
                showStatus('Please select a .dem or .dem.gz file', 'error');
                return;
            }
            showStatus('Analyzing demo... This may take a minute for large files.', 'loading');
            results.classList.remove('visible');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();

                if (!response.ok) {
                    showStatus(data.error || data.detail || 'Analysis failed', 'error');
                    return;
                }

                showStatus('Analysis complete!', 'success');
                displayResults(data);
            } catch (err) {
                showStatus('Network error: ' + err.message, 'error');
            }
        }

        function showStatus(message, type) {
            status.textContent = message;
            status.className = 'status ' + type;
        }

        function getRatingClass(rating) {
            if (rating >= 1.15) return 'high';
            if (rating >= 0.85) return 'medium';
            return 'low';
        }

        function displayResults(data) {
            matchInfo.innerHTML = `
                <h2>Match Information</h2>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-value">${escapeHtml(data.map_name)}</div>
                        <div class="stat-label">Map</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.tick_rate}</div>
                        <div class="stat-label">Tick Rate</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.total_rounds || '-'}</div>
                        <div class="stat-label">Rounds</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.players ? data.players.length : 0}</div>
                        <div class="stat-label">Players</div>
                    </div>
                </div>
            `;

            if (data.players && data.players.length > 0) {
                let tableHtml = `
                    <h2>Player Statistics</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Player</th>
                                <th>Team</th>
                                <th>K/D/A</th>
                                <th>ADR</th>
                                <th>HS%</th>
                                <th>KAST%</th>
                                <th>Rating</th>
                            </tr>
                        </thead>
                        <tbody>
                `;

                data.players.sort((a, b) => b.hltv_rating - a.hltv_rating);

                for (const p of data.players) {
                    const ratingClass = getRatingClass(p.hltv_rating);
                    tableHtml += `
                        <tr>
                            <td>${escapeHtml(p.name)}</td>
                            <td>${escapeHtml(p.team)}</td>
                            <td>${p.kills}/${p.deaths}/${p.assists}</td>
                            <td>${p.adr.toFixed(1)}</td>
                            <td>${p.headshot_percentage.toFixed(1)}%</td>
                            <td>${p.kast_percentage.toFixed(1)}%</td>
                            <td class="rating ${ratingClass}">${p.hltv_rating.toFixed(2)}</td>
                        </tr>
                    `;
                }

                tableHtml += '</tbody></table>';
                playerStats.innerHTML = tableHtml;
            } else {
                playerStats.innerHTML = '<h2>Player Statistics</h2><p>No player data available.</p>';
            }

            results.classList.add('visible');
        }
    </script>
</body>
</html>
"""


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface with drop-zone."""
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_demo(file: UploadFile = File(...)):
    """
    Analyze a CS2 demo file and return player statistics.

    Accepts .dem and .dem.gz files up to 500MB.
    Returns JSON with map_name, tick_rate, and players list.
    """
    tmp_path = None

    try:
        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        filename_lower = file.filename.lower()
        if not filename_lower.endswith(ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: .dem, .dem.gz. Got: {file.filename}"
            )

        # Read file content and check size
        content = await file.read()
        file_size_bytes = len(content)

        if file_size_bytes == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_bytes / (1024*1024):.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB"
            )

        logger.info(f"Received file: {file.filename} ({file_size_bytes / (1024*1024):.1f}MB)")

        # Write to temporary file
        suffix = ".dem.gz" if filename_lower.endswith(".dem.gz") else ".dem"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        logger.info(f"Wrote temp file: {tmp_path}")

        # Import and run analysis
        from opensight.parser import DemoParser
        from opensight.analytics import DemoAnalyzer

        # Parse the demo
        logger.info("Starting demo parsing...")
        parser = DemoParser(tmp_path)
        match_data = parser.parse()
        logger.info(f"Parsed: {len(match_data.kills)} kills, {len(match_data.damages)} damages, {match_data.num_rounds} rounds")

        # Run analytics
        logger.info("Starting analytics...")
        analyzer = DemoAnalyzer(match_data)
        analysis = analyzer.analyze()
        logger.info(f"Analysis complete: {len(analysis.players)} players")

        # Build response
        players_list = [
            player_stats_to_dict(player)
            for player in analysis.get_leaderboard()
        ]

        response = {
            "map_name": analysis.map_name,
            "tick_rate": match_data.tick_rate,
            "total_rounds": analysis.total_rounds,
            "team1_score": analysis.team1_score,
            "team2_score": analysis.team2_score,
            "players": players_list,
        }

        logger.info(f"Returning analysis for {len(players_list)} players on {analysis.map_name}")
        return JSONResponse(content=response)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        # Log full traceback
        tb = traceback.format_exc()
        logger.error(f"Analysis failed: {e}\n{tb}")

        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Analysis failed",
                "detail": str(e)
            }
        )

    finally:
        # Always clean up temporary file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.info(f"Cleaned up temp file: {tmp_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up temp file {tmp_path}: {e}")


# =============================================================================
# Additional Endpoints (for compatibility)
# =============================================================================

@app.get("/readiness")
async def readiness():
    """
    Readiness check for container orchestration.

    Checks disk space, temp directory, and core dependencies.
    """
    import shutil
    import os

    checks = {}
    all_ready = True

    # Check disk space
    try:
        temp_dir = tempfile.gettempdir()
        disk_usage = shutil.disk_usage(temp_dir)
        free_mb = disk_usage.free / (1024 * 1024)
        checks["disk_space"] = {"status": "ok" if free_mb >= 100 else "fail", "free_mb": round(free_mb)}
        if free_mb < 100:
            all_ready = False
    except Exception as e:
        checks["disk_space"] = {"status": "fail", "error": str(e)}
        all_ready = False

    # Check temp directory writable
    try:
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, f"opensight_test_{os.getpid()}.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        checks["temp_writable"] = {"status": "ok", "path": temp_dir}
    except Exception as e:
        checks["temp_writable"] = {"status": "fail", "error": str(e)}
        all_ready = False

    # Check dependencies
    dep_status = {}
    for dep in ["awpy", "pandas", "numpy"]:
        try:
            __import__(dep)
            dep_status[dep] = "ok"
        except ImportError as e:
            dep_status[dep] = f"fail: {e}"
            all_ready = False
    checks["dependencies"] = dep_status

    response = {"ready": all_ready, "version": __version__, "checks": checks}
    return JSONResponse(content=response, status_code=200 if all_ready else 503)


@app.get("/about")
async def about():
    """API documentation and metric descriptions."""
    return {
        "name": "OpenSight",
        "version": __version__,
        "description": "CS2 demo analyzer powered by awpy",
        "endpoints": {
            "GET /": "Web interface with file upload",
            "GET /health": "Health check",
            "GET /readiness": "Readiness check for containers",
            "POST /analyze": "Upload and analyze demo file",
            "GET /about": "This documentation",
        },
        "metrics": {
            "hltv_rating": "HLTV 2.0 Rating (1.0 = average)",
            "kast_percentage": "Rounds with Kill/Assist/Survived/Traded",
            "adr": "Average Damage per Round",
            "ttd_median_ms": "Time to Damage - reaction speed",
            "cp_median_error_deg": "Crosshair Placement - aim accuracy",
        },
    }

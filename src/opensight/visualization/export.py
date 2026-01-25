"""
Export Functionality for OpenSight

Provides multiple export formats for analysis results:
- JSON (default)
- CSV
- Excel (XLSX)
- HTML reports

Each format has its own advantages:
- JSON: Complete data, programmatic access
- CSV: Simple, widely compatible
- Excel: Multi-sheet, formatted
- HTML: Human-readable reports
"""

import csv
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from opensight.analysis.metrics import (
    ComprehensivePlayerMetrics,
    CrosshairPlacementResult,
    EconomyMetrics,
    EngagementMetrics,
    OpeningDuelMetrics,
    PositioningMetrics,
    TradeMetrics,
    TTDResult,
    UtilityMetrics,
)
from opensight.core.parser import DemoData

logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions
# ============================================================================

MetricsDict = dict[
    int,
    EngagementMetrics
    | TTDResult
    | CrosshairPlacementResult
    | EconomyMetrics
    | UtilityMetrics
    | PositioningMetrics
    | TradeMetrics
    | OpeningDuelMetrics
    | ComprehensivePlayerMetrics,
]


# ============================================================================
# Data Conversion Utilities
# ============================================================================


def dataclass_to_dict(obj: Any) -> Any:
    """Convert a dataclass (or nested dataclasses) to a dictionary."""
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for key, value in asdict(obj).items():
            result[key] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten a nested dictionary."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ============================================================================
# JSON Export
# ============================================================================


def export_to_json(
    data: dict[str, Any],
    output_path: Path | None = None,
    indent: int = 2,
    include_metadata: bool = True,
) -> str:
    """
    Export analysis results to JSON format.

    Args:
        data: Analysis results dictionary
        output_path: Optional path to write the file
        indent: JSON indentation level
        include_metadata: Whether to include export metadata

    Returns:
        JSON string
    """
    export_data = dataclass_to_dict(data)

    if include_metadata:
        export_data = {
            "_metadata": {
                "exported_at": datetime.now().isoformat(),
                "format": "opensight_json",
                "version": "1.0",
            },
            **export_data,
        }

    json_str = json.dumps(export_data, indent=indent, default=str)

    if output_path:
        output_path.write_text(json_str)
        logger.info(f"Exported JSON to: {output_path}")

    return json_str


# ============================================================================
# CSV Export
# ============================================================================


def export_metrics_to_csv(
    metrics: MetricsDict,
    output_path: Path | None = None,
    delimiter: str = ",",
    include_header: bool = True,
) -> str:
    """
    Export player metrics to CSV format.

    Args:
        metrics: Dictionary of player metrics
        output_path: Optional path to write the file
        delimiter: CSV delimiter character
        include_header: Whether to include column headers

    Returns:
        CSV string
    """
    if not metrics:
        return ""

    # Convert metrics to flat dictionaries
    rows: list[dict] = []
    for steam_id, metric in metrics.items():
        row = {"steam_id": steam_id}
        metric_dict = dataclass_to_dict(metric)
        flat = flatten_dict(metric_dict)
        row.update(flat)
        rows.append(row)

    if not rows:
        return ""

    # Get all columns
    columns = list(rows[0].keys())

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, delimiter=delimiter)

    if include_header:
        writer.writeheader()

    for row in rows:
        # Handle list values by converting to string
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, list):
                clean_row[k] = ";".join(str(x) for x in v)
            elif isinstance(v, dict):
                clean_row[k] = json.dumps(v)
            else:
                clean_row[k] = v
        writer.writerow(clean_row)

    csv_str = output.getvalue()

    if output_path:
        output_path.write_text(csv_str)
        logger.info(f"Exported CSV to: {output_path}")

    return csv_str


def export_demo_summary_csv(
    demo_data: DemoData,
    metrics: dict[int, ComprehensivePlayerMetrics],
    output_path: Path | None = None,
) -> str:
    """
    Export a summary of demo analysis to CSV.

    Args:
        demo_data: Parsed demo data
        metrics: Comprehensive metrics for all players
        output_path: Optional path to write the file

    Returns:
        CSV string
    """
    rows: list[dict] = []

    for steam_id, player_metrics in metrics.items():
        row = {
            "steam_id": steam_id,
            "player_name": player_metrics.player_name,
            "team": player_metrics.team,
        }

        # Add engagement metrics
        if player_metrics.engagement:
            row["kills"] = player_metrics.engagement.total_kills
            row["deaths"] = player_metrics.engagement.total_deaths
            row["hs_percentage"] = player_metrics.engagement.headshot_percentage
            row["damage_per_round"] = player_metrics.engagement.damage_per_round

        # Add TTD metrics
        if player_metrics.ttd:
            row["ttd_mean_ms"] = player_metrics.ttd.mean_ttd_ms
            row["ttd_median_ms"] = player_metrics.ttd.median_ttd_ms

        # Add crosshair placement
        if player_metrics.crosshair_placement:
            row["cp_mean_angle"] = player_metrics.crosshair_placement.mean_angle_deg
            row["cp_score"] = player_metrics.crosshair_placement.placement_score

        # Add economy
        if player_metrics.economy:
            row["money_spent"] = player_metrics.economy.total_money_spent
            row["weapon_efficiency"] = player_metrics.economy.weapon_efficiency
            row["favorite_weapon"] = player_metrics.economy.favorite_weapon

        # Add utility
        if player_metrics.utility:
            row["grenades_used"] = player_metrics.utility.total_grenades_used
            row["utility_damage"] = player_metrics.utility.utility_damage

        # Add trades
        if player_metrics.trades:
            row["trades_completed"] = player_metrics.trades.trades_completed
            row["deaths_traded"] = player_metrics.trades.deaths_traded

        # Add opening duels
        if player_metrics.opening_duels:
            row["opening_kills"] = player_metrics.opening_duels.opening_kills
            row["opening_deaths"] = player_metrics.opening_duels.opening_deaths
            row["opening_success_rate"] = player_metrics.opening_duels.opening_success_rate

        # Add overall rating
        row["overall_rating"] = player_metrics.overall_rating()

        rows.append(row)

    if not rows:
        return ""

    columns = list(rows[0].keys())

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()

    for row in rows:
        writer.writerow(row)

    csv_str = output.getvalue()

    if output_path:
        output_path.write_text(csv_str)
        logger.info(f"Exported summary CSV to: {output_path}")

    return csv_str


# ============================================================================
# Excel Export
# ============================================================================


def export_to_excel(
    demo_data: DemoData,
    metrics: dict[int, ComprehensivePlayerMetrics],
    output_path: Path,
    include_charts: bool = False,
) -> None:
    """
    Export analysis results to Excel format.

    Creates a multi-sheet workbook with:
    - Summary sheet with overall stats
    - Individual metric sheets
    - Demo information

    Args:
        demo_data: Parsed demo data
        metrics: Comprehensive metrics for all players
        output_path: Path to write the Excel file
        include_charts: Whether to include charts (requires openpyxl)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Excel export") from None

    try:
        import openpyxl as _  # noqa: F401
    except ImportError:
        raise ImportError("openpyxl is required for Excel export") from None

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Demo info sheet
        # Calculate duration_ticks from duration_seconds and tick_rate
        duration_ticks = int(demo_data.duration_seconds * demo_data.tick_rate)
        # Get round count from game_rounds or num_rounds
        round_count = (
            len(demo_data.rounds)
            if hasattr(demo_data, "game_rounds") and demo_data.rounds
            else demo_data.num_rounds
        )
        demo_info = pd.DataFrame(
            [
                {
                    "Map": demo_data.map_name,
                    "Duration (seconds)": demo_data.duration_seconds,
                    "Tick Rate": demo_data.tick_rate,
                    "Total Ticks": duration_ticks,
                    "Rounds": round_count,
                    "Players": len(demo_data.player_names),
                }
            ]
        )
        demo_info.T.to_excel(writer, sheet_name="Demo Info", header=False)

        # Summary sheet
        summary_rows = []
        for _steam_id, pm in metrics.items():
            row = {
                "Player": pm.player_name,
                "Team": pm.team,
                "Rating": round(pm.overall_rating(), 1),
            }

            if pm.engagement:
                row["K"] = pm.engagement.total_kills
                row["D"] = pm.engagement.total_deaths
                row["K/D"] = round(
                    pm.engagement.total_kills / max(pm.engagement.total_deaths, 1), 2
                )
                row["HS%"] = round(pm.engagement.headshot_percentage, 1)
                row["DPR"] = round(pm.engagement.damage_per_round, 1)

            if pm.crosshair_placement:
                row["CP Score"] = round(pm.crosshair_placement.placement_score, 1)

            summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # TTD sheet
        ttd_rows = []
        for _steam_id, pm in metrics.items():
            if pm.ttd:
                ttd_rows.append(
                    {
                        "Player": pm.player_name,
                        "Engagements": pm.ttd.engagement_count,
                        "Mean (ms)": round(pm.ttd.mean_ttd_ms, 0),
                        "Median (ms)": round(pm.ttd.median_ttd_ms, 0),
                        "Min (ms)": round(pm.ttd.min_ttd_ms, 0),
                        "Max (ms)": round(pm.ttd.max_ttd_ms, 0),
                        "Std Dev": round(pm.ttd.std_ttd_ms, 1),
                    }
                )

        if ttd_rows:
            pd.DataFrame(ttd_rows).to_excel(writer, sheet_name="Time to Damage", index=False)

        # Economy sheet
        econ_rows = []
        for _steam_id, pm in metrics.items():
            if pm.economy:
                econ_rows.append(
                    {
                        "Player": pm.player_name,
                        "Money Spent": pm.economy.total_money_spent,
                        "Value Killed": pm.economy.total_value_killed,
                        "Efficiency": round(pm.economy.weapon_efficiency, 2),
                        "Favorite Weapon": pm.economy.favorite_weapon,
                        "Eco Kills": pm.economy.eco_round_kills,
                        "Force Kills": pm.economy.force_buy_kills,
                        "Full Buy Kills": pm.economy.full_buy_kills,
                    }
                )

        if econ_rows:
            pd.DataFrame(econ_rows).to_excel(writer, sheet_name="Economy", index=False)

        # Utility sheet
        util_rows = []
        for _steam_id, pm in metrics.items():
            if pm.utility:
                util_rows.append(
                    {
                        "Player": pm.player_name,
                        "Total Grenades": pm.utility.total_grenades_used,
                        "Smokes": pm.utility.smokes_thrown,
                        "Flashes": pm.utility.flashes_thrown,
                        "HE Grenades": pm.utility.he_grenades_thrown,
                        "Molotovs": pm.utility.molotovs_thrown,
                        "Utility Damage": round(pm.utility.utility_damage, 1),
                        "Efficiency": round(pm.utility.utility_efficiency, 2),
                    }
                )

        if util_rows:
            pd.DataFrame(util_rows).to_excel(writer, sheet_name="Utility", index=False)

        # Trades sheet
        trade_rows = []
        for _steam_id, pm in metrics.items():
            if pm.trades:
                trade_rows.append(
                    {
                        "Player": pm.player_name,
                        "Trades Completed": pm.trades.trades_completed,
                        "Deaths Traded": pm.trades.deaths_traded,
                        "Trade Success %": round(pm.trades.trade_success_rate, 1),
                        "Avg Trade Time (ms)": round(pm.trades.avg_trade_time_ms, 0),
                    }
                )

        if trade_rows:
            pd.DataFrame(trade_rows).to_excel(writer, sheet_name="Trades", index=False)

        # Opening Duels sheet
        opening_rows = []
        for _steam_id, pm in metrics.items():
            if pm.opening_duels:
                opening_rows.append(
                    {
                        "Player": pm.player_name,
                        "Opening Kills": pm.opening_duels.opening_kills,
                        "Opening Deaths": pm.opening_duels.opening_deaths,
                        "Attempts": pm.opening_duels.opening_attempts,
                        "Success %": round(pm.opening_duels.opening_success_rate, 1),
                        "Avg Time (ms)": round(pm.opening_duels.avg_opening_time_ms, 0),
                        "Weapon": pm.opening_duels.opening_weapon,
                    }
                )

        if opening_rows:
            pd.DataFrame(opening_rows).to_excel(writer, sheet_name="Opening Duels", index=False)

    logger.info(f"Exported Excel to: {output_path}")


# ============================================================================
# HTML Report Export
# ============================================================================


def export_to_html(
    demo_data: DemoData,
    metrics: dict[int, ComprehensivePlayerMetrics],
    output_path: Path | None = None,
    title: str | None = None,
) -> str:
    """
    Export analysis results to an HTML report.

    Args:
        demo_data: Parsed demo data
        metrics: Comprehensive metrics for all players
        output_path: Optional path to write the file
        title: Optional custom title

    Returns:
        HTML string
    """
    if title is None:
        title = f"OpenSight Analysis - {demo_data.map_name}"

    # Get round count
    round_count = (
        len(demo_data.rounds)
        if hasattr(demo_data, "game_rounds") and demo_data.rounds
        else demo_data.num_rounds
    )

    # Sort players by rating
    sorted_players = sorted(metrics.values(), key=lambda x: x.overall_rating(), reverse=True)

    # Build player rows
    player_rows = ""
    for pm in sorted_players:
        kills = pm.engagement.total_kills if pm.engagement else 0
        deaths = pm.engagement.total_deaths if pm.engagement else 0
        kd = kills / max(deaths, 1)
        hs = pm.engagement.headshot_percentage if pm.engagement else 0
        dpr = pm.engagement.damage_per_round if pm.engagement else 0
        rating = pm.overall_rating()

        team_class = "ct" if pm.team == "CT" else "t"

        player_rows += f"""
        <tr class="{team_class}">
            <td>{pm.player_name}</td>
            <td>{pm.team}</td>
            <td>{kills}</td>
            <td>{deaths}</td>
            <td>{kd:.2f}</td>
            <td>{hs:.1f}%</td>
            <td>{dpr:.1f}</td>
            <td class="rating">{rating:.1f}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5rem;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .info-card {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }}
        .info-item {{
            text-align: center;
        }}
        .info-item .value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #4ecdc4;
        }}
        .info-item .label {{
            font-size: 0.875rem;
            color: #aaa;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            overflow: hidden;
        }}
        th, td {{
            padding: 1rem;
            text-align: left;
        }}
        th {{
            background: rgba(255, 255, 255, 0.1);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        tr.ct td:first-child {{
            border-left: 3px solid #5c7cfa;
        }}
        tr.t td:first-child {{
            border-left: 3px solid #ff6b6b;
        }}
        .rating {{
            font-weight: bold;
            color: #4ecdc4;
        }}
        .footer {{
            text-align: center;
            margin-top: 2rem;
            color: #666;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>

        <div class="info-card">
            <div class="info-item">
                <div class="value">{demo_data.map_name}</div>
                <div class="label">Map</div>
            </div>
            <div class="info-item">
                <div class="value">{demo_data.duration_seconds / 60:.1f}m</div>
                <div class="label">Duration</div>
            </div>
            <div class="info-item">
                <div class="value">{round_count}</div>
                <div class="label">Rounds</div>
            </div>
            <div class="info-item">
                <div class="value">{len(demo_data.player_names)}</div>
                <div class="label">Players</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Player</th>
                    <th>Team</th>
                    <th>K</th>
                    <th>D</th>
                    <th>K/D</th>
                    <th>HS%</th>
                    <th>DPR</th>
                    <th>Rating</th>
                </tr>
            </thead>
            <tbody>
                {player_rows}
            </tbody>
        </table>

        <div class="footer">
            Generated by OpenSight | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>
"""

    if output_path:
        output_path.write_text(html)
        logger.info(f"Exported HTML to: {output_path}")

    return html


# ============================================================================
# Unified Export Function
# ============================================================================


def export_analysis(
    demo_data: DemoData,
    metrics: dict[int, ComprehensivePlayerMetrics],
    output_path: Path,
    format: str | None = None,
) -> None:
    """
    Export analysis results to the specified format.

    Format is detected from file extension if not specified.

    Args:
        demo_data: Parsed demo data
        metrics: Comprehensive metrics for all players
        output_path: Path to write the export
        format: Optional format override (json, csv, xlsx, html)
    """
    if format is None:
        format = output_path.suffix.lstrip(".").lower()

    if format == "json":
        # Get round count
        json_round_count = (
            len(demo_data.rounds)
            if hasattr(demo_data, "game_rounds") and demo_data.rounds
            else demo_data.num_rounds
        )
        data = {
            "demo_info": {
                "file": str(demo_data.file_path),
                "map": demo_data.map_name,
                "duration_seconds": demo_data.duration_seconds,
                "tick_rate": demo_data.tick_rate,
                "rounds": json_round_count,
            },
            "players": {str(sid): dataclass_to_dict(pm) for sid, pm in metrics.items()},
        }
        export_to_json(data, output_path)

    elif format == "csv":
        export_demo_summary_csv(demo_data, metrics, output_path)

    elif format in ("xlsx", "excel"):
        export_to_excel(demo_data, metrics, output_path)

    elif format in ("html", "htm"):
        export_to_html(demo_data, metrics, output_path)

    elif format == "pdf":
        export_to_pdf(demo_data, metrics, output_path)

    else:
        raise ValueError(f"Unsupported export format: {format}")


# ============================================================================
# PDF Export (FREE - uses built-in HTML conversion or optional reportlab)
# ============================================================================


def export_to_pdf(
    demo_data: DemoData,
    metrics: dict[int, ComprehensivePlayerMetrics],
    output_path: Path | None = None,
) -> bytes:
    """
    Export analysis results to PDF format.

    Uses a pure-Python approach that works without external dependencies.
    If reportlab is available, uses it for better formatting.
    Otherwise, generates an HTML file with print-friendly styles.

    ALL FUNCTIONALITY IS FREE - no paid services required.

    Args:
        demo_data: Parsed demo data
        metrics: Player metrics dictionary
        output_path: Optional path to write PDF

    Returns:
        PDF bytes if reportlab available, HTML bytes otherwise
    """
    try:
        # Try to use reportlab (free, optional)
        return _export_pdf_reportlab(demo_data, metrics, output_path)
    except ImportError:
        # Fallback to print-friendly HTML
        logger.warning(
            "reportlab not installed. Generating print-friendly HTML instead. "
            "Install with: pip install reportlab"
        )
        html = _generate_pdf_html(demo_data, metrics)
        if output_path:
            # Save as HTML with .pdf extension note
            html_path = output_path.with_suffix(".html")
            html_path.write_text(html)
            logger.info(f"Exported print-friendly HTML to: {html_path}")
        return html.encode()


def _export_pdf_reportlab(
    demo_data: DemoData,
    metrics: dict[int, ComprehensivePlayerMetrics],
    output_path: Path | None = None,
) -> bytes:
    """Generate PDF using reportlab (free library)."""
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=24,
        spaceAfter=20,
        textColor=colors.HexColor("#1a1a2e"),
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading1"],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.HexColor("#4a5568"),
    )
    normal_style = styles["Normal"]

    elements = []

    # Title
    elements.append(Paragraph("OpenSight Match Analysis Report", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Match Info
    elements.append(Paragraph("Match Information", heading_style))
    match_info = [
        ["Map", demo_data.map_name],
        ["Duration", f"{demo_data.duration_seconds / 60:.1f} minutes"],
        ["Tick Rate", str(demo_data.tick_rate)],
        ["Total Rounds", str(len(demo_data.round_starts))],
        ["Players", str(len(demo_data.player_names))],
    ]
    match_table = Table(match_info, colWidths=[2 * inch, 3 * inch])
    match_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f7fafc")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#2d3748")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ]
        )
    )
    elements.append(match_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Player Statistics
    elements.append(Paragraph("Player Statistics", heading_style))

    # Sort players by rating
    sorted_players = sorted(
        metrics.items(),
        key=lambda x: x[1].overall_rating(),
        reverse=True,
    )

    # Build player table
    headers = ["Player", "Team", "K", "D", "A", "HS%", "ADR", "KAST", "Rating"]
    player_data = [headers]

    for steam_id, pm in sorted_players:
        player_name = demo_data.player_names.get(steam_id, "Unknown")[:20]
        team = demo_data.player_teams.get(steam_id, "")
        row = [
            player_name,
            team,
            str(pm.engagement.total_kills),
            str(pm.engagement.total_deaths),
            str(pm.engagement.assists),
            f"{pm.engagement.headshot_percentage:.1f}%",
            f"{pm.engagement.damage_per_round:.1f}",
            f"{pm.engagement.kast_percentage:.1f}%",
            f"{pm.overall_rating():.2f}",
        ]
        player_data.append(row)

    player_table = Table(player_data, repeatRows=1)
    player_table.setStyle(
        TableStyle(
            [
                # Header
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                # Body
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                # Alternating rows
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
                # Borders
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(player_table)
    elements.append(Spacer(1, 0.3 * inch))

    # Footer
    footer_text = f"Generated by OpenSight | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elements.append(Paragraph(footer_text, normal_style))

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        output_path.write_bytes(pdf_bytes)
        logger.info(f"Exported PDF to: {output_path}")

    return pdf_bytes


def _generate_pdf_html(
    demo_data: DemoData,
    metrics: dict[int, ComprehensivePlayerMetrics],
) -> str:
    """Generate print-friendly HTML as PDF fallback."""
    # Sort players by rating
    sorted_players = sorted(
        metrics.items(),
        key=lambda x: x[1].overall_rating(),
        reverse=True,
    )

    player_rows = ""
    for steam_id, pm in sorted_players:
        name = demo_data.player_names.get(steam_id, "Unknown")
        team = demo_data.player_teams.get(steam_id, "")
        kd = pm.engagement.total_kills / max(pm.engagement.total_deaths, 1)
        rating = pm.overall_rating()

        player_rows += f"""
        <tr>
            <td>{name}</td>
            <td>{team}</td>
            <td>{pm.engagement.total_kills}</td>
            <td>{pm.engagement.total_deaths}</td>
            <td>{kd:.2f}</td>
            <td>{pm.engagement.headshot_percentage:.1f}%</td>
            <td>{pm.engagement.damage_per_round:.1f}</td>
            <td class="rating">{rating:.2f}</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OpenSight Match Report</title>
    <style>
        @media print {{
            body {{ margin: 0; padding: 20px; }}
            .no-print {{ display: none; }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: white;
            color: #1a1a2e;
            padding: 40px;
            max-width: 1000px;
            margin: 0 auto;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #ffd700;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4a5568;
            margin-top: 30px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
            margin: 20px 0;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
        }}
        .info-item {{ text-align: center; }}
        .info-item .value {{ font-size: 1.5em; font-weight: bold; color: #1a1a2e; }}
        .info-item .label {{ color: #718096; font-size: 0.9em; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background: #1a1a2e;
            color: white;
            padding: 12px 8px;
            text-align: left;
        }}
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:nth-child(even) {{ background: #f7fafc; }}
        .rating {{ font-weight: bold; }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: #718096;
            font-size: 0.9em;
        }}
        .print-btn {{
            background: #ffd700;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 1em;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <button class="print-btn no-print" onclick="window.print()">Print / Save as PDF</button>

    <h1>OpenSight Match Analysis Report</h1>

    <h2>Match Information</h2>
    <div class="info-grid">
        <div class="info-item">
            <div class="value">{demo_data.map_name}</div>
            <div class="label">Map</div>
        </div>
        <div class="info-item">
            <div class="value">{demo_data.duration_seconds / 60:.1f}m</div>
            <div class="label">Duration</div>
        </div>
        <div class="info-item">
            <div class="value">{demo_data.tick_rate}</div>
            <div class="label">Tick Rate</div>
        </div>
        <div class="info-item">
            <div class="value">{len(demo_data.round_starts)}</div>
            <div class="label">Rounds</div>
        </div>
        <div class="info-item">
            <div class="value">{len(demo_data.player_names)}</div>
            <div class="label">Players</div>
        </div>
    </div>

    <h2>Player Statistics</h2>
    <table>
        <thead>
            <tr>
                <th>Player</th>
                <th>Team</th>
                <th>K</th>
                <th>D</th>
                <th>K/D</th>
                <th>HS%</th>
                <th>ADR</th>
                <th>Rating</th>
            </tr>
        </thead>
        <tbody>
            {player_rows}
        </tbody>
    </table>

    <div class="footer">
        Generated by OpenSight | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
        <span class="no-print">Use browser's Print function (Ctrl+P) to save as PDF</span>
    </div>
</body>
</html>
"""

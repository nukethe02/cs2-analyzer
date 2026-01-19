"""
Export Module for OpenSight Demo Analysis

Supports multiple output formats:
- JSON: Single file with all data
- CSV: Multiple files (players.csv, kills.csv, etc.)
- Excel: Formatted .xlsx with multiple sheets
- Dict: Python dictionary for programmatic use
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import logging

from opensight.analytics import MatchAnalysis, PlayerMatchStats

logger = logging.getLogger(__name__)


def player_to_dict(player: PlayerMatchStats) -> dict[str, Any]:
    """Convert PlayerMatchStats to a serializable dictionary."""
    return {
        "steam_id": player.steam_id,
        "name": player.name,
        "team": player.team,
        # Basic stats
        "kills": player.kills,
        "deaths": player.deaths,
        "assists": player.assists,
        "headshots": player.headshots,
        "total_damage": player.total_damage,
        "rounds_played": player.rounds_played,
        # Derived stats
        "kd_ratio": player.kd_ratio,
        "kd_diff": player.kd_diff,
        "adr": player.adr,
        "headshot_percentage": player.headshot_percentage,
        "kast_percentage": player.kast_percentage,
        "survival_rate": player.survival_rate,
        "kills_per_round": player.kills_per_round,
        "deaths_per_round": player.deaths_per_round,
        # HLTV 2.0 Rating
        "hltv_rating": player.hltv_rating,
        "impact_rating": player.impact_rating,
        # Opening duels
        "opening_duel_attempts": player.opening_duels.attempts,
        "opening_duel_wins": player.opening_duels.wins,
        "opening_duel_losses": player.opening_duels.losses,
        "opening_duel_win_rate": player.opening_duels.win_rate,
        # Trades
        "kills_traded": player.trades.kills_traded,
        "deaths_traded": player.trades.deaths_traded,
        # Clutches
        "clutch_situations": player.clutches.total_situations,
        "clutch_wins": player.clutches.total_wins,
        "clutch_1v1_attempts": player.clutches.situations_1v1,
        "clutch_1v1_wins": player.clutches.wins_1v1,
        "clutch_1v2_attempts": player.clutches.situations_1v2,
        "clutch_1v2_wins": player.clutches.wins_1v2,
        # Multi-kills
        "rounds_with_2k": player.multi_kills.rounds_with_2k,
        "rounds_with_3k": player.multi_kills.rounds_with_3k,
        "rounds_with_4k": player.multi_kills.rounds_with_4k,
        "rounds_with_5k": player.multi_kills.rounds_with_5k,
        # TTD
        "ttd_median_ms": player.ttd_median_ms,
        "ttd_mean_ms": player.ttd_mean_ms,
        "prefire_count": player.prefire_count,
        # CP
        "cp_median_error_deg": player.cp_median_error_deg,
        "cp_mean_error_deg": player.cp_mean_error_deg,
        # Weapon breakdown
        "weapon_kills": player.weapon_kills,
        # Utility
        "flash_assists": player.utility.flash_assists,
    }


def analysis_to_dict(analysis: MatchAnalysis) -> dict[str, Any]:
    """Convert MatchAnalysis to a serializable dictionary."""
    return {
        "match": {
            "map_name": analysis.map_name,
            "total_rounds": analysis.total_rounds,
            "team1_score": analysis.team1_score,
            "team2_score": analysis.team2_score,
        },
        "players": [player_to_dict(p) for p in analysis.get_leaderboard()],
        "mvp": {
            "steam_id": analysis.get_mvp().steam_id if analysis.get_mvp() else None,
            "name": analysis.get_mvp().name if analysis.get_mvp() else None,
            "rating": analysis.get_mvp().hltv_rating if analysis.get_mvp() else None,
        },
    }


def export_json(analysis: MatchAnalysis, output_path: str | Path) -> Path:
    """Export analysis to JSON file."""
    output_path = Path(output_path)
    data = analysis_to_dict(analysis)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported JSON to {output_path}")
    return output_path


def export_csv(analysis: MatchAnalysis, output_dir: str | Path) -> Path:
    """Export analysis to CSV files in a directory."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for CSV export")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Players CSV
    players_data = [player_to_dict(p) for p in analysis.get_leaderboard()]
    players_df = pd.DataFrame(players_data)

    # Flatten weapon_kills column
    players_df["top_weapon"] = players_df["weapon_kills"].apply(
        lambda x: max(x, key=x.get) if x else ""
    )
    players_df = players_df.drop(columns=["weapon_kills"])

    players_path = output_dir / "players.csv"
    players_df.to_csv(players_path, index=False)

    # Match info CSV
    match_data = {
        "map_name": [analysis.map_name],
        "total_rounds": [analysis.total_rounds],
        "team1_score": [analysis.team1_score],
        "team2_score": [analysis.team2_score],
        "mvp_name": [analysis.get_mvp().name if analysis.get_mvp() else ""],
        "mvp_rating": [analysis.get_mvp().hltv_rating if analysis.get_mvp() else 0],
    }
    match_df = pd.DataFrame(match_data)
    match_path = output_dir / "match.csv"
    match_df.to_csv(match_path, index=False)

    logger.info(f"Exported CSV files to {output_dir}")
    return output_dir


def export_excel(analysis: MatchAnalysis, output_path: str | Path) -> Path:
    """Export analysis to Excel workbook with multiple sheets."""
    try:
        import pandas as pd
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError:
        raise ImportError("pandas and openpyxl are required for Excel export")

    output_path = Path(output_path)

    # Create workbook
    wb = Workbook()

    # Summary sheet
    ws_summary = wb.active
    ws_summary.title = "Summary"

    # Add match info
    ws_summary["A1"] = "Match Summary"
    ws_summary["A1"].font = Font(bold=True, size=14)

    ws_summary["A3"] = "Map"
    ws_summary["B3"] = analysis.map_name
    ws_summary["A4"] = "Total Rounds"
    ws_summary["B4"] = analysis.total_rounds
    ws_summary["A5"] = "Score"
    ws_summary["B5"] = f"{analysis.team1_score} - {analysis.team2_score}"

    mvp = analysis.get_mvp()
    if mvp:
        ws_summary["A7"] = "MVP"
        ws_summary["B7"] = mvp.name
        ws_summary["A8"] = "MVP Rating"
        ws_summary["B8"] = mvp.hltv_rating

    # Players sheet
    ws_players = wb.create_sheet("Players")

    # Create DataFrame
    players_data = [player_to_dict(p) for p in analysis.get_leaderboard()]
    players_df = pd.DataFrame(players_data)

    # Select key columns for Excel
    key_columns = [
        "name", "team", "kills", "deaths", "assists", "adr",
        "kd_ratio", "headshot_percentage", "kast_percentage",
        "hltv_rating", "opening_duel_wins", "opening_duel_losses",
        "kills_traded", "clutch_wins"
    ]
    players_df = players_df[[c for c in key_columns if c in players_df.columns]]

    # Write to sheet
    for r_idx, row in enumerate(dataframe_to_rows(players_df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_players.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF")

    # Adjust column widths
    for column in ws_players.columns:
        max_length = max(len(str(cell.value or "")) for cell in column)
        ws_players.column_dimensions[column[0].column_letter].width = min(max_length + 2, 20)

    # Detailed stats sheet
    ws_detailed = wb.create_sheet("Detailed Stats")
    detailed_df = pd.DataFrame(players_data)

    for r_idx, row in enumerate(dataframe_to_rows(detailed_df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            if isinstance(value, dict):
                value = str(value)
            cell = ws_detailed.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:
                cell.font = Font(bold=True)

    # Save workbook
    wb.save(output_path)
    logger.info(f"Exported Excel to {output_path}")
    return output_path


def export_demo(
    analysis: MatchAnalysis,
    output_path: str | Path,
    format: Optional[str] = None
) -> Path:
    """
    Export analysis to specified format.

    Args:
        analysis: MatchAnalysis object
        output_path: Output file or directory path
        format: "json", "csv", "excel", or None (auto-detect from extension)

    Returns:
        Path to exported file/directory
    """
    output_path = Path(output_path)

    # Auto-detect format from extension
    if format is None:
        suffix = output_path.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix in [".xlsx", ".xls"]:
            format = "excel"
        elif suffix == ".csv" or output_path.is_dir() or not suffix:
            format = "csv"
        else:
            format = "json"  # Default

    if format == "json":
        return export_json(analysis, output_path)
    elif format == "csv":
        return export_csv(analysis, output_path)
    elif format == "excel":
        return export_excel(analysis, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")

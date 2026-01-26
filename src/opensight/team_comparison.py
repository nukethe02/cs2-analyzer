"""
Team Comparison Module

Compare team performance metrics between matches or teams.
Includes data visualization using SVG charts (no external dependencies - FREE).

Features:
- Compare team KDA ratios
- Compare average kill distances
- Identify top-performing teams
- Identify areas of improvement
- Generate visual comparisons (SVG/HTML charts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from opensight.team_performance_metrics import (
    TeamMetrics,
    calculate_team_metrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MetricComparison:
    """Comparison of a single metric between teams."""

    metric_name: str
    team_a_value: float
    team_b_value: float
    difference: float
    percentage_diff: float
    better_team: str  # "A", "B", or "tie"
    interpretation: str  # Human-readable interpretation

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "team_a": round(self.team_a_value, 2),
            "team_b": round(self.team_b_value, 2),
            "difference": round(self.difference, 2),
            "percentage_diff": round(self.percentage_diff, 1),
            "better_team": self.better_team,
            "interpretation": self.interpretation,
        }


@dataclass
class TeamStrength:
    """Identified strength or weakness of a team."""

    category: str  # "strength" or "weakness"
    metric: str
    value: float
    benchmark: float
    description: str
    improvement_tip: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "metric": self.metric,
            "value": round(self.value, 2),
            "benchmark": round(self.benchmark, 2),
            "description": self.description,
            "improvement_tip": self.improvement_tip,
        }


@dataclass
class TeamComparisonResult:
    """Complete comparison result between two teams."""

    team_a_name: str
    team_b_name: str
    overall_winner: str  # "A", "B", or "tie"
    win_margin: float  # How decisive the comparison is (0-100)

    metric_comparisons: list[MetricComparison] = field(default_factory=list)
    team_a_strengths: list[TeamStrength] = field(default_factory=list)
    team_b_strengths: list[TeamStrength] = field(default_factory=list)
    team_a_weaknesses: list[TeamStrength] = field(default_factory=list)
    team_b_weaknesses: list[TeamStrength] = field(default_factory=list)

    recommendations: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_a": self.team_a_name,
            "team_b": self.team_b_name,
            "overall_winner": self.overall_winner,
            "win_margin": round(self.win_margin, 1),
            "metric_comparisons": [m.to_dict() for m in self.metric_comparisons],
            "team_a_strengths": [s.to_dict() for s in self.team_a_strengths],
            "team_b_strengths": [s.to_dict() for s in self.team_b_strengths],
            "team_a_weaknesses": [w.to_dict() for w in self.team_a_weaknesses],
            "team_b_weaknesses": [w.to_dict() for w in self.team_b_weaknesses],
            "recommendations": self.recommendations,
        }


# =============================================================================
# Benchmark Values
# =============================================================================

# Industry benchmarks for team metrics (based on professional play)
BENCHMARKS = {
    "kda_ratio": {"excellent": 1.5, "good": 1.2, "average": 1.0, "poor": 0.8},
    "headshot_percentage": {"excellent": 55, "good": 45, "average": 35, "poor": 25},
    "avg_kill_distance": {"long_range": 1500, "medium": 1000, "close": 500},
    "round_win_rate": {"excellent": 60, "good": 53, "average": 47, "poor": 40},
    "trade_success_rate": {"excellent": 50, "good": 35, "average": 25, "poor": 15},
}


# =============================================================================
# Team Comparison Calculator
# =============================================================================


class TeamComparisonCalculator:
    """
    Compare team performance metrics and generate insights.

    All processing is done locally - no external API calls (FREE).
    """

    # Metrics where higher is better
    HIGHER_IS_BETTER = {
        "kda_ratio",
        "total_kills",
        "total_assists",
        "headshot_percentage",
        "round_win_rate",
        "trade_success_rate",
        "bombs_planted",
        "bombs_defused",
        "successful_trades",
        "eco_rounds_won",
    }

    # Metrics where lower is better
    LOWER_IS_BETTER = {
        "total_deaths",
        "deaths_by_enemy",
        "team_kills",
    }

    # Metric weights for overall comparison
    METRIC_WEIGHTS = {
        "kda_ratio": 2.0,
        "round_win_rate": 2.5,
        "headshot_percentage": 1.0,
        "avg_kill_distance": 0.5,
        "trade_success_rate": 1.5,
        "total_kills": 1.0,
    }

    def compare_teams(self, team_a: TeamMetrics, team_b: TeamMetrics) -> TeamComparisonResult:
        """
        Compare two teams and generate comprehensive analysis.

        Args:
            team_a: First team's metrics
            team_b: Second team's metrics

        Returns:
            TeamComparisonResult with detailed comparison
        """
        result = TeamComparisonResult(
            team_a_name=team_a.team_name,
            team_b_name=team_b.team_name,
            overall_winner="tie",
            win_margin=0.0,
        )

        # Compare key metrics
        metrics_to_compare = [
            ("KDA Ratio", "kda_ratio", team_a.kda_ratio, team_b.kda_ratio),
            ("Total Kills", "total_kills", team_a.total_kills, team_b.total_kills),
            ("Total Deaths", "total_deaths", team_a.total_deaths, team_b.total_deaths),
            (
                "Headshot %",
                "headshot_percentage",
                team_a.headshot_percentage,
                team_b.headshot_percentage,
            ),
            (
                "Avg Kill Distance",
                "avg_kill_distance",
                team_a.avg_kill_distance,
                team_b.avg_kill_distance,
            ),
            ("Round Win Rate", "round_win_rate", team_a.round_win_rate, team_b.round_win_rate),
            (
                "Trade Success %",
                "trade_success_rate",
                team_a.trade_success_rate,
                team_b.trade_success_rate,
            ),
            (
                "Successful Trades",
                "successful_trades",
                team_a.successful_trades,
                team_b.successful_trades,
            ),
        ]

        team_a_wins = 0
        team_b_wins = 0
        weighted_score_a = 0.0
        weighted_score_b = 0.0

        for name, key, val_a, val_b in metrics_to_compare:
            comparison = self._compare_metric(name, key, val_a, val_b)
            result.metric_comparisons.append(comparison)

            weight = self.METRIC_WEIGHTS.get(key, 1.0)

            if comparison.better_team == "A":
                team_a_wins += 1
                weighted_score_a += weight
            elif comparison.better_team == "B":
                team_b_wins += 1
                weighted_score_b += weight

        # Determine overall winner
        total_weight = weighted_score_a + weighted_score_b
        if total_weight > 0:
            if weighted_score_a > weighted_score_b:
                result.overall_winner = "A"
                result.win_margin = (weighted_score_a / total_weight) * 100
            elif weighted_score_b > weighted_score_a:
                result.overall_winner = "B"
                result.win_margin = (weighted_score_b / total_weight) * 100
            else:
                result.overall_winner = "tie"
                result.win_margin = 50.0

        # Identify strengths and weaknesses
        result.team_a_strengths, result.team_a_weaknesses = self._analyze_team_performance(team_a)
        result.team_b_strengths, result.team_b_weaknesses = self._analyze_team_performance(team_b)

        # Generate recommendations
        result.recommendations = {
            team_a.team_name: self._generate_recommendations(team_a, result.team_a_weaknesses),
            team_b.team_name: self._generate_recommendations(team_b, result.team_b_weaknesses),
        }

        return result

    def _compare_metric(self, name: str, key: str, val_a: float, val_b: float) -> MetricComparison:
        """Compare a single metric between teams."""
        diff = val_a - val_b
        pct_diff = 0.0
        if val_b != 0:
            pct_diff = ((val_a - val_b) / abs(val_b)) * 100
        elif val_a != 0:
            pct_diff = 100.0

        # Determine which team is better
        if key in self.HIGHER_IS_BETTER:
            if val_a > val_b:
                better = "A"
            elif val_b > val_a:
                better = "B"
            else:
                better = "tie"
        elif key in self.LOWER_IS_BETTER:
            if val_a < val_b:
                better = "A"
            elif val_b < val_a:
                better = "B"
            else:
                better = "tie"
        else:
            # Neutral metric - no preference
            better = "tie" if val_a == val_b else ("A" if val_a > val_b else "B")

        # Generate interpretation
        interpretation = self._interpret_comparison(name, key, val_a, val_b, better)

        return MetricComparison(
            metric_name=name,
            team_a_value=val_a,
            team_b_value=val_b,
            difference=diff,
            percentage_diff=pct_diff,
            better_team=better,
            interpretation=interpretation,
        )

    def _interpret_comparison(
        self, name: str, key: str, val_a: float, val_b: float, better: str
    ) -> str:
        """Generate human-readable interpretation of a comparison."""
        if better == "tie":
            return f"Both teams have similar {name}"

        pct = abs(val_a - val_b) / max(val_a, val_b, 1) * 100

        if pct < 5:
            magnitude = "slightly"
        elif pct < 15:
            magnitude = "moderately"
        elif pct < 30:
            magnitude = "significantly"
        else:
            magnitude = "dramatically"

        winner = "Team A" if better == "A" else "Team B"

        if key in self.HIGHER_IS_BETTER:
            return f"{winner} has {magnitude} better {name}"
        elif key in self.LOWER_IS_BETTER:
            return f"{winner} has {magnitude} fewer {name}"
        else:
            return f"{winner} has higher {name}"

    def _analyze_team_performance(
        self, team: TeamMetrics
    ) -> tuple[list[TeamStrength], list[TeamStrength]]:
        """Identify strengths and weaknesses of a team."""
        strengths: list[TeamStrength] = []
        weaknesses: list[TeamStrength] = []

        # Analyze KDA
        kda_bench = BENCHMARKS["kda_ratio"]
        if team.kda_ratio >= kda_bench["excellent"]:
            strengths.append(
                TeamStrength(
                    category="strength",
                    metric="KDA Ratio",
                    value=team.kda_ratio,
                    benchmark=kda_bench["excellent"],
                    description="Excellent kill/death efficiency",
                )
            )
        elif team.kda_ratio < kda_bench["poor"]:
            weaknesses.append(
                TeamStrength(
                    category="weakness",
                    metric="KDA Ratio",
                    value=team.kda_ratio,
                    benchmark=kda_bench["poor"],
                    description="Below average kill/death efficiency",
                    improvement_tip="Focus on trading deaths and avoiding unnecessary risks",
                )
            )

        # Analyze headshot percentage
        hs_bench = BENCHMARKS["headshot_percentage"]
        if team.headshot_percentage >= hs_bench["excellent"]:
            strengths.append(
                TeamStrength(
                    category="strength",
                    metric="Headshot %",
                    value=team.headshot_percentage,
                    benchmark=hs_bench["excellent"],
                    description="Elite-level aim precision",
                )
            )
        elif team.headshot_percentage < hs_bench["poor"]:
            weaknesses.append(
                TeamStrength(
                    category="weakness",
                    metric="Headshot %",
                    value=team.headshot_percentage,
                    benchmark=hs_bench["poor"],
                    description="Low headshot accuracy",
                    improvement_tip="Practice crosshair placement and aim training",
                )
            )

        # Analyze trade success
        trade_bench = BENCHMARKS["trade_success_rate"]
        if team.trade_success_rate >= trade_bench["excellent"]:
            strengths.append(
                TeamStrength(
                    category="strength",
                    metric="Trade Success",
                    value=team.trade_success_rate,
                    benchmark=trade_bench["excellent"],
                    description="Excellent team coordination on trades",
                )
            )
        elif team.trade_success_rate < trade_bench["poor"]:
            weaknesses.append(
                TeamStrength(
                    category="weakness",
                    metric="Trade Success",
                    value=team.trade_success_rate,
                    benchmark=trade_bench["poor"],
                    description="Poor trading discipline",
                    improvement_tip="Work on positioning to support teammates",
                )
            )

        # Analyze round win rate
        rwr_bench = BENCHMARKS["round_win_rate"]
        if team.round_win_rate >= rwr_bench["excellent"]:
            strengths.append(
                TeamStrength(
                    category="strength",
                    metric="Round Win Rate",
                    value=team.round_win_rate,
                    benchmark=rwr_bench["excellent"],
                    description="Dominant round performance",
                )
            )
        elif team.round_win_rate < rwr_bench["poor"]:
            weaknesses.append(
                TeamStrength(
                    category="weakness",
                    metric="Round Win Rate",
                    value=team.round_win_rate,
                    benchmark=rwr_bench["poor"],
                    description="Struggling to close out rounds",
                    improvement_tip="Review round execution and common failure points",
                )
            )

        return strengths, weaknesses

    def _generate_recommendations(
        self, team: TeamMetrics, weaknesses: list[TeamStrength]
    ) -> list[str]:
        """Generate improvement recommendations for a team."""
        recommendations: list[str] = []

        for weakness in weaknesses:
            if weakness.improvement_tip:
                recommendations.append(weakness.improvement_tip)

        # Add general recommendations based on metrics
        if team.total_flash_assists < 2:
            recommendations.append("Use more utility to support teammates")

        if team.deaths_by_enemy > team.total_kills:
            recommendations.append("Focus on survival and avoiding unfavorable fights")

        if len(recommendations) == 0:
            recommendations.append("Continue current strategies - performance is solid")

        return recommendations[:5]  # Limit to top 5


# =============================================================================
# Visualization Generator
# =============================================================================


class TeamVisualizationGenerator:
    """
    Generate SVG-based visualizations for team comparisons.

    Uses pure SVG - no external libraries required (FREE).
    """

    # Color palette
    COLORS = {
        "team_a": "#3498db",  # Blue
        "team_b": "#e74c3c",  # Red
        "background": "#f8f9fa",
        "grid": "#dee2e6",
        "text": "#212529",
        "positive": "#28a745",
        "negative": "#dc3545",
    }

    def generate_comparison_chart(
        self, comparison: TeamComparisonResult, width: int = 800, height: int = 400
    ) -> str:
        """
        Generate an SVG bar chart comparing team metrics.

        Args:
            comparison: TeamComparisonResult from compare_teams()
            width: Chart width in pixels
            height: Chart height in pixels

        Returns:
            SVG string
        """
        metrics = comparison.metric_comparisons[:6]  # Top 6 metrics
        bar_height = 30
        gap = 15
        margin = {"top": 40, "right": 120, "bottom": 40, "left": 150}

        chart_width = width - margin["left"] - margin["right"]

        # Find max value for scaling
        max_val = max(max(m.team_a_value, m.team_b_value) for m in metrics) or 1

        svg_parts = [
            f'<svg width="{width}" height="{height + 50}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="100%" height="100%" fill="{self.COLORS["background"]}"/>',
            # Title
            f'<text x="{width / 2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="{self.COLORS["text"]}">',
            f"Team Comparison: {comparison.team_a_name} vs {comparison.team_b_name}",
            "</text>",
            # Chart group
            f'<g transform="translate({margin["left"]}, {margin["top"]})">',
        ]

        # Draw bars for each metric
        for i, metric in enumerate(metrics):
            y = i * (bar_height * 2 + gap)

            # Metric label
            svg_parts.append(
                f'<text x="-10" y="{y + bar_height}" text-anchor="end" '
                f'font-size="12" fill="{self.COLORS["text"]}">{metric.metric_name}</text>'
            )

            # Team A bar
            width_a = (metric.team_a_value / max_val) * chart_width if max_val > 0 else 0
            svg_parts.append(
                f'<rect x="0" y="{y}" width="{width_a}" height="{bar_height}" '
                f'fill="{self.COLORS["team_a"]}" rx="3"/>'
            )
            svg_parts.append(
                f'<text x="{width_a + 5}" y="{y + bar_height / 2 + 4}" '
                f'font-size="11" fill="{self.COLORS["text"]}">{metric.team_a_value:.1f}</text>'
            )

            # Team B bar
            width_b = (metric.team_b_value / max_val) * chart_width if max_val > 0 else 0
            svg_parts.append(
                f'<rect x="0" y="{y + bar_height + 2}" width="{width_b}" height="{bar_height}" '
                f'fill="{self.COLORS["team_b"]}" rx="3"/>'
            )
            svg_parts.append(
                f'<text x="{width_b + 5}" y="{y + bar_height * 1.5 + 6}" '
                f'font-size="11" fill="{self.COLORS["text"]}">{metric.team_b_value:.1f}</text>'
            )

        svg_parts.append("</g>")

        # Legend
        legend_y = height - 20
        svg_parts.extend(
            [
                f'<rect x="{width / 2 - 100}" y="{legend_y}" width="15" height="15" fill="{self.COLORS["team_a"]}"/>',
                f'<text x="{width / 2 - 80}" y="{legend_y + 12}" font-size="12">{comparison.team_a_name}</text>',
                f'<rect x="{width / 2 + 20}" y="{legend_y}" width="15" height="15" fill="{self.COLORS["team_b"]}"/>',
                f'<text x="{width / 2 + 40}" y="{legend_y + 12}" font-size="12">{comparison.team_b_name}</text>',
            ]
        )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def generate_radar_chart(
        self, comparison: TeamComparisonResult, width: int = 500, height: int = 500
    ) -> str:
        """
        Generate an SVG radar/spider chart for team comparison.

        Args:
            comparison: TeamComparisonResult
            width: Chart width
            height: Chart height

        Returns:
            SVG string
        """
        import math

        metrics = comparison.metric_comparisons[:6]
        cx, cy = width / 2, height / 2
        radius = min(cx, cy) - 80

        # Normalize values to 0-100 scale
        max_vals = {m.metric_name: max(m.team_a_value, m.team_b_value, 1) for m in metrics}

        def get_point(index: int, value: float, max_val: float) -> tuple[float, float]:
            angle = (2 * math.pi * index / len(metrics)) - math.pi / 2
            r = (value / max_val) * radius if max_val > 0 else 0
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            return x, y

        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="100%" height="100%" fill="{self.COLORS["background"]}"/>',
            # Title
            f'<text x="{cx}" y="25" text-anchor="middle" font-size="14" font-weight="bold">Performance Radar</text>',
        ]

        # Draw grid circles
        for i in range(1, 5):
            r = radius * i / 4
            svg_parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
                f'stroke="{self.COLORS["grid"]}" stroke-width="1"/>'
            )

        # Draw axis lines and labels
        for i, metric in enumerate(metrics):
            angle = (2 * math.pi * i / len(metrics)) - math.pi / 2
            x_end = cx + radius * math.cos(angle)
            y_end = cy + radius * math.sin(angle)
            svg_parts.append(
                f'<line x1="{cx}" y1="{cy}" x2="{x_end}" y2="{y_end}" '
                f'stroke="{self.COLORS["grid"]}" stroke-width="1"/>'
            )
            # Label
            label_x = cx + (radius + 30) * math.cos(angle)
            label_y = cy + (radius + 30) * math.sin(angle)
            anchor = "middle"
            if angle > -0.1 and angle < math.pi / 2:
                anchor = "start"
            elif angle > math.pi / 2:
                anchor = "end"
            svg_parts.append(
                f'<text x="{label_x}" y="{label_y}" text-anchor="{anchor}" '
                f'font-size="10" fill="{self.COLORS["text"]}">{metric.metric_name}</text>'
            )

        # Draw Team A polygon
        points_a = []
        for i, metric in enumerate(metrics):
            x, y = get_point(i, metric.team_a_value, max_vals[metric.metric_name])
            points_a.append(f"{x},{y}")
        svg_parts.append(
            f'<polygon points="{" ".join(points_a)}" fill="{self.COLORS["team_a"]}" '
            f'fill-opacity="0.3" stroke="{self.COLORS["team_a"]}" stroke-width="2"/>'
        )

        # Draw Team B polygon
        points_b = []
        for i, metric in enumerate(metrics):
            x, y = get_point(i, metric.team_b_value, max_vals[metric.metric_name])
            points_b.append(f"{x},{y}")
        svg_parts.append(
            f'<polygon points="{" ".join(points_b)}" fill="{self.COLORS["team_b"]}" '
            f'fill-opacity="0.3" stroke="{self.COLORS["team_b"]}" stroke-width="2"/>'
        )

        # Legend
        svg_parts.extend(
            [
                f'<rect x="20" y="{height - 40}" width="15" height="15" fill="{self.COLORS["team_a"]}"/>',
                f'<text x="40" y="{height - 28}" font-size="12">{comparison.team_a_name}</text>',
                f'<rect x="20" y="{height - 20}" width="15" height="15" fill="{self.COLORS["team_b"]}"/>',
                f'<text x="40" y="{height - 8}" font-size="12">{comparison.team_b_name}</text>',
            ]
        )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def generate_html_report(self, comparison: TeamComparisonResult) -> str:
        """
        Generate a complete HTML report with charts and analysis.

        Args:
            comparison: TeamComparisonResult

        Returns:
            HTML string
        """
        bar_chart = self.generate_comparison_chart(comparison)
        radar_chart = self.generate_radar_chart(comparison)

        winner_text = (
            f"{comparison.team_a_name}"
            if comparison.overall_winner == "A"
            else (f"{comparison.team_b_name}" if comparison.overall_winner == "B" else "Tie")
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Team Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .charts {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .chart-container {{
            flex: 1;
            min-width: 400px;
        }}
        h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metric-table th, .metric-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .metric-table th {{
            background: #f8f9fa;
        }}
        .better {{
            color: #28a745;
            font-weight: bold;
        }}
        .worse {{
            color: #dc3545;
        }}
        .strength {{
            background: #d4edda;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .weakness {{
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .recommendation {{
            background: #cce5ff;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .winner-badge {{
            display: inline-block;
            padding: 5px 15px;
            background: #28a745;
            color: white;
            border-radius: 20px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Team Performance Comparison</h1>
        <p>{comparison.team_a_name} vs {comparison.team_b_name}</p>
        <div class="winner-badge">Winner: {winner_text} ({comparison.win_margin:.1f}%)</div>
    </div>

    <div class="card">
        <h2>Visual Comparison</h2>
        <div class="charts">
            <div class="chart-container">{bar_chart}</div>
            <div class="chart-container">{radar_chart}</div>
        </div>
    </div>

    <div class="card">
        <h2>Detailed Metrics</h2>
        <table class="metric-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>{comparison.team_a_name}</th>
                    <th>{comparison.team_b_name}</th>
                    <th>Difference</th>
                    <th>Analysis</th>
                </tr>
            </thead>
            <tbody>
"""

        for m in comparison.metric_comparisons:
            a_class = (
                "better" if m.better_team == "A" else ("worse" if m.better_team == "B" else "")
            )
            b_class = (
                "better" if m.better_team == "B" else ("worse" if m.better_team == "A" else "")
            )
            html += f"""
                <tr>
                    <td>{m.metric_name}</td>
                    <td class="{a_class}">{m.team_a_value:.2f}</td>
                    <td class="{b_class}">{m.team_b_value:.2f}</td>
                    <td>{m.difference:+.2f} ({m.percentage_diff:+.1f}%)</td>
                    <td>{m.interpretation}</td>
                </tr>
"""

        html += f"""
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Team Analysis</h2>
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1;">
                <h3>{comparison.team_a_name}</h3>
                <h4>Strengths</h4>
"""

        for s in comparison.team_a_strengths:
            html += f'<div class="strength">{s.metric}: {s.description}</div>'

        html += "<h4>Areas for Improvement</h4>"
        for w in comparison.team_a_weaknesses:
            tip = f" - {w.improvement_tip}" if w.improvement_tip else ""
            html += f'<div class="weakness">{w.metric}: {w.description}{tip}</div>'

        html += f"""
            </div>
            <div style="flex: 1;">
                <h3>{comparison.team_b_name}</h3>
                <h4>Strengths</h4>
"""

        for s in comparison.team_b_strengths:
            html += f'<div class="strength">{s.metric}: {s.description}</div>'

        html += "<h4>Areas for Improvement</h4>"
        for w in comparison.team_b_weaknesses:
            tip = f" - {w.improvement_tip}" if w.improvement_tip else ""
            html += f'<div class="weakness">{w.metric}: {w.description}{tip}</div>'

        html += """
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Recommendations</h2>
"""

        for team_name, recs in comparison.recommendations.items():
            html += f"<h3>{team_name}</h3>"
            for rec in recs:
                html += f'<div class="recommendation">{rec}</div>'

        html += """
    </div>

    <footer style="text-align: center; color: #666; padding: 20px;">
        Generated by OpenSight CS2 Analyzer - FREE local analytics
    </footer>
</body>
</html>
"""
        return html


# =============================================================================
# Convenience Functions
# =============================================================================


def compare_teams_from_match(match_data: Any) -> TeamComparisonResult:
    """
    Compare CT and T teams from a single match.

    Args:
        match_data: MatchData/DemoData from parser

    Returns:
        TeamComparisonResult
    """
    analysis = calculate_team_metrics(match_data)
    if not analysis.ct_metrics or not analysis.t_metrics:
        raise ValueError("Could not extract team metrics from match data")

    calculator = TeamComparisonCalculator()
    return calculator.compare_teams(analysis.ct_metrics, analysis.t_metrics)


def compare_teams(team_a: TeamMetrics, team_b: TeamMetrics) -> TeamComparisonResult:
    """
    Compare two team metrics objects.

    Args:
        team_a: First team's metrics
        team_b: Second team's metrics

    Returns:
        TeamComparisonResult
    """
    calculator = TeamComparisonCalculator()
    return calculator.compare_teams(team_a, team_b)


def generate_comparison_html(comparison: TeamComparisonResult) -> str:
    """
    Generate HTML report for a team comparison.

    Args:
        comparison: TeamComparisonResult

    Returns:
        HTML string
    """
    generator = TeamVisualizationGenerator()
    return generator.generate_html_report(comparison)


def generate_comparison_charts(
    comparison: TeamComparisonResult,
) -> dict[str, str]:
    """
    Generate SVG charts for a team comparison.

    Args:
        comparison: TeamComparisonResult

    Returns:
        Dictionary with 'bar_chart' and 'radar_chart' SVG strings
    """
    generator = TeamVisualizationGenerator()
    return {
        "bar_chart": generator.generate_comparison_chart(comparison),
        "radar_chart": generator.generate_radar_chart(comparison),
    }

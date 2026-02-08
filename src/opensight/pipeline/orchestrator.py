"""
Demo Analysis Orchestrator - Main pipeline for processing demos.

Extracted from cache.py to separate orchestration logic from caching.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from opensight.core.utils import build_round_boundaries, infer_round_from_tick

logger = logging.getLogger(__name__)


class DemoOrchestrator:
    """
    Orchestrates the complete demo analysis pipeline.

    Handles:
    - Demo parsing
    - Analysis execution
    - Result aggregation and serialization
    - Tactical insights generation
    - Automatic cache integration (hash file → check cache → short-circuit or analyze)
    """

    def __init__(self, *, use_cache: bool = True):
        self._use_cache = use_cache
        self._cache = None  # Lazy-init to avoid circular import

    def _get_cache(self):
        """Lazy-import and create DemoCache to avoid circular import."""
        if self._cache is None:
            from opensight.infra.cache import DemoCache

            self._cache = DemoCache()
        return self._cache

    def analyze(self, demo_path: Path, *, force: bool = False) -> dict:
        """
        Execute complete analysis pipeline for a demo file.

        Checks the file-hash cache first; returns cached result in ~2s
        instead of re-parsing (~80s) when the demo has been seen before.

        Args:
            demo_path: Path to demo file (.dem or .dem.gz)
            force: Skip cache and re-analyze even if cached

        Returns:
            Comprehensive analysis result dict including tactical data
        """
        cache_key = None

        # --- Cache short-circuit (before ANY expensive work) ---
        if self._use_cache and not force and demo_path.exists():
            try:
                cache = self._get_cache()
                cache_key = cache.get_cache_key(demo_path)  # ~1.5s SHA256
                cached = cache.get(demo_path, cache_key=cache_key)
                if cached is not None:
                    logger.info(f"Cache hit for {demo_path.name}, skipping analysis")
                    return cached
            except Exception as e:
                logger.warning(f"Cache lookup failed, proceeding with analysis: {e}")

        # Execute analysis pipeline
        logger.info(f"Analyzing {demo_path.name}")
        from opensight.analysis.analytics import DemoAnalyzer
        from opensight.core.parser import DemoParser

        parser = DemoParser(demo_path)
        demo_data = parser.parse()

        analyzer = DemoAnalyzer(demo_data)
        analysis = analyzer.analyze()

        # Calculate RWS using direct method (more reliable)
        rws_data = self._calculate_rws_direct(demo_data)

        # Calculate multi-kills (2K, 3K, 4K, 5K)
        multikills = self._calculate_multikills(demo_data)

        # Build timeline graph data
        timeline_graph = self._build_timeline_graph_data(demo_data)

        # Build comprehensive player data
        players = {}
        for sid, p in analysis.players.items():
            mk = multikills.get(sid, {"2k": 0, "3k": 0, "4k": 0, "5k": 0})
            players[str(sid)] = {
                "steam_id": str(sid),
                "name": p.name,
                "team": p.team,
                # Top-level convenience fields for frontend flat-access fallbacks
                "rounds_played": p.rounds_played,
                "total_damage": p.total_damage,
                "weapon_kills": p.weapon_kills or {},
                "stats": {
                    "kills": p.kills,
                    "deaths": p.deaths,
                    "assists": p.assists,
                    "rounds_played": p.rounds_played,
                    # NOTE: duplicates analytics.py computation — p.adr is already rounded
                    # in PlayerMatchStats.adr property. Consider using p.adr directly.
                    "adr": round(p.adr, 1) if p.adr is not None else 0,
                    # NOTE: duplicates analytics.py computation — p.headshot_percentage is
                    # already rounded in PlayerMatchStats.headshot_percentage property.
                    "headshot_pct": (
                        round(p.headshot_percentage, 1) if p.headshot_percentage is not None else 0
                    ),
                    # NOTE: duplicates analytics.py computation — consider consolidating
                    # with PlayerMatchStats.kd_ratio property (models.py)
                    "kd_ratio": round(p.kills / max(1, p.deaths), 2),
                    "total_damage": p.total_damage,
                    "2k": mk["2k"],
                    "3k": mk["3k"],
                    "4k": mk["4k"],
                    "5k": mk["5k"],
                },
                "rating": {
                    "hltv_rating": round(p.hltv_rating, 2) if p.hltv_rating is not None else 0,
                    "kast_percentage": (
                        round(p.kast_percentage, 1) if p.kast_percentage is not None else 0
                    ),
                    # aim_rating: 0 = missing data, actual values 1-100
                    "aim_rating": round(p.aim_rating, 1),
                    "utility_rating": (
                        round(p.utility_rating, 1) if p.utility_rating is not None else 0
                    ),
                    "impact_rating": (
                        round(p.impact_rating, 2) if p.impact_rating is not None else 0
                    ),
                },
                "advanced": {
                    # TTD - Time to Damage (engagement duration)
                    "ttd_median_ms": (
                        round(p.ttd_median_ms, 1) if p.ttd_median_ms is not None else 0
                    ),
                    "ttd_mean_ms": (round(p.ttd_mean_ms, 1) if p.ttd_mean_ms is not None else 0),
                    "ttd_95th_ms": (
                        round(float(np.percentile(p.engagement_duration_values, 95)), 1)
                        if p.engagement_duration_values
                        else 0
                    ),
                    # CP - Crosshair Placement
                    "cp_median_error_deg": (
                        round(p.cp_median_error_deg, 1) if p.cp_median_error_deg is not None else 0
                    ),
                    "cp_mean_error_deg": (
                        round(p.cp_mean_error_deg, 1) if p.cp_mean_error_deg is not None else 0
                    ),
                    # Other advanced stats — read from actual dataclass attributes
                    "prefire_kills": p.prefire_count,
                    "opening_kills": p.opening_duels.wins if p.opening_duels else 0,
                    "opening_deaths": p.opening_duels.losses if p.opening_duels else 0,
                },
                "utility": {
                    # Raw counts
                    "flashbangs_thrown": (p.utility.flashbangs_thrown if p.utility else 0),
                    "smokes_thrown": p.utility.smokes_thrown if p.utility else 0,
                    "he_thrown": p.utility.he_thrown if p.utility else 0,
                    "molotovs_thrown": p.utility.molotovs_thrown if p.utility else 0,
                    "flash_assists": p.utility.flash_assists if p.utility else 0,
                    "enemies_flashed": p.utility.enemies_flashed if p.utility else 0,
                    "teammates_flashed": (p.utility.teammates_flashed if p.utility else 0),
                    "he_damage": p.utility.he_damage if p.utility else 0,
                    "molotov_damage": p.utility.molotov_damage if p.utility else 0,
                    # Computed per-round and average metrics (needed by frontend)
                    "enemies_flashed_per_round": (
                        p.utility.enemies_flashed_per_round if p.utility else 0
                    ),
                    "friends_flashed_per_round": (
                        p.utility.friends_flashed_per_round if p.utility else 0
                    ),
                    "avg_blind_time": p.utility.avg_blind_time if p.utility else 0,
                    "avg_he_damage": p.utility.avg_he_damage if p.utility else 0,
                    "flash_effectiveness_pct": (
                        p.utility.flash_effectiveness_pct if p.utility else 0
                    ),
                    "flash_assist_pct": (p.utility.flash_assist_pct if p.utility else 0),
                    "he_team_damage": p.utility.he_team_damage if p.utility else 0,
                    "unused_utility_value": p.utility.unused_utility_value if p.utility else 0,
                    "utility_quality_rating": (
                        round(p.utility.utility_quality_rating, 1) if p.utility else 0
                    ),
                    "utility_quantity_rating": (
                        round(p.utility.utility_quantity_rating, 1) if p.utility else 0
                    ),
                    "effective_flashes": p.utility.effective_flashes if p.utility else 0,
                    "total_blind_time": p.utility.total_blind_time if p.utility else 0,
                    "times_blinded": p.utility.times_blinded if p.utility else 0,
                    "total_time_blinded": p.utility.total_time_blinded if p.utility else 0,
                    "avg_time_blinded": p.utility.avg_time_blinded if p.utility else 0,
                },
                "aim_stats": {
                    # Accuracy metrics (from weapon_fire events)
                    "shots_fired": p.shots_fired,
                    "shots_hit": p.shots_hit,
                    "accuracy_all": round(p.accuracy, 1),
                    "headshot_hits": p.headshot_hits,
                    "head_accuracy": round(p.head_hit_rate, 1),
                    # Spray accuracy (hits after 4th bullet in burst)
                    "spray_shots_fired": p.spray_shots_fired,
                    "spray_shots_hit": p.spray_shots_hit,
                    "spray_accuracy": round(p.spray_accuracy, 1),
                    # Counter-strafing (shots while near-stationary)
                    "shots_stationary": p.shots_stationary,
                    "shots_with_velocity": p.shots_with_velocity,
                    "counter_strafe_pct": round(p.counter_strafe_pct, 1),
                    # TTD and CP (duplicated here for frontend convenience)
                    "time_to_damage_ms": (
                        round(p.ttd_median_ms, 1) if p.ttd_median_ms is not None else 0
                    ),
                    "crosshair_placement_deg": (
                        round(p.cp_median_error_deg, 1) if p.cp_median_error_deg is not None else 0
                    ),
                },
                "duels": {
                    "trade_kills": p.trades.kills_traded if p.trades else 0,
                    "traded_deaths": p.trades.deaths_traded if p.trades else 0,
                    "trade_kill_opportunities": (
                        p.trades.trade_kill_opportunities if p.trades else 0
                    ),
                    "untraded_deaths": p.untraded_deaths,
                    "clutch_wins": p.clutches.total_wins if p.clutches else 0,
                    "clutch_attempts": p.clutches.total_situations if p.clutches else 0,
                    "opening_kills": p.opening_duels.wins if p.opening_duels else 0,
                    "opening_deaths": p.opening_duels.losses if p.opening_duels else 0,
                    "opening_wins": p.opening_duels.wins if p.opening_duels else 0,
                    "opening_losses": p.opening_duels.losses if p.opening_duels else 0,
                    "opening_win_rate": (p.opening_duels.win_rate if p.opening_duels else 0),
                },
                "spray_transfers": {
                    "double_sprays": p.spray_transfers.double_sprays if p.spray_transfers else 0,
                    "triple_sprays": p.spray_transfers.triple_sprays if p.spray_transfers else 0,
                    "quad_sprays": p.spray_transfers.quad_sprays if p.spray_transfers else 0,
                    "ace_sprays": p.spray_transfers.ace_sprays if p.spray_transfers else 0,
                    "total_sprays": p.spray_transfers.total_sprays if p.spray_transfers else 0,
                    "total_spray_kills": (
                        p.spray_transfers.total_spray_kills if p.spray_transfers else 0
                    ),
                    "avg_spray_time_ms": (
                        p.spray_transfers.avg_spray_time_ms if p.spray_transfers else 0
                    ),
                    "avg_kills_per_spray": (
                        round(
                            p.spray_transfers.total_spray_kills
                            / max(1, p.spray_transfers.total_sprays),
                            1,
                        )
                        if p.spray_transfers and p.spray_transfers.total_sprays > 0
                        else 0
                    ),
                },
                "entry": self._get_entry_stats(p),
                "trades": self._get_trade_stats(p),
                "clutches": self._get_clutch_stats(p),
                "rws": rws_data.get(
                    sid,
                    {
                        "avg_rws": 0,
                        "total_rws": 0,
                        "rounds_won": 0,
                        "rounds_played": 0,
                        "damage_per_round": 0,
                        "objective_completions": 0,
                    },
                ),
                "economy": {
                    "avg_equipment_value": round(p.avg_equipment_value, 0),
                    "eco_rounds": p.eco_rounds,
                    "force_rounds": p.force_rounds,
                    "full_buy_rounds": p.full_buy_rounds,
                    "damage_per_dollar": round(p.damage_per_dollar, 4),
                    "kills_per_dollar": round(p.kills_per_dollar, 6),
                },
                "discipline": {
                    "discipline_rating": round(p.discipline_rating, 1),
                    "greedy_repeeks": p.greedy_repeeks,
                },
            }

        # Build round timeline
        round_timeline = self._build_round_timeline(demo_data, analysis)
        logger.debug(f"Built round timeline with {len(round_timeline)} rounds")

        # Build kill matrix
        kill_matrix = self._build_kill_matrix(demo_data)

        # Build heatmap data
        heatmap_data = self._build_heatmap_data(demo_data)

        # Generate coaching insights
        coaching = self._generate_coaching_insights(demo_data, analysis, players)

        # Generate AI-powered match summaries for each player
        self._generate_ai_summaries(players, analysis)

        # Get tactical summary
        tactical = self._get_tactical_summary(demo_data, analysis)

        # Analyze team synergy (NOT in Leetify - our differentiator)
        try:
            from opensight.domains.synergy import analyze_synergy

            synergy = analyze_synergy(demo_data)
            logger.debug(
                f"Synergy analysis: {len(synergy.get('pair_synergies', []))} pairs analyzed"
            )
        except Exception as e:
            logger.debug(f"Synergy analysis not available: {e}")
            synergy = {
                "pair_synergies": [],
                "best_duo": None,
                "trade_network": {},
                "flash_network": {},
            }

        # Find MVP
        mvp = None
        if players:
            mvp_data = max(players.values(), key=lambda x: x["rating"]["hltv_rating"])
            mvp = {
                "name": mvp_data["name"],
                "rating": mvp_data["rating"]["hltv_rating"],
            }

        # Sort players by HLTV rating (descending) before returning
        players_sorted = sorted(
            players.values(), key=lambda p: p["rating"]["hltv_rating"], reverse=True
        )

        # Build map metadata for frontend coordinate transformation
        from opensight.map_data import RADAR_IMAGE_SIZE, get_map_metadata

        map_name = analysis.map_name
        map_meta = get_map_metadata(map_name)
        map_metadata = {
            "map_name": map_name,
            "pos_x": map_meta["pos_x"] if map_meta else None,
            "pos_y": map_meta["pos_y"] if map_meta else None,
            "scale": map_meta["scale"] if map_meta else None,
            "radar_image_size": RADAR_IMAGE_SIZE,
            "radar_image_url": f"/static/maps/{map_name}.png" if map_meta else None,
        }

        # Convert to dict for caching
        result = {
            "demo_path": str(demo_path),
            "demo_info": {
                "map": map_name,
                "rounds": analysis.total_rounds,
                "duration_minutes": getattr(analysis, "duration_minutes", 30),
                "score": f"{analysis.team1_score} - {analysis.team2_score}",
                "score_ct": analysis.team1_score,  # Numeric CT score for AI modules
                "score_t": analysis.team2_score,  # Numeric T score for AI modules
                "total_kills": sum(p["stats"]["kills"] for p in players.values()),
                "team1_name": getattr(analysis, "team1_name", "Counter-Terrorists"),
                "team2_name": getattr(analysis, "team2_name", "Terrorists"),
            },
            "map_metadata": map_metadata,
            "players": {p["steam_id"]: p for p in players_sorted},
            "mvp": mvp,
            "round_timeline": round_timeline,
            "kill_matrix": kill_matrix,
            "heatmap_data": heatmap_data,
            "coaching": coaching,
            "tactical": tactical,
            "synergy": synergy,
            "timeline_graph": timeline_graph,
            "analyzed_at": datetime.now().isoformat(),
        }

        # --- Store in cache for next time ---
        if self._use_cache and demo_path.exists():
            try:
                cache = self._get_cache()
                if cache_key is None:
                    cache_key = cache.get_cache_key(demo_path)
                cache.put(demo_path, result, cache_key=cache_key)
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")

        return result

    def _get_entry_stats(self, player) -> dict:
        """Get comprehensive entry/opening duel stats like FACEIT."""
        opening = getattr(player, "opening_duels", None)
        if opening:
            attempts = opening.attempts
            wins = opening.wins
            losses = opening.losses
            rounds = getattr(player, "rounds_played", 0) or 1
            return {
                "entry_attempts": attempts,
                "entry_kills": wins,
                "entry_deaths": losses,
                "entry_diff": wins - losses,
                "entry_attempts_pct": (round(attempts / rounds * 100, 0) if rounds > 0 else 0),
                "entry_success_pct": (round(wins / attempts * 100, 0) if attempts > 0 else 0),
            }
        return {
            "entry_attempts": 0,
            "entry_kills": 0,
            "entry_deaths": 0,
            "entry_diff": 0,
            "entry_attempts_pct": 0,
            "entry_success_pct": 0,
        }

    def _get_trade_stats(self, player) -> dict:
        """Get comprehensive Leetify-style trade stats."""
        trades = getattr(player, "trades", None)
        # Compute untraded_deaths: deaths that were NOT avenged by teammates
        # Use None sentinel to distinguish "not set" from "explicitly 0"
        untraded = getattr(player, "untraded_deaths", None)
        if untraded is None and trades:
            # Fallback: compute from deaths - deaths_traded
            deaths = getattr(player, "deaths", 0)
            untraded = max(0, deaths - trades.deaths_traded)
        if untraded is None:
            untraded = 0
        if trades:
            return {
                # Trade kill stats (you trading for teammates)
                "trade_kill_opportunities": trades.trade_kill_opportunities,
                "trade_kill_attempts": trades.trade_kill_attempts,
                "trade_kill_attempts_pct": round(trades.trade_kill_attempts_pct, 1),
                "trade_kill_success": trades.trade_kill_success,
                "trade_kill_success_pct": round(trades.trade_kill_success_pct, 1),
                # Traded death stats (teammates trading for you)
                "traded_death_opportunities": trades.traded_death_opportunities,
                "traded_death_attempts": trades.traded_death_attempts,
                "traded_death_attempts_pct": round(trades.traded_death_attempts_pct, 1),
                "traded_death_success": trades.traded_death_success,
                "traded_death_success_pct": round(trades.traded_death_success_pct, 1),
                # Untraded deaths (needed by lurker persona detection)
                "untraded_deaths": untraded,
                # Legacy aliases
                "trade_kills": trades.kills_traded,
                "deaths_traded": trades.deaths_traded,
                "traded_entry_kills": trades.traded_entry_kills,
                "traded_entry_deaths": trades.traded_entry_deaths,
            }
        return {
            "trade_kill_opportunities": 0,
            "trade_kill_attempts": 0,
            "trade_kill_attempts_pct": 0,
            "trade_kill_success": 0,
            "trade_kill_success_pct": 0,
            "traded_death_opportunities": 0,
            "traded_death_attempts": 0,
            "traded_death_attempts_pct": 0,
            "traded_death_success": 0,
            "traded_death_success_pct": 0,
            "untraded_deaths": 0,
            "trade_kills": 0,
            "deaths_traded": 0,
            "traded_entry_kills": 0,
            "traded_entry_deaths": 0,
        }

    def _get_clutch_stats(self, player) -> dict:
        """Get comprehensive clutch stats like FACEIT."""
        clutches = getattr(player, "clutches", None)
        if clutches:
            total = clutches.total_situations
            wins = clutches.total_wins
            return {
                "clutch_wins": wins,
                "clutch_losses": total - wins,
                "clutch_success_pct": round(wins / total * 100, 0) if total > 0 else 0,
                "total_situations": total,
                "v1_wins": clutches.v1_wins,
                "v2_wins": clutches.v2_wins,
                "v3_wins": clutches.v3_wins,
                "v4_wins": clutches.v4_wins,
                "v5_wins": clutches.v5_wins,
            }
        return {
            "clutch_wins": 0,
            "clutch_losses": 0,
            "clutch_success_pct": 0,
            "total_situations": 0,
            "v1_wins": 0,
            "v2_wins": 0,
            "v3_wins": 0,
            "v4_wins": 0,
            "v5_wins": 0,
        }

    def _get_rws_for_player(self, steam_id: int, rws_data: dict) -> dict:
        """Get RWS data for a specific player."""
        if steam_id in rws_data:
            rws = rws_data[steam_id]
            return {
                "avg_rws": round(rws.avg_rws, 2),
                "total_rws": round(rws.total_rws, 1),
                "rounds_won": rws.rounds_won,
                "rounds_played": rws.rounds_played,
                "damage_per_round": round(rws.damage_per_round, 1),
                "objective_completions": rws.objective_completions,
            }
        return {
            "avg_rws": 0.0,
            "total_rws": 0.0,
            "rounds_won": 0,
            "rounds_played": 0,
            "damage_per_round": 0.0,
            "objective_completions": 0,
        }

    def _build_round_timeline(self, demo_data, analysis) -> list[dict]:
        """Build round-by-round timeline data with detailed events and win probability."""
        # Import win probability calculation
        try:
            from opensight.analysis.analytics import calculate_win_probability
        except ImportError:
            calculate_win_probability = None

        # Import Economy IQ analysis and Prediction Engine
        # NOTE(perf): EconomyAnalyzer is also instantiated in DemoAnalyzer._integrate_economy()
        # (analytics.py). Both instances are needed: analytics merges economy data into player
        # stats, while this instance enriches the round timeline. The analyzer is lightweight
        # (stateless, reads from demo_data) so the duplicate instantiation cost is negligible
        # compared to the actual analysis work.
        economy_by_round: dict[int, dict[int, Any]] = {}
        economy_predictions: dict[int, dict[str, dict]] = {}  # round -> {team: prediction}
        try:
            from opensight.domains.economy import EconomyAnalyzer, EconomyPredictor

            economy_analyzer = EconomyAnalyzer(demo_data)
            economy_stats = economy_analyzer.analyze()
            # Build lookup: round_num -> {team: TeamRoundEconomy}
            for team, team_rounds in economy_stats.team_economies.items():
                for tr in team_rounds:
                    if tr.round_num not in economy_by_round:
                        economy_by_round[tr.round_num] = {}
                    economy_by_round[tr.round_num][team] = tr
            logger.debug(f"Economy IQ loaded for {len(economy_by_round)} rounds")

            # Generate economy predictions for each team
            predictor = EconomyPredictor()
            ct_history: list = []
            t_history: list = []

            # Sort rounds by number for proper prediction ordering
            sorted_rounds = sorted(economy_by_round.keys())
            for round_num in sorted_rounds:
                round_econ = economy_by_round[round_num]
                economy_predictions[round_num] = {}

                # Predict for CT (team 3)
                if 3 in round_econ:
                    ct_pred = predictor.predict_next_round(round_num, "CT", ct_history)
                    economy_predictions[round_num]["ct"] = {
                        "predicted_buy": ct_pred.predicted_buy,
                        "confidence": round(ct_pred.confidence, 2),
                        "reasoning": ct_pred.reasoning,
                        "estimated_money": ct_pred.estimated_team_money,
                        "estimated_loadout": ct_pred.estimated_avg_loadout,
                    }
                    ct_history.append(round_econ[3])

                # Predict for T (team 2)
                if 2 in round_econ:
                    t_pred = predictor.predict_next_round(round_num, "T", t_history)
                    economy_predictions[round_num]["t"] = {
                        "predicted_buy": t_pred.predicted_buy,
                        "confidence": round(t_pred.confidence, 2),
                        "reasoning": t_pred.reasoning,
                        "estimated_money": t_pred.estimated_team_money,
                        "estimated_loadout": t_pred.estimated_avg_loadout,
                    }
                    t_history.append(round_econ[2])

                # Update predictor with round result
                winner = "CT" if round_econ.get(3, round_econ.get(2)).round_won else "T"
                if 2 in round_econ and round_econ[2].round_won:
                    winner = "T"
                predictor.record_round_result(winner)

                # Reset at halftime (round 12 -> 13 is halftime in MR12)
                if round_num == 12:
                    predictor.reset_half()

            logger.debug(f"Generated {len(economy_predictions)} round predictions")
        except Exception as e:
            logger.debug(f"Economy analysis not available: {e}")

        # Get clutch data from combat analysis for timeline enrichment
        # NOTE(perf): CombatAnalyzer is also instantiated in DemoAnalyzer._integrate_combat()
        # (analytics.py). Same pattern as EconomyAnalyzer above — both instances serve
        # different purposes (player stat merging vs. timeline enrichment). Single-instantiation
        # would require threading the analyzer instance through the pipeline, which adds
        # complexity for minimal gain since CombatAnalyzer is stateless.
        clutch_by_round: dict[int, list[dict]] = {}
        try:
            from opensight.domains.combat import ClutchResult, CombatAnalyzer

            combat_analyzer = CombatAnalyzer(demo_data)
            combat_result = combat_analyzer.analyze()
            for c in combat_result.clutch_situations:
                rn = c.round_num
                if rn not in clutch_by_round:
                    clutch_by_round[rn] = []
                clutch_by_round[rn].append(
                    {
                        "player": c.clutcher_name,
                        "player_steamid": c.clutcher_id,
                        "player_team": "CT" if c.clutcher_team == 3 else "T",
                        "scenario": c.scenario,
                        "won": c.result == ClutchResult.WON,
                        "kills_in_clutch": c.kills_in_clutch,
                    }
                )
            logger.debug(f"Extracted clutch data for {len(clutch_by_round)} rounds")
        except Exception as e:
            logger.warning(f"Could not extract clutch data for timeline: {e}")

        timeline = []
        kills = getattr(demo_data, "kills", [])
        rounds_data = getattr(demo_data, "rounds", [])
        player_names = getattr(demo_data, "player_names", {})
        grenades = getattr(demo_data, "grenades", [])
        blinds = getattr(demo_data, "blinds", [])
        map_name = getattr(demo_data, "map_name", "")
        # Persistent team IDs for teammate matching (handles halftime side swaps).
        # NOTE: kill events may use different steamids than player_names / grenades
        # (demoparser2 quirk), so we build a name→team lookup for reliable matching.
        player_persistent_teams = getattr(demo_data, "player_persistent_teams", {})
        name_to_team: dict[str, str] = {}
        for sid, team in player_persistent_teams.items():
            pname = player_names.get(sid, "")
            if pname:
                name_to_team[pname] = team

        # Import zone lookup for kill event enrichment
        try:
            from opensight.visualization.radar import get_zone_for_position
        except ImportError:
            get_zone_for_position = None

        logger.info(
            f"Building timeline: {len(kills)} kills, {len(rounds_data)} rounds, "
            f"{len(player_names)} players"
        )

        # Build round boundaries for tick-based inference (uses shared utility)
        round_boundaries = build_round_boundaries(rounds_data)
        round_info = {}  # round_num -> round data
        for r in rounds_data:
            round_num = getattr(r, "round_num", 0)
            if round_num:
                round_info[round_num] = r

        def tick_to_round_time(tick: int, round_start: int) -> float:
            """Convert tick to seconds from round start."""
            tick_rate = 64  # CS2 default
            return max(0, (tick - round_start) / tick_rate)

        # Check if kills have valid round data
        has_round_data = any(getattr(k, "round_num", 0) > 0 for k in kills[:10])

        # Group kills and events by round
        round_events: dict[int, list] = {}
        round_stats: dict[int, dict] = {}

        for kill in kills:
            round_num = getattr(kill, "round_num", 0)
            tick = getattr(kill, "tick", 0)

            # Infer round from tick if needed
            if round_num == 0:
                if round_boundaries and not has_round_data:
                    round_num = infer_round_from_tick(tick, round_boundaries)
                else:
                    round_num = 1

            if round_num not in round_events:
                round_events[round_num] = []
                round_stats[round_num] = {"ct_kills": 0, "t_kills": 0}

            # Get round start tick for time calculation
            round_start = round_boundaries.get(round_num, (0, 0))[0]
            time_seconds = tick_to_round_time(tick, round_start)

            # TODO(DRY): extract player name resolution to shared utility — this
            # "getattr name or player_names.get or Player_{id[-4:]}" pattern repeats
            # 4+ times in orchestrator.py. Consider resolve_player_name(event, attr, steam_id, names).
            # Resolve attacker name with proper fallback
            attacker_id = getattr(kill, "attacker_steamid", 0)
            attacker_name = getattr(kill, "attacker_name", "") or player_names.get(attacker_id)
            if not attacker_name and attacker_id:
                attacker_name = f"Player_{str(attacker_id)[-4:]}"
            elif not attacker_name:
                attacker_name = "Unknown"

            # Resolve victim name with proper fallback
            victim_id = getattr(kill, "victim_steamid", 0)
            victim_name = getattr(kill, "victim_name", "") or player_names.get(victim_id)
            if not victim_name and victim_id:
                victim_name = f"Player_{str(victim_id)[-4:]}"
            elif not victim_name:
                victim_name = "Unknown"
            attacker_side = str(getattr(kill, "attacker_side", "")).upper()
            victim_side = str(getattr(kill, "victim_side", "")).upper()

            # TODO(DRY): extract team side normalization to shared utility — this
            # '"CT" if "CT" in side else "T"' pattern repeats 9+ times across
            # orchestrator.py and parser.py. Consider normalize_team_side(side_str).
            attacker_team = "CT" if "CT" in attacker_side else "T"
            victim_team = "CT" if "CT" in victim_side else "T"

            # Track kill counts
            if attacker_team == "CT":
                round_stats[round_num]["ct_kills"] += 1
            else:
                round_stats[round_num]["t_kills"] += 1

            # Get position data
            killer_x = getattr(kill, "attacker_x", None)
            killer_y = getattr(kill, "attacker_y", None)
            victim_x = getattr(kill, "victim_x", None)
            victim_y = getattr(kill, "victim_y", None)

            # Create kill event with position data
            kill_event = {
                "tick": tick,
                "time_seconds": round(time_seconds, 1),
                "type": "kill",
                "killer": attacker_name,
                "killer_team": attacker_team,
                "killer_steamid": attacker_id,
                "victim": victim_name,
                "victim_team": victim_team,
                "victim_steamid": victim_id,
                "weapon": getattr(kill, "weapon", "unknown"),
                "headshot": bool(getattr(kill, "headshot", False)),
                "is_first_kill": len(round_events[round_num]) == 0,
            }

            # Add position data if available
            if killer_x is not None:
                kill_event["killer_x"] = killer_x
            if killer_y is not None:
                kill_event["killer_y"] = killer_y
            if victim_x is not None:
                kill_event["victim_x"] = victim_x
            if victim_y is not None:
                kill_event["victim_y"] = victim_y

            # Add zone data for strat engine
            if get_zone_for_position and map_name:
                if killer_x is not None and killer_y is not None:
                    kill_event["killer_zone"] = get_zone_for_position(map_name, killer_x, killer_y)
                if victim_x is not None and victim_y is not None:
                    kill_event["victim_zone"] = get_zone_for_position(map_name, victim_x, victim_y)

            round_events[round_num].append(kill_event)

        # Add bomb events from round data
        for round_num, r in round_info.items():
            if round_num not in round_events:
                round_events[round_num] = []
                round_stats[round_num] = {"ct_kills": 0, "t_kills": 0}

            round_start = round_boundaries.get(round_num, (0, 0))[0]

            # Check for bomb plant
            bomb_plant_tick = getattr(r, "bomb_plant_tick", None)
            if bomb_plant_tick:
                time_seconds = tick_to_round_time(bomb_plant_tick, round_start)
                round_events[round_num].append(
                    {
                        "tick": bomb_plant_tick,
                        "time_seconds": round(time_seconds, 1),
                        "type": "bomb_plant",
                        "player": getattr(r, "bomb_planter", "Unknown"),
                        "site": getattr(r, "bomb_site", "?"),
                    }
                )

            # Check for bomb defuse
            win_reason = str(getattr(r, "win_reason", "")).lower()
            if "defuse" in win_reason:
                defuse_tick = getattr(r, "end_tick", 0)
                time_seconds = tick_to_round_time(defuse_tick, round_start)
                round_events[round_num].append(
                    {
                        "tick": defuse_tick,
                        "time_seconds": round(time_seconds, 1),
                        "type": "bomb_defuse",
                        "player": getattr(r, "bomb_defuser", "Unknown"),
                    }
                )

            # Check for bomb explosion
            if "explod" in win_reason:
                explode_tick = getattr(r, "end_tick", 0)
                time_seconds = tick_to_round_time(explode_tick, round_start)
                round_events[round_num].append(
                    {
                        "tick": explode_tick,
                        "time_seconds": round(time_seconds, 1),
                        "type": "bomb_explode",
                    }
                )

        # Import zone lookup for utility event enrichment
        try:
            from opensight.core.map_zones import get_callout
        except ImportError:
            get_callout = None

        # Build utility events by round
        utility_by_round: dict[int, list] = {}
        for grenade in grenades:
            round_num = getattr(grenade, "round_num", 0)
            if round_num == 0 and round_boundaries:
                tick = getattr(grenade, "tick", 0)
                round_num = infer_round_from_tick(tick, round_boundaries)
            if round_num == 0:
                continue

            if round_num not in utility_by_round:
                utility_by_round[round_num] = []

            round_start = round_boundaries.get(round_num, (0, 0))[0]
            tick = getattr(grenade, "tick", 0)
            time_seconds = tick_to_round_time(tick, round_start)

            player_name = getattr(grenade, "player_name", "") or player_names.get(
                getattr(grenade, "player_steamid", 0), "Unknown"
            )
            player_side = str(getattr(grenade, "player_side", "Unknown")).upper()
            player_team = "CT" if "CT" in player_side else "T"

            # Determine zone from grenade coordinates (needed by strat engine)
            grenade_x = getattr(grenade, "x", None)
            grenade_y = getattr(grenade, "y", None)
            grenade_z = getattr(grenade, "z", None)
            zone = "Unknown"
            if get_callout and map_name and grenade_x is not None and grenade_y is not None:
                zone = get_callout(
                    map_name,
                    float(grenade_x),
                    float(grenade_y),
                    float(grenade_z) if grenade_z is not None else 0.0,
                )

            utility_by_round[round_num].append(
                {
                    "tick": tick,
                    "time_seconds": round(time_seconds, 1),
                    "type": getattr(grenade, "grenade_type", "unknown"),
                    "player": player_name,
                    "player_team": player_team,
                    "player_steamid": getattr(grenade, "player_steamid", 0),
                    "x": grenade_x,
                    "y": grenade_y,
                    "zone": zone,
                }
            )

        # Build blind events by round
        blinds_by_round: dict[int, list] = {}
        for blind in blinds:
            round_num = getattr(blind, "round_num", 0)
            if round_num == 0 and round_boundaries:
                tick = getattr(blind, "tick", 0)
                round_num = infer_round_from_tick(tick, round_boundaries)
            if round_num == 0:
                continue

            if round_num not in blinds_by_round:
                blinds_by_round[round_num] = []

            round_start = round_boundaries.get(round_num, (0, 0))[0]
            tick = getattr(blind, "tick", 0)
            time_seconds = tick_to_round_time(tick, round_start)

            attacker_name = getattr(blind, "attacker_name", "") or player_names.get(
                getattr(blind, "attacker_steamid", 0), "Unknown"
            )
            attacker_side = str(getattr(blind, "attacker_side", "Unknown")).upper()
            player_team = "CT" if "CT" in attacker_side else "T"

            blinds_by_round[round_num].append(
                {
                    "tick": tick,
                    "time_seconds": round(time_seconds, 1),
                    "type": "flash",
                    "player": attacker_name,
                    "player_team": player_team,
                    "victim": getattr(blind, "victim_name", ""),
                    "duration": getattr(blind, "blind_duration", 0),
                    "enemy": not getattr(blind, "is_teammate", False),
                }
            )

        # Enrich kill events with was_dry_peek detection.
        # A dry peek = player dies without any teammate utility within ~10s.
        #
        # Uses RAW TICK proximity instead of round-based matching because
        # round_boundaries can be unreliable (negative spans, overlapping starts).
        # Teams matched by name→persistent_team (kill steamids may differ from
        # grenade steamids — demoparser2 quirk).
        #
        # Window: 1280 ticks ≈ 10s @128tick, 20s @64tick.  Produces ~35-40%
        # dry peeks in a typical match (validated against golden_master.dem).
        import bisect

        DRY_PEEK_WINDOW_TICKS = 1280
        # Pre-build sorted grenade tick lists per team for O(log n) lookup
        _dp_team_ticks: dict[str, list[int]] = {}
        _dp_team_names: dict[str, list[str]] = {}
        for grenade in grenades:
            g_name = getattr(grenade, "player_name", "")
            g_team = name_to_team.get(g_name, "")
            g_tick = getattr(grenade, "tick", 0)
            if g_team and g_tick:
                _dp_team_ticks.setdefault(g_team, []).append(g_tick)
                _dp_team_names.setdefault(g_team, []).append(g_name)
        # Sort parallel lists by tick
        for t in _dp_team_ticks:
            pairs = sorted(zip(_dp_team_ticks[t], _dp_team_names[t], strict=True))
            _dp_team_ticks[t] = [p[0] for p in pairs]
            _dp_team_names[t] = [p[1] for p in pairs]

        for _round_num, events in round_events.items():
            for event in events:
                if event.get("type") != "kill":
                    continue
                kill_tick = event.get("tick", 0)
                victim_name = event.get("victim", "")
                victim_team_id = name_to_team.get(victim_name, "")
                if not kill_tick or not victim_team_id:
                    event["was_dry_peek"] = False
                    continue
                # Binary-search for nearest teammate utility within window
                ticks = _dp_team_ticks.get(victim_team_id, [])
                names = _dp_team_names.get(victim_team_id, [])
                idx = bisect.bisect_right(ticks, kill_tick) - 1
                had_support = False
                while idx >= 0:
                    delta = kill_tick - ticks[idx]
                    if delta > DRY_PEEK_WINDOW_TICKS:
                        break
                    if names[idx] != victim_name:  # exclude self utility
                        had_support = True
                        break
                    idx -= 1
                event["was_dry_peek"] = not had_support

        # Use actual round data if available, otherwise use analysis total_rounds
        total_rounds = (
            getattr(analysis, "total_rounds", 0) or len(round_boundaries) or len(round_events) or 30
        )

        # Build timeline entries for all rounds
        for round_num in range(1, total_rounds + 1):
            events = round_events.get(round_num, [])
            stats = round_stats.get(round_num, {"ct_kills": 0, "t_kills": 0})

            # Sort events by tick
            events.sort(key=lambda e: e.get("tick", 0))

            # Mark first kill
            kill_events = [e for e in events if e.get("type") == "kill"]
            if kill_events:
                kill_events[0]["is_first_kill"] = True
                for e in kill_events[1:]:
                    e["is_first_kill"] = False

            # Get winner from rounds data if available
            winner = "CT" if stats["ct_kills"] > stats["t_kills"] else "T"
            win_reason = "Elimination"
            round_type = "full_buy"

            if round_num in round_info:
                r = round_info[round_num]
                winner = str(getattr(r, "winner", winner)).upper()
                if "CT" not in winner and "T" not in winner:
                    winner = "CT" if stats["ct_kills"] > stats["t_kills"] else "T"
                win_reason = getattr(r, "win_reason", win_reason) or win_reason

                # Use round_type from parser (populated from equipment values)
                stored_round_type = getattr(r, "round_type", "")
                if stored_round_type:
                    round_type = stored_round_type

            # Fallback pistol detection if round_type not set from parser
            # Use is_pistol_round() for proper OT handling
            if round_type == "full_buy":
                from opensight.core.parser import is_pistol_round

                # Detect MR format from total rounds
                rounds_per_half = 12 if total_rounds <= 30 else 15
                if is_pistol_round(round_num, rounds_per_half):
                    round_type = "pistol"

            # Get first kill/death info
            first_kill = None
            first_death = None
            if kill_events:
                first_kill = kill_events[0].get("killer")
                first_death = kill_events[0].get("victim")

            # Calculate win probability timeline for this round
            momentum = None
            if calculate_win_probability is not None:
                momentum = self._calculate_round_momentum(events, winner, calculate_win_probability)
                # Add win probability to each event
                if momentum:
                    prob_by_tick = {p["tick"]: p for p in momentum.get("timeline", [])}
                    for event in events:
                        tick = event.get("tick", 0)
                        if tick in prob_by_tick:
                            event["ct_prob"] = prob_by_tick[tick]["ct_prob"]
                            event["t_prob"] = prob_by_tick[tick]["t_prob"]

            # Build Economy IQ data for this round (includes predictions)
            economy_iq = None
            if round_num in economy_by_round:
                round_econ = economy_by_round[round_num]
                round_preds = economy_predictions.get(round_num, {})
                economy_iq = {}
                for team_id, team_name in [(2, "t"), (3, "ct")]:
                    if team_id in round_econ:
                        tr = round_econ[team_id]
                        team_pred = round_preds.get(team_name, {})
                        economy_iq[team_name] = {
                            "loss_bonus": tr.loss_bonus,
                            "consecutive_losses": tr.consecutive_losses,
                            "equipment": tr.total_equipment,
                            "buy_type": tr.buy_type.value,
                            "decision_flag": tr.decision_flag,
                            "decision_grade": tr.decision_grade,
                            "loss_bonus_next": tr.loss_bonus_next,
                            "is_bad_force": tr.is_bad_force,
                            "is_good_force": tr.is_good_force,
                            # Economy Prediction Engine data
                            "prediction": team_pred if team_pred else None,
                        }

            # Extract structured event lists for consumers
            kills_list = [e for e in events if e.get("type") == "kill"]
            utility_list = utility_by_round.get(round_num, [])
            blinds_list = blinds_by_round.get(round_num, [])

            timeline.append(
                {
                    "round_num": round_num,
                    "round_type": round_type,
                    "winner": winner,
                    "win_reason": win_reason,
                    "first_kill": first_kill,
                    "first_death": first_death,
                    "ct_kills": stats["ct_kills"],
                    "t_kills": stats["t_kills"],
                    "events": events,
                    # Structured access to events by type (so consumers don't have to filter)
                    "kills": kills_list,
                    "utility": utility_list,
                    "blinds": blinds_list,
                    # player_positions: NOT populated in standard pipeline.
                    # Per-round position snapshots require tick data (include_ticks=True)
                    # which is too memory-intensive for default analysis.
                    # Kill events already contain attacker/victim positions and zone data.
                    # The strat engine handles absence gracefully (skips default detection).
                    # Schema: list[PlayerPositionSnapshot] - see core/schemas.py
                    "player_positions": [],
                    "momentum": momentum,
                    "economy": economy_iq,
                    "clutches": clutch_by_round.get(round_num, []),
                }
            )

        # Log timeline generation stats
        total_events = sum(len(r.get("events", [])) for r in timeline)
        rounds_with_events = sum(1 for r in timeline if r.get("events"))
        throws = sum(1 for r in timeline if r.get("momentum", {}).get("round_tag"))
        logger.info(
            f"Built round timeline: {len(timeline)} rounds, "
            f"{rounds_with_events} with events, {total_events} total events, "
            f"{throws} throw/heroic rounds"
        )

        return timeline

    def _calculate_round_momentum(self, events: list[dict], winner: str, calc_prob_fn) -> dict:
        """
        Calculate win probability timeline for a single round.

        Tracks probability at each state change (kill, bomb plant) to identify
        "throw" rounds (had >=80% prob, lost) and "heroic" rounds (had <=20%, won).

        Args:
            events: List of round events (kills, bomb plants, etc.)
            winner: Round winner ("CT" or "T")
            calc_prob_fn: Win probability calculation function

        Returns:
            Dict with momentum data including timeline, peak/min probs, and flags
        """
        prob_timeline = []

        # Initial state: 5v5, no bomb planted
        ct_alive = 5
        t_alive = 5
        bomb_planted = False

        # Add round start
        ct_prob = calc_prob_fn("CT", ct_alive, t_alive, bomb_planted)
        t_prob = calc_prob_fn("T", ct_alive, t_alive, bomb_planted)
        prob_timeline.append(
            {
                "tick": 0,
                "time": 0.0,
                "event": "round_start",
                "ct_alive": ct_alive,
                "t_alive": t_alive,
                "bomb_planted": bomb_planted,
                "ct_prob": round(ct_prob, 2),
                "t_prob": round(t_prob, 2),
                "desc": "Round start (5v5)",
            }
        )

        # Process each event in order
        for event in events:
            event_type = event.get("type", "")
            tick = event.get("tick", 0)
            time_sec = event.get("time_seconds", 0.0)

            if event_type == "kill":
                # Update alive count
                victim_team = event.get("victim_team", "")
                if victim_team == "CT":
                    ct_alive = max(0, ct_alive - 1)
                elif victim_team == "T":
                    t_alive = max(0, t_alive - 1)

                desc = f"{event.get('killer', 'Unknown')} killed {event.get('victim', 'Unknown')}"

            elif event_type == "bomb_plant":
                bomb_planted = True
                desc = "Bomb planted"

            elif event_type == "bomb_defuse":
                bomb_planted = False
                desc = "Bomb defused"

            elif event_type == "bomb_explode":
                desc = "Bomb exploded"

            else:
                continue  # Skip unknown events

            # Calculate new probabilities
            ct_prob = calc_prob_fn("CT", ct_alive, t_alive, bomb_planted)
            t_prob = calc_prob_fn("T", ct_alive, t_alive, bomb_planted)

            prob_timeline.append(
                {
                    "tick": tick,
                    "time": round(time_sec, 1),
                    "event": event_type,
                    "ct_alive": ct_alive,
                    "t_alive": t_alive,
                    "bomb_planted": bomb_planted,
                    "ct_prob": round(ct_prob, 2),
                    "t_prob": round(t_prob, 2),
                    "desc": desc,
                }
            )

        # Calculate peak/min probabilities
        ct_probs = [p["ct_prob"] for p in prob_timeline]
        t_probs = [p["t_prob"] for p in prob_timeline]

        ct_peak = max(ct_probs) if ct_probs else 0.5
        ct_min = min(ct_probs) if ct_probs else 0.5
        t_peak = max(t_probs) if t_probs else 0.5
        t_min = min(t_probs) if t_probs else 0.5

        # Determine throw/heroic status
        ct_is_throw = ct_peak >= 0.80 and winner == "T"
        ct_is_heroic = ct_min <= 0.20 and winner == "CT"
        t_is_throw = t_peak >= 0.80 and winner == "CT"
        t_is_heroic = t_min <= 0.20 and winner == "T"

        # Determine round tag
        if ct_is_throw:
            round_tag = "CT_THROW"
        elif t_is_throw:
            round_tag = "T_THROW"
        elif ct_is_heroic:
            round_tag = "CT_HEROIC"
        elif t_is_heroic:
            round_tag = "T_HEROIC"
        else:
            round_tag = ""

        return {
            "winner": winner,
            "ct_peak_prob": ct_peak,
            "ct_min_prob": ct_min,
            "t_peak_prob": t_peak,
            "t_min_prob": t_min,
            "ct_is_throw": ct_is_throw,
            "ct_is_heroic": ct_is_heroic,
            "t_is_throw": t_is_throw,
            "t_is_heroic": t_is_heroic,
            "round_tag": round_tag,
            "timeline": prob_timeline,
        }

    def _calculate_multikills(self, demo_data) -> dict[int, dict]:
        """Calculate multi-kill counts (2K, 3K, 4K, 5K) per player per round."""
        kills = getattr(demo_data, "kills", [])
        rounds = getattr(demo_data, "rounds", [])

        # Build round boundaries using shared utility
        round_boundaries = build_round_boundaries(rounds)

        # Check if kills have valid round data
        has_round_data = any(getattr(k, "round_num", 0) > 0 for k in kills[:10])

        # Count kills per player per round
        player_round_kills: dict[int, dict[int, int]] = {}  # steam_id -> {round_num -> kill_count}

        for kill in kills:
            attacker_id = getattr(kill, "attacker_steamid", 0)
            round_num = getattr(kill, "round_num", 0)

            if not attacker_id:
                continue

            # Infer round if missing (CRITICAL FIX for multikill detection)
            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(kill, "tick", 0)
                    round_num = infer_round_from_tick(tick, round_boundaries)
                else:
                    round_num = 1  # Fallback to round 1 if no boundary data

            if attacker_id not in player_round_kills:
                player_round_kills[attacker_id] = {}
            if round_num not in player_round_kills[attacker_id]:
                player_round_kills[attacker_id][round_num] = 0
            player_round_kills[attacker_id][round_num] += 1

        # Count 2K, 3K, 4K, 5K for each player
        result: dict[int, dict] = {}
        for steam_id, round_kills in player_round_kills.items():
            counts = {"2k": 0, "3k": 0, "4k": 0, "5k": 0}
            for _, kill_count in round_kills.items():
                if kill_count == 2:
                    counts["2k"] += 1
                elif kill_count == 3:
                    counts["3k"] += 1
                elif kill_count == 4:
                    counts["4k"] += 1
                elif kill_count >= 5:
                    counts["5k"] += 1
            result[steam_id] = counts

        return result

    def _build_timeline_graph_data(self, demo_data) -> dict:
        """Build round-by-round data for Leetify-style timeline graphs.

        Tracks per-round cumulative stats for all players:
        - kills, deaths, damage, awp_kills, enemies_flashed
        - Team information for grouping (CT/T)
        """
        kills = getattr(demo_data, "kills", [])
        damages = getattr(demo_data, "damages", [])
        blinds = getattr(demo_data, "blinds", [])
        player_names = getattr(demo_data, "player_names", {})
        player_teams = getattr(demo_data, "player_teams", {})
        rounds = getattr(demo_data, "rounds", [])

        # Build round boundaries using shared utility
        round_boundaries = build_round_boundaries(rounds)

        # Check if kills have valid round data
        has_round_data = any(getattr(k, "round_num", 0) > 0 for k in kills[:10])

        # Initialize per-player round data with all metrics
        # steam_id -> {round_num -> {kills, deaths, damage, awp_kills, enemies_flashed}}
        player_round_data: dict[int, dict[int, dict]] = {}

        # Get max rounds from demo_data or round boundaries
        max_round = getattr(demo_data, "num_rounds", 0) or len(round_boundaries) or 1

        def ensure_player_round(steam_id: int, round_num: int) -> None:
            """Ensure player and round entry exists with all metric fields."""
            if steam_id not in player_round_data:
                player_round_data[steam_id] = {}
            if round_num not in player_round_data[steam_id]:
                player_round_data[steam_id][round_num] = {
                    "kills": 0,
                    "deaths": 0,
                    "damage": 0,
                    "awp_kills": 0,
                    "enemies_flashed": 0,
                }

        # Process kills - track both attacker (kills) and victim (deaths)
        for kill in kills:
            attacker_id = getattr(kill, "attacker_steamid", 0)
            victim_id = getattr(kill, "victim_steamid", 0)
            round_num = getattr(kill, "round_num", 0)
            weapon = str(getattr(kill, "weapon", "")).lower()

            # Infer round if missing
            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(kill, "tick", 0)
                    round_num = infer_round_from_tick(tick, round_boundaries)
                else:
                    round_num = 1

            max_round = max(max_round, round_num)

            # Track kills for attacker
            if attacker_id:
                ensure_player_round(attacker_id, round_num)
                player_round_data[attacker_id][round_num]["kills"] += 1
                # Track AWP kills
                if "awp" in weapon:
                    player_round_data[attacker_id][round_num]["awp_kills"] += 1

            # Track deaths for victim
            if victim_id:
                ensure_player_round(victim_id, round_num)
                player_round_data[victim_id][round_num]["deaths"] += 1

        # Process damage
        for dmg in damages:
            attacker_id = getattr(dmg, "attacker_steamid", 0)
            round_num = getattr(dmg, "round_num", 0)
            damage_val = getattr(dmg, "damage", 0)

            if not attacker_id:
                continue

            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(dmg, "tick", 0)
                    round_num = infer_round_from_tick(tick, round_boundaries)
                else:
                    round_num = 1

            max_round = max(max_round, round_num)
            ensure_player_round(attacker_id, round_num)
            player_round_data[attacker_id][round_num]["damage"] += damage_val

        # Process blinds - count enemies flashed (duration > 0.5s, not teammate)
        for blind in blinds:
            attacker_id = getattr(blind, "attacker_steamid", 0)
            round_num = getattr(blind, "round_num", 0)
            duration = getattr(blind, "blind_duration", 0.0)
            is_teammate = getattr(blind, "is_teammate", False)

            # Only count enemy flashes with meaningful duration
            if not attacker_id or duration < 0.5 or is_teammate:
                continue

            if round_num == 0:
                if round_boundaries and not has_round_data:
                    tick = getattr(blind, "tick", 0)
                    round_num = infer_round_from_tick(tick, round_boundaries)
                else:
                    round_num = 1

            max_round = max(max_round, round_num)
            ensure_player_round(attacker_id, round_num)
            player_round_data[attacker_id][round_num]["enemies_flashed"] += 1

        # Filter to known players only (removes phantom entities)
        if player_names:
            known_players = set(player_names.keys())
            player_round_data = {
                sid: data for sid, data in player_round_data.items() if sid in known_players
            }

        # Build cumulative data for graphs
        players_timeline = []
        for steam_id, round_data in player_round_data.items():
            # Use last 4 digits for readability instead of raw Steam ID
            suffix = str(steam_id)[-4:] if steam_id else "0000"
            player_name = player_names.get(steam_id, f"Player_{suffix}")
            # Get team - prefer from player_teams dict, fallback to inferring from kills
            team = player_teams.get(steam_id, "Unknown")
            if team == "Unknown":
                # Try to infer from kills
                for kill in kills:
                    if getattr(kill, "attacker_steamid", 0) == steam_id:
                        side = str(getattr(kill, "attacker_side", "")).upper()
                        if "CT" in side:
                            team = "CT"
                        elif "T" in side:
                            team = "T"
                        break
                    if getattr(kill, "victim_steamid", 0) == steam_id:
                        side = str(getattr(kill, "victim_side", "")).upper()
                        if "CT" in side:
                            team = "CT"
                        elif "T" in side:
                            team = "T"
                        break

            # Build cumulative stats per round
            cumulative = {
                "kills": 0,
                "deaths": 0,
                "damage": 0,
                "awp_kills": 0,
                "enemies_flashed": 0,
            }
            rounds_list = []

            for r in range(1, max_round + 1):
                rd = round_data.get(
                    r,
                    {
                        "kills": 0,
                        "deaths": 0,
                        "damage": 0,
                        "awp_kills": 0,
                        "enemies_flashed": 0,
                    },
                )
                cumulative["kills"] += rd["kills"]
                cumulative["deaths"] += rd["deaths"]
                cumulative["damage"] += rd["damage"]
                cumulative["awp_kills"] += rd["awp_kills"]
                cumulative["enemies_flashed"] += rd["enemies_flashed"]

                rounds_list.append(
                    {
                        "round": r,
                        "kills": cumulative["kills"],
                        "deaths": cumulative["deaths"],
                        "damage": cumulative["damage"],
                        "awp_kills": cumulative["awp_kills"],
                        "enemies_flashed": cumulative["enemies_flashed"],
                        # Per-round values for tooltips
                        "round_kills": rd["kills"],
                        "round_deaths": rd["deaths"],
                        "round_damage": rd["damage"],
                    }
                )

            players_timeline.append(
                {
                    "steam_id": steam_id,
                    "name": player_name,
                    "team": team,
                    "rounds": rounds_list,
                }
            )

        # Sort players by team for better grouping
        players_timeline.sort(key=lambda p: (p["team"] != "CT", p["name"]))

        # Build round scores from rounds data (for Round Difference chart)
        round_scores = []
        ct_score = 0
        t_score = 0
        for r in range(1, max_round + 1):
            round_info = None
            for rd in rounds:
                if getattr(rd, "round_num", 0) == r:
                    round_info = rd
                    break

            if round_info:
                winner = str(getattr(round_info, "winner", "")).upper()
                if "CT" in winner:
                    ct_score += 1
                elif "T" in winner:
                    t_score += 1
                else:
                    # Infer from kill differential if no winner
                    ct_kills = sum(
                        1
                        for k in kills
                        if getattr(k, "round_num", 0) == r
                        and "CT" in str(getattr(k, "attacker_side", "")).upper()
                    )
                    t_kills = sum(
                        1
                        for k in kills
                        if getattr(k, "round_num", 0) == r
                        and "T" in str(getattr(k, "attacker_side", "")).upper()
                    )
                    if ct_kills > t_kills:
                        ct_score += 1
                    elif t_kills > ct_kills:
                        t_score += 1

            round_scores.append(
                {
                    "round": r,
                    "ct_score": ct_score,
                    "t_score": t_score,
                    "diff": ct_score - t_score,  # Positive = CT leading
                }
            )

        return {
            "max_rounds": max_round,
            "players": players_timeline,
            "round_scores": round_scores,
        }

    def _calculate_rws_direct(self, demo_data) -> dict[int, dict]:
        """Calculate RWS directly from demo data with better team handling."""
        kills = getattr(demo_data, "kills", [])
        damages = getattr(demo_data, "damages", [])
        rounds = getattr(demo_data, "rounds", [])
        player_names = getattr(demo_data, "player_names", {})

        if not rounds or not kills:
            return {}

        # Build player teams from FIRST HALF kills only (rounds 1-12)
        # to establish starting side, then handle halftime swap
        player_starting_teams: dict[int, str] = {}
        for kill in kills:
            round_num = getattr(kill, "round_num", 0)
            # Only use first half kills to determine starting team
            if round_num > 12:
                continue

            att_id = getattr(kill, "attacker_steamid", 0)
            att_side = str(getattr(kill, "attacker_side", "")).upper()
            vic_id = getattr(kill, "victim_steamid", 0)
            vic_side = str(getattr(kill, "victim_side", "")).upper()

            if att_id and att_id not in player_starting_teams:
                if "CT" in att_side:
                    player_starting_teams[att_id] = "CT"
                elif "T" in att_side:
                    player_starting_teams[att_id] = "T"
            if vic_id and vic_id not in player_starting_teams:
                if "CT" in vic_side:
                    player_starting_teams[vic_id] = "CT"
                elif "T" in vic_side:
                    player_starting_teams[vic_id] = "T"

        # If no first-half kills found, fall back to any kill data
        if not player_starting_teams:
            for kill in kills:
                att_id = getattr(kill, "attacker_steamid", 0)
                att_side = str(getattr(kill, "attacker_side", "")).upper()
                vic_id = getattr(kill, "victim_steamid", 0)
                vic_side = str(getattr(kill, "victim_side", "")).upper()

                if att_id and att_id not in player_starting_teams:
                    if "CT" in att_side:
                        player_starting_teams[att_id] = "CT"
                    elif "T" in att_side:
                        player_starting_teams[att_id] = "T"
                if vic_id and vic_id not in player_starting_teams:
                    if "CT" in vic_side:
                        player_starting_teams[vic_id] = "CT"
                    elif "T" in vic_side:
                        player_starting_teams[vic_id] = "T"

        # Group damage by round with attacker's side for that specific event
        round_damages: dict[int, dict[int, int]] = {}  # round_num -> {steam_id -> damage}
        round_player_sides: dict[int, dict[int, str]] = {}  # round_num -> {steam_id -> side}
        for dmg in damages:
            round_num = getattr(dmg, "round_num", 0)
            attacker_id = getattr(dmg, "attacker_steamid", 0)
            damage_val = getattr(dmg, "damage", 0)
            attacker_side = str(getattr(dmg, "attacker_side", "")).upper()
            victim_side = str(getattr(dmg, "victim_side", "")).upper()

            # Only count damage to enemies
            is_enemy_damage = (
                "CT" in attacker_side and "T" in victim_side and "CT" not in victim_side
            ) or ("T" in attacker_side and "CT" not in attacker_side and "CT" in victim_side)

            if attacker_id and round_num and is_enemy_damage:
                if round_num not in round_damages:
                    round_damages[round_num] = {}
                    round_player_sides[round_num] = {}
                if attacker_id not in round_damages[round_num]:
                    round_damages[round_num][attacker_id] = 0
                round_damages[round_num][attacker_id] += damage_val
                # Track the side for this player in this round
                if "CT" in attacker_side:
                    round_player_sides[round_num][attacker_id] = "CT"
                elif "T" in attacker_side:
                    round_player_sides[round_num][attacker_id] = "T"

        # Initialize player stats
        player_stats: dict[int, dict] = {}
        for pid in player_names:
            player_stats[pid] = {
                "rounds_played": 0,
                "rounds_won": 0,
                "total_rws": 0.0,
                "total_damage": 0,
            }

        # Calculate RWS for each round
        for round_info in rounds:
            round_num = getattr(round_info, "round_num", 0)
            winner = str(getattr(round_info, "winner", "")).upper()

            if not winner or winner == "UNKNOWN":
                continue

            round_dmg = round_damages.get(round_num, {})
            round_sides = round_player_sides.get(round_num, {})

            # Find winning players based on their side for THIS round
            winning_players = []
            for pid in player_stats:
                # Use halftime-aware side resolution (handles side swaps correctly)
                player_side = demo_data.get_player_side_for_round(pid, round_num)
                # Fall back to damage event sides if persistent teams unavailable
                if player_side not in ["CT", "T"]:
                    player_side = round_sides.get(pid, "")

                if player_side in ["CT", "T"]:
                    player_stats[pid]["rounds_played"] += 1
                    # Check if player is on winning team
                    if player_side == winner:
                        winning_players.append(pid)

            if not winning_players:
                continue

            # Calculate total damage by winning team
            winning_team_damage = sum(round_dmg.get(pid, 0) for pid in winning_players)

            # Distribute 100 RWS among winning team based on damage
            for pid in winning_players:
                player_stats[pid]["rounds_won"] += 1
                player_damage = round_dmg.get(pid, 0)
                player_stats[pid]["total_damage"] += player_damage

                if winning_team_damage > 0:
                    damage_share = player_damage / winning_team_damage
                    rws_this_round = damage_share * 100
                else:
                    # Equal share if no damage recorded
                    rws_this_round = 100 / len(winning_players)

                player_stats[pid]["total_rws"] += rws_this_round

        # Build results
        results = {}
        for pid, stats in player_stats.items():
            rounds_played = max(stats["rounds_played"], 1)
            results[pid] = {
                "avg_rws": round(stats["total_rws"] / rounds_played, 2),
                "total_rws": round(stats["total_rws"], 1),
                "rounds_won": stats["rounds_won"],
                "rounds_played": stats["rounds_played"],
                "damage_per_round": round(stats["total_damage"] / rounds_played, 1),
                "objective_completions": 0,
            }

        return results

    def _build_kill_matrix(self, demo_data) -> list[dict]:
        """Build kill matrix showing who killed who."""
        kills = getattr(demo_data, "kills", [])
        player_names = getattr(demo_data, "player_names", {})

        matrix = {}
        for kill in kills:
            attacker_id = getattr(kill, "attacker_steamid", 0)
            victim_id = getattr(kill, "victim_steamid", 0)
            attacker_name = player_names.get(attacker_id, getattr(kill, "attacker_name", "Unknown"))
            victim_name = player_names.get(victim_id, getattr(kill, "victim_name", "Unknown"))

            key = (attacker_name, victim_name)
            matrix[key] = matrix.get(key, 0) + 1

        return [{"attacker": k[0], "victim": k[1], "count": v} for k, v in matrix.items()]

    def _collect_dry_peek_events(self, demo_data, player_names: dict) -> dict:
        """Collect dry peek events with utility support data for visualization.

        Identifies opening duels (first kills of each round) and checks if they
        were supported by teammate utility (flash/smoke within 3s and 2000 units).

        Returns dict with:
        - events: List of entry events with positions and support info
        - summary: Aggregate statistics
        - constants: Support detection parameters
        """

        kills = getattr(demo_data, "kills", [])
        grenades = getattr(demo_data, "grenades", [])
        player_persistent_teams = getattr(demo_data, "player_persistent_teams", {})
        pnames = getattr(demo_data, "player_names", {})
        # Name→team lookup (kill steamids may differ from grenade steamids)
        name_to_team: dict[str, str] = {}
        for sid, team in player_persistent_teams.items():
            n = pnames.get(sid, "")
            if n:
                name_to_team[n] = team

        # Temporal window: 20 * TICK_RATE ticks.  time_seconds = tick/64 so this
        # equals 20 time_seconds — same window as the inline was_dry_peek check.
        # Covers 10-20 real seconds depending on whether the demo is 128 or 64 tick.
        TICK_RATE = 64
        SUPPORT_WINDOW_TICKS = int(20.0 * TICK_RATE)  # 1280 ticks
        SUPPORT_DISTANCE = 2000.0  # Game units

        # Group kills by round to find first kills
        round_kills: dict[int, list] = {}
        for kill in kills:
            round_num = getattr(kill, "round_num", 0)
            if round_num not in round_kills:
                round_kills[round_num] = []
            round_kills[round_num].append(kill)

        # Sort each round's kills by tick to find first kill
        for round_num in round_kills:
            round_kills[round_num].sort(key=lambda k: getattr(k, "tick", 0))

        # Build grenade lookup for faster access
        grenade_by_round: dict[int, list] = {}
        for grenade in grenades:
            g_round = getattr(grenade, "round_num", 0)
            if g_round not in grenade_by_round:
                grenade_by_round[g_round] = []
            grenade_by_round[g_round].append(grenade)

        events = []
        summary_by_player: dict[int, dict] = {}

        for round_num, kills_list in round_kills.items():
            if not kills_list:
                continue

            # First kill is the opening duel
            first_kill = kills_list[0]
            kill_tick = getattr(first_kill, "tick", 0)

            # Get attacker (entry fragger) info
            attacker_id = getattr(first_kill, "attacker_steamid", 0)
            attacker_name = player_names.get(attacker_id)
            if not attacker_name:
                # Try fallback to kill event's attacker_name attribute
                attacker_name = getattr(first_kill, "attacker_name", None)
            if not attacker_name and attacker_id:
                # Use friendly format instead of "Unknown"
                attacker_name = f"Player_{str(attacker_id)[-4:]}"
            elif not attacker_name:
                attacker_name = "Unknown"
            attacker_side = getattr(first_kill, "attacker_side", "") or ""
            attacker_team = "CT" if "CT" in attacker_side.upper() else "T"

            # Get victim (entry death) info
            victim_id = getattr(first_kill, "victim_steamid", 0)
            victim_name = player_names.get(victim_id)
            if not victim_name:
                # Try fallback to kill event's victim_name attribute
                victim_name = getattr(first_kill, "victim_name", None)
            if not victim_name and victim_id:
                # Use friendly format instead of "Unknown"
                victim_name = f"Player_{str(victim_id)[-4:]}"
            elif not victim_name:
                victim_name = "Unknown"
            victim_side = getattr(first_kill, "victim_side", "") or ""
            victim_team = "CT" if "CT" in victim_side.upper() else "T"

            # Get positions
            attacker_x = getattr(first_kill, "attacker_x", None)
            attacker_y = getattr(first_kill, "attacker_y", None)
            attacker_z = getattr(first_kill, "attacker_z", None) or 0
            victim_x = getattr(first_kill, "victim_x", None)
            victim_y = getattr(first_kill, "victim_y", None)
            victim_z = getattr(first_kill, "victim_z", None) or 0

            weapon = getattr(first_kill, "weapon", "unknown")

            # Find supporting utilities for the attacker
            def find_support_utilities(
                pos_x: float | None,
                pos_y: float | None,
                pos_z: float,
                team: str,
                player_name_arg: str = "",
            ) -> list[dict]:
                """Find flashes/smokes that supported this engagement."""
                if pos_x is None or pos_y is None:
                    return []

                # Resolve persistent team via name (steamids may differ)
                persistent_team = name_to_team.get(player_name_arg, "")

                support = []
                # Search ALL grenades by raw tick proximity (round_num-based
                # lookup is unreliable — round assignments can mismatch).
                for grenade in grenades:
                    # Only count flashes and smokes as support
                    g_type = getattr(grenade, "grenade_type", "").lower()
                    if "flash" not in g_type and "smoke" not in g_type:
                        continue

                    # Only detonations
                    if getattr(grenade, "event_type", "") != "detonate":
                        continue

                    # Teammate check via name→persistent team (halftime-safe)
                    thrower_name = getattr(grenade, "player_name", "")
                    if persistent_team:
                        if name_to_team.get(thrower_name, "") != persistent_team:
                            continue
                    else:
                        # Fallback: CT/T side match (works when sides are known)
                        g_side = getattr(grenade, "player_side", "") or ""
                        g_team = "CT" if "CT" in g_side.upper() else "T"
                        if g_team != team:
                            continue

                    # Need position
                    g_x = getattr(grenade, "x", None)
                    g_y = getattr(grenade, "y", None)
                    if g_x is None or g_y is None:
                        continue

                    # Temporal: detonated within 3s BEFORE the kill
                    g_tick = getattr(grenade, "tick", 0)
                    tick_diff = kill_tick - g_tick
                    if tick_diff < 0 or tick_diff > SUPPORT_WINDOW_TICKS:
                        continue

                    # Spatial: within 2000 units
                    g_z = getattr(grenade, "z", 0) or 0
                    dx = pos_x - g_x
                    dy = pos_y - g_y
                    dz = pos_z - g_z
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                    if distance <= SUPPORT_DISTANCE:
                        thrower_id = getattr(grenade, "player_steamid", 0)
                        support.append(
                            {
                                "type": "flashbang" if "flash" in g_type else "smoke",
                                "x": g_x,
                                "y": g_y,
                                "z": g_z,
                                "tick": g_tick,
                                "thrower_steamid": thrower_id,
                                "thrower_name": player_names.get(
                                    thrower_id,
                                    getattr(grenade, "player_name", "Unknown"),
                                ),
                                "time_before_ms": int(tick_diff * 1000 / TICK_RATE),
                                "distance": round(distance, 1),
                            }
                        )

                return support

            # Check support for attacker (entry fragger)
            attacker_support = find_support_utilities(
                attacker_x, attacker_y, attacker_z, attacker_team, attacker_name
            )
            attacker_is_supported = len(attacker_support) > 0

            # Record entry kill event (attacker perspective)
            if attacker_x is not None and attacker_y is not None:
                events.append(
                    {
                        "id": f"r{round_num}_entry_kill",
                        "event_type": "entry_kill",
                        "x": attacker_x,
                        "y": attacker_y,
                        "z": attacker_z,
                        "is_supported": attacker_is_supported,
                        "player_steamid": attacker_id,
                        "player_name": attacker_name,
                        "side": attacker_team,
                        "round_num": round_num,
                        "tick": kill_tick,
                        "weapon": weapon,
                        "support_utilities": attacker_support,
                    }
                )

                # Update summary for attacker
                if attacker_id not in summary_by_player:
                    summary_by_player[attacker_id] = {
                        "name": attacker_name,
                        "supported": 0,
                        "unsupported": 0,
                    }
                if attacker_is_supported:
                    summary_by_player[attacker_id]["supported"] += 1
                else:
                    summary_by_player[attacker_id]["unsupported"] += 1

            # Check support for victim (entry death)
            victim_support = find_support_utilities(
                victim_x, victim_y, victim_z, victim_team, victim_name
            )
            victim_is_supported = len(victim_support) > 0

            # Record entry death event (victim perspective)
            if victim_x is not None and victim_y is not None:
                events.append(
                    {
                        "id": f"r{round_num}_entry_death",
                        "event_type": "entry_death",
                        "x": victim_x,
                        "y": victim_y,
                        "z": victim_z,
                        "is_supported": victim_is_supported,
                        "player_steamid": victim_id,
                        "player_name": victim_name,
                        "side": victim_team,
                        "round_num": round_num,
                        "tick": kill_tick,
                        "weapon": weapon,
                        "support_utilities": victim_support,
                    }
                )

                # Update summary for victim
                if victim_id not in summary_by_player:
                    summary_by_player[victim_id] = {
                        "name": victim_name,
                        "supported": 0,
                        "unsupported": 0,
                    }
                # Victim deaths count towards their dry peek stats
                if victim_is_supported:
                    summary_by_player[victim_id]["supported"] += 1
                else:
                    summary_by_player[victim_id]["unsupported"] += 1

        # Calculate summary statistics
        total_entries = len([e for e in events if e["event_type"] == "entry_kill"])
        supported_entries = len(
            [e for e in events if e["event_type"] == "entry_kill" and e["is_supported"]]
        )
        dry_peek_entries = total_entries - supported_entries

        # Calculate per-player dry peek rates
        by_player = {}
        for steam_id, data in summary_by_player.items():
            total = data["supported"] + data["unsupported"]
            by_player[str(steam_id)] = {
                "name": data["name"],
                "supported": data["supported"],
                "unsupported": data["unsupported"],
                "dry_peek_rate": round(data["unsupported"] / max(total, 1) * 100, 1),
            }

        return {
            "events": events,
            "summary": {
                "total_entries": total_entries,
                "supported_entries": supported_entries,
                "dry_peek_entries": dry_peek_entries,
                "dry_peek_rate": round(dry_peek_entries / max(total_entries, 1) * 100, 1),
                "by_player": by_player,
            },
            "constants": {
                "support_radius_units": int(SUPPORT_DISTANCE),
                "support_window_seconds": 20.0,
            },
        }

    def _build_heatmap_data(self, demo_data) -> dict:
        """Build comprehensive position data for heatmap visualization.

        Includes zone detection, side info, phase (pre/post plant), and economy context.
        """
        kills = getattr(demo_data, "kills", [])
        rounds = getattr(demo_data, "rounds", [])
        player_names = getattr(demo_data, "player_names", {})
        map_name = getattr(demo_data, "map_name", "").lower()

        # Build round lookup for bomb plant and economy data
        round_info = {}
        for r in rounds:
            round_num = getattr(r, "round_num", 0)
            round_info[round_num] = {
                "bomb_plant_tick": getattr(r, "bomb_plant_tick", None),
                "bomb_site": getattr(r, "bomb_site", ""),
                "ct_equipment": getattr(r, "ct_equipment_value", 0),
                "t_equipment": getattr(r, "t_equipment_value", 0),
                "round_type": getattr(r, "round_type", ""),
            }

        # Import zone detection function
        try:
            from opensight.visualization.radar import (
                MAP_ZONES,
                classify_round_economy,
                get_zone_for_position,
            )

            has_zones = map_name in MAP_ZONES
        except ImportError:
            has_zones = False
            MAP_ZONES: dict = {}  # type: ignore[no-redef]

            def get_zone_for_position(
                map_name: str, x: float, y: float, z: float | None = None
            ) -> str:
                _ = map_name, x, y, z  # Unused in fallback
                return "World"  # Consistent fallback instead of "Unknown"

            def classify_round_economy(equipment_value: int, is_pistol_round: bool) -> str:
                _ = equipment_value, is_pistol_round  # Unused in fallback
                return "unknown"

        # Import pistol round detection
        try:
            from opensight.core.parser import is_pistol_round as check_pistol
        except ImportError:
            # Simple fallback if import fails
            def check_pistol(round_num: int, rounds_per_half: int = 12) -> bool:
                return round_num == 1 or round_num == rounds_per_half + 1

        # Detect MR format from total rounds
        total_rounds = len(rounds)
        rounds_per_half = 12 if total_rounds <= 30 else 15

        kill_positions = []
        death_positions = []
        zone_stats: dict[str, dict] = {}

        for kill in kills:
            round_num = getattr(kill, "round_num", 0)
            tick = getattr(kill, "tick", 0)
            r_info = round_info.get(round_num, {})  # type: ignore[arg-type]

            # Determine phase (pre-plant vs post-plant)
            bomb_plant_tick = r_info.get("bomb_plant_tick", 0)  # type: ignore[union-attr]
            phase = "pre_plant"
            if bomb_plant_tick and tick >= int(bomb_plant_tick or 0):  # type: ignore[arg-type]
                phase = "post_plant"

            # Determine economy round type
            attacker_side = getattr(kill, "attacker_side", "") or ""
            is_pistol = check_pistol(round_num, rounds_per_half)
            eq_key = "t_equipment" if "T" in attacker_side.upper() else "ct_equipment"
            eq_raw = r_info.get(eq_key, 0)  # type: ignore[union-attr]
            eq_value = int(eq_raw) if eq_raw else 0  # type: ignore[arg-type]
            round_type_raw = r_info.get("round_type", "")  # type: ignore[union-attr]
            stored_round_type = str(round_type_raw) if round_type_raw else ""  # type: ignore[arg-type]
            if stored_round_type:
                round_type = str(stored_round_type)
            elif has_zones:
                round_type = classify_round_economy(eq_value, is_pistol)
            else:
                round_type = "pistol" if is_pistol else "unknown"

            # Kill position (attacker)
            ax = getattr(kill, "attacker_x", None)
            ay = getattr(kill, "attacker_y", None)
            if ax is not None and ay is not None:
                az = getattr(kill, "attacker_z", 0) or 0
                # Always call zone detection - it handles unsupported maps with "World" fallback
                zone = get_zone_for_position(map_name, ax, ay, az)
                kill_positions.append(
                    {
                        "x": ax,
                        "y": ay,
                        "z": az,
                        "zone": zone,
                        "side": attacker_side,
                        "phase": phase,
                        "round_type": round_type,
                        "round_num": round_num,
                        "player_name": player_names.get(
                            getattr(kill, "attacker_steamid", 0), "Unknown"
                        ),
                        "player_steamid": getattr(kill, "attacker_steamid", 0),
                        "weapon": getattr(kill, "weapon", ""),
                        "headshot": getattr(kill, "headshot", False),
                    }
                )

                # Update zone stats for kills
                if zone not in zone_stats:
                    zone_stats[zone] = {
                        "kills": 0,
                        "deaths": 0,
                        "ct_kills": 0,
                        "t_kills": 0,
                    }
                zone_stats[zone]["kills"] += 1
                if "CT" in attacker_side.upper():
                    zone_stats[zone]["ct_kills"] += 1
                else:
                    zone_stats[zone]["t_kills"] += 1

            # Death position (victim)
            vx = getattr(kill, "victim_x", None)
            vy = getattr(kill, "victim_y", None)
            if vx is not None and vy is not None:
                vz = getattr(kill, "victim_z", 0) or 0
                victim_side = getattr(kill, "victim_side", "") or ""
                # Always call zone detection - it handles unsupported maps with "World" fallback
                zone = get_zone_for_position(map_name, vx, vy, vz)
                death_positions.append(
                    {
                        "x": vx,
                        "y": vy,
                        "z": vz,
                        "zone": zone,
                        "side": victim_side,
                        "phase": phase,
                        "round_type": round_type,
                        "round_num": round_num,
                        "player_name": player_names.get(
                            getattr(kill, "victim_steamid", 0), "Unknown"
                        ),
                        "player_steamid": getattr(kill, "victim_steamid", 0),
                    }
                )

                # Update zone stats for deaths
                if zone not in zone_stats:
                    zone_stats[zone] = {
                        "kills": 0,
                        "deaths": 0,
                        "ct_kills": 0,
                        "t_kills": 0,
                    }
                zone_stats[zone]["deaths"] += 1

        # Calculate zone K/D ratios and percentages
        total_kills = len(kill_positions)
        for _zone, stats in zone_stats.items():
            k = stats["kills"]
            d = stats["deaths"]
            stats["kd_ratio"] = round(k / max(d, 1), 2)
            stats["kill_pct"] = round(k / max(total_kills, 1) * 100, 1)

        # Get zone definitions for frontend if available
        zone_definitions = {}
        if has_zones:
            try:
                zone_definitions = MAP_ZONES.get(map_name, {})
            except Exception:
                pass

        # Collect dry peek visualization data
        dry_peek_data = self._collect_dry_peek_events(demo_data, player_names)

        # Collect grenade landing positions for visualization
        # Only include detonation events to avoid duplicates (thrown + detonate + expire)
        grenades = getattr(demo_data, "grenades", [])
        grenade_positions = []
        for grenade in grenades:
            if getattr(grenade, "event_type", "") != "detonate":
                continue
            gx = getattr(grenade, "x", None)
            gy = getattr(grenade, "y", None)
            if gx is not None and gy is not None:
                gz = getattr(grenade, "z", 0) or 0
                round_num = getattr(grenade, "round_num", 0)
                r_info = round_info.get(round_num, {})

                # Determine grenade type
                grenade_type = getattr(grenade, "grenade_type", "unknown")

                # Get player info
                player_steamid = getattr(grenade, "player_steamid", 0)
                player_name = player_names.get(player_steamid, "Unknown")
                player_side = str(getattr(grenade, "player_side", "")).upper()
                team = "CT" if "CT" in player_side else "T"

                # Get zone
                zone = get_zone_for_position(map_name, gx, gy, gz)

                # Determine round type
                stored_round_type = (
                    str(r_info.get("round_type", "")) if r_info.get("round_type") else ""
                )
                if stored_round_type:
                    round_type = stored_round_type
                else:
                    is_pistol = check_pistol(round_num, rounds_per_half)
                    round_type = "pistol" if is_pistol else "unknown"

                grenade_positions.append(
                    {
                        "x": gx,
                        "y": gy,
                        "z": gz,
                        "zone": zone,
                        "grenade_type": grenade_type,
                        "player_name": player_name,
                        "player_steamid": player_steamid,
                        "team": team,
                        "round_num": round_num,
                        "round_type": round_type,
                    }
                )

        return {
            "map_name": map_name,
            "kill_positions": kill_positions,
            "death_positions": death_positions,
            "grenade_positions": grenade_positions,
            "zone_stats": zone_stats,
            "zone_definitions": zone_definitions,
            "dry_peek_data": dry_peek_data,
        }

    def _generate_coaching_insights(self, demo_data, analysis, players: dict) -> list[dict]:
        """
        Generate comprehensive, data-driven coaching insights for each player.

        Inspired by Leetify's detailed coaching system, this provides:
        - Specific metrics with actual numbers and comparisons
        - Role-specific analysis
        - Comparative insights vs teammates and match averages
        - Player identity/archetype detection
        - Actionable improvement recommendations
        """
        coaching = []

        # Calculate match-wide statistics for comparison
        match_stats = self._calculate_match_averages(players)

        # Find top performers in each category for comparative insights
        top_performers = self._find_top_performers(players)

        for steam_id, player in players.items():
            insights = []
            name = player["name"]
            stats = player["stats"]
            rating = player["rating"]
            advanced = player["advanced"]
            utility = player["utility"]
            duels = player["duels"]
            entry = player.get("entry", {})
            trades = player.get("trades", {})
            clutches = player.get("clutches", {})
            rws = player.get("rws", {})

            # Detect role with more detail
            role, role_confidence = self._detect_role_detailed(player, match_stats)

            # Determine player identity/archetype (like Leetify's "The Cleanup")
            identity = self._determine_player_identity(player, match_stats, top_performers)

            # ==========================================
            # AIM & MECHANICS INSIGHTS
            # ==========================================

            # Time to Damage (TTD) - reaction time analysis
            ttd = advanced.get("ttd_median_ms", 0)
            if ttd > 0:
                ttd_rating = self._get_ttd_rating(ttd)
                avg_ttd = match_stats.get("avg_ttd", 350)
                diff = ttd - avg_ttd
                if diff > 50:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"TTD {ttd:.0f}ms is {diff:.0f}ms slower than match average ({avg_ttd:.0f}ms) - practice pre-aiming common angles",
                            "category": "Aim",
                            "metric": "ttd_ms",
                            "value": ttd,
                            "benchmark": avg_ttd,
                            "severity": "high" if diff > 100 else "medium",
                        }
                    )
                elif diff < -50:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Fast reactions: {ttd:.0f}ms TTD ({abs(diff):.0f}ms faster than match avg) - {ttd_rating}",
                            "category": "Aim",
                            "metric": "ttd_ms",
                            "value": ttd,
                            "benchmark": avg_ttd,
                        }
                    )

            # Crosshair Placement (CP) - angle accuracy
            cp = advanced.get("cp_median_error_deg", 0)
            if cp > 0:
                cp_rating = self._get_cp_rating(cp)
                avg_cp = match_stats.get("avg_cp", 8)
                diff = cp - avg_cp
                if cp > 10:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Crosshair placement: {cp:.1f}Â° error ({diff:+.1f}Â° vs avg). Pre-aim head level at common angles",
                            "category": "Aim",
                            "metric": "cp_error_deg",
                            "value": cp,
                            "benchmark": avg_cp,
                            "severity": "high" if cp > 15 else "medium",
                        }
                    )
                elif cp < 5:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Excellent crosshair placement: {cp:.1f}Â° error - {cp_rating}",
                            "category": "Aim",
                            "metric": "cp_error_deg",
                            "value": cp,
                        }
                    )

            # Headshot percentage analysis
            hs_pct = stats.get("headshot_pct", 0)
            if hs_pct > 0:
                avg_hs = match_stats.get("avg_hs_pct", 35)
                if hs_pct >= 50:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Elite headshot %: {hs_pct:.0f}% ({hs_pct - avg_hs:+.0f}% vs match avg) - precision aiming",
                            "category": "Aim",
                            "metric": "headshot_pct",
                            "value": hs_pct,
                        }
                    )
                elif hs_pct < 25:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Low HS%: {hs_pct:.0f}% (avg: {avg_hs:.0f}%). Focus on head-level crosshair, less spraying",
                            "category": "Aim",
                            "metric": "headshot_pct",
                            "value": hs_pct,
                            "benchmark": avg_hs,
                        }
                    )

            # ==========================================
            # OPENING DUEL / ENTRY INSIGHTS
            # ==========================================

            entry_attempts = entry.get("entry_attempts", 0)
            entry_kills = entry.get("entry_kills", 0)
            entry_deaths = entry.get("entry_deaths", 0)
            entry_success = entry.get("entry_success_pct", 0)

            if entry_attempts >= 3:
                avg_entry_success = match_stats.get("avg_entry_success", 50)

                if entry_success < 35:
                    insights.append(
                        {
                            "type": "mistake",
                            "message": f"Opening duels: {entry_success:.0f}% success ({entry_kills}W-{entry_deaths}L). Use utility before peeking or change entry spots",
                            "category": "Entry",
                            "metric": "opening_duel_success",
                            "value": entry_success,
                            "benchmark": avg_entry_success,
                            "severity": "high",
                        }
                    )
                elif entry_success >= 65:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Dominant entry fragging: {entry_success:.0f}% ({entry_kills}W-{entry_deaths}L) - {entry_success - avg_entry_success:+.0f}% vs match avg",
                            "category": "Entry",
                            "metric": "opening_duel_success",
                            "value": entry_success,
                        }
                    )

                # Entry attempt rate analysis
                entry_rate = entry.get("entry_attempts_pct", 0)
                if role == "entry" and entry_rate < 20:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Low entry rate for entry role: {entry_rate:.0f}% of rounds. Lead more site takes with utility",
                            "category": "Role",
                            "metric": "entry_attempt_rate",
                            "value": entry_rate,
                        }
                    )

            # ==========================================
            # TRADE KILL INSIGHTS
            # ==========================================

            trade_kills = trades.get("trade_kills", 0) or duels.get("trade_kills", 0)
            deaths_traded = trades.get("deaths_traded", 0) or duels.get("traded_deaths", 0)
            total_deaths = stats.get("deaths", 1)

            trade_rate = (deaths_traded / max(1, total_deaths)) * 100 if total_deaths > 0 else 0
            avg_trade_rate = match_stats.get("avg_trade_rate", 40)

            # Find best trader on team for comparison
            best_trader = top_performers.get("trade_kills", {})

            if trade_kills >= 5:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"Strong trader: {trade_kills} trade kills - reliable teammate support",
                        "category": "Trading",
                        "metric": "trade_kills",
                        "value": trade_kills,
                    }
                )

            if total_deaths >= 8 and trade_rate < 30:
                best_trader_name = best_trader.get("name", "teammate")
                insights.append(
                    {
                        "type": "warning",
                        "message": f"Only {trade_rate:.0f}% of deaths traded ({deaths_traded}/{total_deaths}). Stay closer to teammates, especially {best_trader_name}",
                        "category": "Trading",
                        "metric": "deaths_traded_rate",
                        "value": trade_rate,
                        "benchmark": avg_trade_rate,
                        "severity": "medium",
                    }
                )

            # ==========================================
            # CLUTCH INSIGHTS
            # ==========================================

            clutch_wins = clutches.get("clutch_wins", 0) or duels.get("clutch_wins", 0)
            clutch_attempts = clutches.get("clutch_wins", 0) + clutches.get("clutch_losses", 0)
            if clutch_attempts == 0:
                clutch_attempts = duels.get("clutch_attempts", 0)

            if clutch_attempts >= 3:
                clutch_pct = clutches.get("clutch_success_pct", 0)
                if clutch_pct == 0 and clutch_attempts > 0:
                    clutch_pct = (clutch_wins / clutch_attempts) * 100

                # Check for impressive clutches (1v3, 1v4, 1v5)
                v3_plus = (
                    clutches.get("v3_wins", 0)
                    + clutches.get("v4_wins", 0)
                    + clutches.get("v5_wins", 0)
                )

                if v3_plus >= 1:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Clutch master: Won 1v3+ situations {v3_plus} time(s) - ice cold under pressure",
                            "category": "Clutch",
                            "metric": "difficult_clutches",
                            "value": v3_plus,
                        }
                    )
                elif clutch_pct >= 40 and clutch_attempts >= 3:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Reliable clutcher: {clutch_pct:.0f}% success rate ({clutch_wins}/{clutch_attempts})",
                            "category": "Clutch",
                            "metric": "clutch_success",
                            "value": clutch_pct,
                        }
                    )
                elif clutch_pct < 20 and clutch_attempts >= 4:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Clutch struggles: {clutch_pct:.0f}% ({clutch_wins}/{clutch_attempts}). Play for info, use utility to isolate fights",
                            "category": "Clutch",
                            "metric": "clutch_success",
                            "value": clutch_pct,
                            "severity": "medium",
                        }
                    )

            # ==========================================
            # UTILITY USAGE INSIGHTS
            # ==========================================

            flashes = utility.get("flashbangs_thrown", 0)
            smokes = utility.get("smokes_thrown", 0)
            he_thrown = utility.get("he_thrown", 0)
            molotovs = utility.get("molotovs_thrown", 0)
            total_util = flashes + smokes + he_thrown + molotovs
            enemies_flashed = utility.get("enemies_flashed", 0)
            flash_assists = utility.get("flash_assists", 0)
            he_damage = utility.get("he_damage", 0)

            rounds_played = rws.get("rounds_played", 0) or 30  # Approximate if missing
            util_per_round = total_util / max(1, rounds_played)
            avg_util_per_round = match_stats.get("avg_util_per_round", 2.5)

            if flashes > 0:
                flash_effectiveness = (enemies_flashed / max(1, flashes)) * 100
                if flash_effectiveness < 30 and flashes >= 5:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Flash effectiveness: {flash_effectiveness:.0f}% ({enemies_flashed} blinds from {flashes} flashes). Learn pop-flashes for common angles",
                            "category": "Utility",
                            "metric": "flash_effectiveness",
                            "value": flash_effectiveness,
                            "severity": "medium",
                        }
                    )
                elif flash_effectiveness >= 60 and flash_assists >= 2:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Quality flashbangs: {enemies_flashed} enemies blinded, {flash_assists} flash assists - great support play",
                            "category": "Utility",
                            "metric": "flash_effectiveness",
                            "value": flash_effectiveness,
                        }
                    )

            if util_per_round < 1.5 and rounds_played >= 15:
                insights.append(
                    {
                        "type": "warning",
                        "message": f"Low utility usage: {util_per_round:.1f}/round (avg: {avg_util_per_round:.1f}). Buy and use more grenades",
                        "category": "Utility",
                        "metric": "utility_per_round",
                        "value": util_per_round,
                        "benchmark": avg_util_per_round,
                    }
                )

            if he_damage >= 150:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"Effective HE grenades: {he_damage} damage total - valuable chip damage",
                        "category": "Utility",
                        "metric": "he_damage",
                        "value": he_damage,
                    }
                )

            # ==========================================
            # IMPACT & DAMAGE INSIGHTS
            # ==========================================

            adr = stats.get("adr", 0)
            avg_adr = match_stats.get("avg_adr", 75)
            kast = rating.get("kast_percentage", 0)
            hltv = rating.get("hltv_rating", 0)
            kd_ratio = stats.get("kd_ratio", 1)

            # Multi-kill analysis
            multikills = stats.get("3k", 0) + stats.get("4k", 0) + stats.get("5k", 0)

            if adr >= 90:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"High impact: {adr:.0f} ADR ({adr - avg_adr:+.0f} vs avg) - consistent round damage",
                        "category": "Impact",
                        "metric": "adr",
                        "value": adr,
                    }
                )
            elif adr < 60:
                insights.append(
                    {
                        "type": "warning",
                        "message": f"Low ADR: {adr:.0f} (avg: {avg_adr:.0f}). Find more engagements, use utility to create opportunities",
                        "category": "Impact",
                        "metric": "adr",
                        "value": adr,
                        "benchmark": avg_adr,
                        "severity": "high" if adr < 50 else "medium",
                    }
                )

            if multikills >= 3:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"Round-winning plays: {multikills} multi-kills (3K+) - clutch performer in key rounds",
                        "category": "Impact",
                        "metric": "multikills",
                        "value": multikills,
                    }
                )

            if kast < 55 and rounds_played >= 15:
                insights.append(
                    {
                        "type": "warning",
                        "message": f"Low KAST: {kast:.0f}% - dying without impact too often. Focus on trading and staying alive",
                        "category": "Consistency",
                        "metric": "kast",
                        "value": kast,
                        "severity": "high",
                    }
                )
            elif kast >= 80:
                insights.append(
                    {
                        "type": "positive",
                        "message": f"Exceptional consistency: {kast:.0f}% KAST - contributing in nearly every round",
                        "category": "Consistency",
                        "metric": "kast",
                        "value": kast,
                    }
                )

            # ==========================================
            # RWS (Round Win Share) INSIGHTS
            # ==========================================

            avg_rws = rws.get("avg_rws", 0)
            if avg_rws > 0:
                if avg_rws >= 12:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"High impact in won rounds: {avg_rws:.1f} RWS - key contributor to team victories",
                            "category": "Impact",
                            "metric": "rws",
                            "value": avg_rws,
                        }
                    )
                elif avg_rws < 6:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Low RWS: {avg_rws:.1f} - limited impact in rounds your team wins. Be more aggressive in won rounds",
                            "category": "Impact",
                            "metric": "rws",
                            "value": avg_rws,
                        }
                    )

            # ==========================================
            # ROLE-SPECIFIC INSIGHTS
            # ==========================================

            if role == "entry":
                if entry_success < 45 and entry_attempts >= 4:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Entry role struggling ({entry_success:.0f}% success). Coordinate utility with teammates before taking fights",
                            "category": "Role",
                            "severity": "medium",
                        }
                    )
            elif role == "support":
                if flash_assists < 2 and enemies_flashed < 5:
                    insights.append(
                        {
                            "type": "warning",
                            "message": "Support role needs more flash impact. Focus on enabling teammates with utility",
                            "category": "Role",
                        }
                    )
            elif role == "awp":
                awp_kd = kd_ratio  # Simplified - ideally would track AWP-specific K/D
                if awp_kd < 1.0:
                    insights.append(
                        {
                            "type": "warning",
                            "message": "AWPer dying too often - hold angles, avoid aggressive peeks. $4750 value at risk each death",
                            "category": "Role",
                            "severity": "high",
                        }
                    )

            # ==========================================
            # SORT AND PRIORITIZE INSIGHTS
            # ==========================================

            # Sort by severity and type (mistakes first, then warnings, then positives)
            type_priority = {"mistake": 0, "warning": 1, "positive": 2}
            severity_priority = {"high": 0, "medium": 1, "low": 2}

            insights.sort(
                key=lambda x: (
                    type_priority.get(x.get("type"), 2),
                    severity_priority.get(x.get("severity", "low"), 2),
                )
            )

            # Ensure at least one insight
            if not insights:
                if hltv >= 1.0:
                    insights.append(
                        {
                            "type": "positive",
                            "message": f"Solid match performance: {hltv:.2f} rating, {kast:.0f}% KAST",
                            "category": "Overall",
                        }
                    )
                else:
                    insights.append(
                        {
                            "type": "warning",
                            "message": f"Below average match ({hltv:.2f} rating). Focus on survival and trading with teammates",
                            "category": "Overall",
                        }
                    )

            coaching.append(
                {
                    "player_name": name,
                    "steam_id": steam_id,
                    "role": role,
                    "role_confidence": role_confidence,
                    "identity": identity,
                    "stats_summary": {
                        "rating": round(hltv, 2),
                        "adr": round(adr, 0),
                        "kast": round(kast, 0),
                        "kd": round(kd_ratio, 2),
                        "ttd_ms": round(advanced.get("ttd_median_ms", 0), 0),
                        "cp_deg": round(advanced.get("cp_median_error_deg", 0), 1),
                        "entry_success": round(entry_success, 0),
                        "trade_rate": round(trade_rate, 0),
                        "clutch_pct": round(clutches.get("clutch_success_pct", 0), 0),
                    },
                    "insights": insights[:8],  # Top 8 insights per player
                    "ai_summary": player.get("ai_summary", ""),  # LLM-generated summary
                }
            )

        return coaching

    def _generate_ai_summaries(self, players: dict, analysis) -> None:
        """
        Generate AI-powered match summaries for each player using LLM.

        Uses a single batched LLM call for all players (1 call instead of 10).
        Falls back to per-player calls if batching fails.

        Modifies players dict in-place, adding "ai_summary" field to each player.

        Args:
            players: Dict of player data (steam_id -> player_dict)
            analysis: MatchAnalysis object with match metadata
        """
        try:
            from opensight.ai.llm_client import generate_batch_summaries, generate_match_summary

            logger.info("Generating AI-powered coaching summaries with LLM")

            # Build match context
            match_context = {
                "map_name": getattr(analysis, "map_name", ""),
                "total_rounds": getattr(analysis, "total_rounds", 0),
                "team1_score": getattr(analysis, "team1_score", 0),
                "team2_score": getattr(analysis, "team2_score", 0),
            }

            # Build stats list for all players
            all_stats = []
            steam_id_to_name = {}
            for steam_id, player_data in players.items():
                player_stats = self._extract_player_stats_for_llm(player_data)
                all_stats.append(player_stats)
                steam_id_to_name[steam_id] = player_stats["name"]

            # Try batched call first (1 call instead of 10)
            batch_results = {}
            try:
                batch_results = generate_batch_summaries(all_stats, match_context)
            except Exception as e:
                logger.warning(
                    f"Batched summary generation failed, falling back to per-player: {e}"
                )

            # Distribute batch results and fill gaps with per-player calls
            for steam_id, player_data in players.items():
                name = steam_id_to_name.get(steam_id, "Unknown")
                summary = batch_results.get(name)

                if summary:
                    player_data["ai_summary"] = summary
                    logger.debug(f"Batch summary for {name} ({len(summary)} chars)")
                else:
                    # Fallback: individual call for this player
                    try:
                        ps = self._extract_player_stats_for_llm(player_data)
                        player_data["ai_summary"] = generate_match_summary(ps, match_context)
                        logger.debug(f"Individual summary fallback for {name}")
                    except Exception as e:
                        logger.warning(f"AI summary generation failed for player {steam_id}: {e}")
                        player_data["ai_summary"] = (
                            "**AI Summary Unavailable**\n\n"
                            "Unable to generate personalized insights at this time. "
                            "Check your ANTHROPIC_API_KEY configuration."
                        )

            logger.info(f"Generated AI summaries for {len(players)} players")

        except ImportError as e:
            logger.warning(f"LLM client not available, skipping AI summaries: {e}")
            # Add placeholder summaries
            for player_data in players.values():
                player_data["ai_summary"] = (
                    "**AI Coaching Not Configured**\n\n"
                    "To enable AI-powered coaching insights, install the Anthropic library:\n"
                    "```\npip install anthropic\n```\n"
                    "Then set your `ANTHROPIC_API_KEY` environment variable."
                )
        except Exception as e:
            logger.error(f"Unexpected error in AI summary generation: {e}")
            # Add error message to all players
            for player_data in players.values():
                player_data["ai_summary"] = (
                    f"**AI Coaching Error**\n\n"
                    f"An error occurred: {type(e).__name__}\n\n"
                    "Please check logs for details."
                )

    @staticmethod
    def _extract_player_stats_for_llm(player_data: dict) -> dict:
        """Extract a compact subset of player stats for LLM consumption.

        Reduces a full ~150-field player dict to ~20 key metrics that the LLM
        needs for generating coaching summaries.
        """
        stats = player_data.get("stats", {})
        rating = player_data.get("rating", {})
        advanced = player_data.get("advanced", {})
        entry = player_data.get("entry", {})
        trades = player_data.get("trades", {})
        duels = player_data.get("duels", {})
        return {
            "name": player_data.get("name", "Unknown"),
            "team": player_data.get("team", ""),
            "kills": stats.get("kills", 0),
            "deaths": stats.get("deaths", 0),
            "assists": stats.get("assists", 0),
            "adr": stats.get("adr", 0),
            "headshot_pct": stats.get("headshot_pct", 0),
            "hltv_rating": rating.get("hltv_rating", 0),
            "kast_percentage": rating.get("kast_percentage", 0),
            "aim_rating": rating.get("aim_rating", 0),
            "utility_rating": rating.get("utility_rating", 0),
            "impact_rating": rating.get("impact_rating", 0),
            "ttd_median_ms": advanced.get("ttd_median_ms", 0),
            "cp_median_error_deg": advanced.get("cp_median_error_deg", 0),
            "entry_kills": entry.get("entry_kills", 0),
            "entry_deaths": entry.get("entry_deaths", 0),
            "entry_success_pct": entry.get("entry_success_pct", 0),
            "trade_kill_success": trades.get("trade_kill_success", 0),
            "trade_kill_opportunities": trades.get("trade_kill_opportunities", 0),
            "clutch_wins": duels.get("clutch_wins", 0),
            "clutch_attempts": duels.get("clutch_attempts", 0),
        }

    def _calculate_match_averages(self, players: dict) -> dict:
        """Calculate match-wide statistics for comparison benchmarks."""
        if not players:
            return {}

        ttd_values = []
        cp_values = []
        adr_values = []
        hs_values = []
        entry_success_values = []
        trade_rates = []
        util_per_round_values = []

        for p in players.values():
            ttd = p.get("advanced", {}).get("ttd_median_ms", 0)
            if ttd > 0:
                ttd_values.append(ttd)

            cp = p.get("advanced", {}).get("cp_median_error_deg", 0)
            if cp > 0:
                cp_values.append(cp)

            adr = p.get("stats", {}).get("adr", 0)
            if adr > 0:
                adr_values.append(adr)

            hs = p.get("stats", {}).get("headshot_pct", 0)
            if hs > 0:
                hs_values.append(hs)

            entry_success = p.get("entry", {}).get("entry_success_pct", 0)
            entry_attempts = p.get("entry", {}).get("entry_attempts", 0)
            if entry_attempts >= 2:
                entry_success_values.append(entry_success)

            deaths = p.get("stats", {}).get("deaths", 0)
            traded = p.get("trades", {}).get("deaths_traded", 0) or p.get("duels", {}).get(
                "traded_deaths", 0
            )
            if deaths >= 5:
                trade_rates.append((traded / deaths) * 100)

            util = p.get("utility", {})
            total_util = sum(
                [
                    util.get("flashbangs_thrown", 0),
                    util.get("smokes_thrown", 0),
                    util.get("he_thrown", 0),
                    util.get("molotovs_thrown", 0),
                ]
            )
            rounds = p.get("rws", {}).get("rounds_played", 0) or 30
            if rounds > 0:
                util_per_round_values.append(total_util / rounds)

        return {
            "avg_ttd": sum(ttd_values) / len(ttd_values) if ttd_values else 350,
            "avg_cp": sum(cp_values) / len(cp_values) if cp_values else 8,
            "avg_adr": sum(adr_values) / len(adr_values) if adr_values else 75,
            "avg_hs_pct": sum(hs_values) / len(hs_values) if hs_values else 35,
            "avg_entry_success": (
                sum(entry_success_values) / len(entry_success_values)
                if entry_success_values
                else 50
            ),
            "avg_trade_rate": (sum(trade_rates) / len(trade_rates) if trade_rates else 40),
            "avg_util_per_round": (
                sum(util_per_round_values) / len(util_per_round_values)
                if util_per_round_values
                else 2.5
            ),
        }

    def _find_top_performers(self, players: dict) -> dict:
        """Find top performers in each stat category for comparative insights."""
        top = {}

        # Best trader
        best_trade_kills = 0
        for sid, p in players.items():
            tk = p.get("trades", {}).get("trade_kills", 0) or p.get("duels", {}).get(
                "trade_kills", 0
            )
            if tk > best_trade_kills:
                best_trade_kills = tk
                top["trade_kills"] = {"steam_id": sid, "name": p["name"], "value": tk}

        # Best entry
        best_entry_pct = 0
        for sid, p in players.items():
            entry_pct = p.get("entry", {}).get("entry_success_pct", 0)
            entry_attempts = p.get("entry", {}).get("entry_attempts", 0)
            if entry_attempts >= 3 and entry_pct > best_entry_pct:
                best_entry_pct = entry_pct
                top["entry"] = {"steam_id": sid, "name": p["name"], "value": entry_pct}

        # Best clutcher
        best_clutch_pct = 0
        for sid, p in players.items():
            clutch_pct = p.get("clutches", {}).get("clutch_success_pct", 0)
            clutch_attempts = p.get("clutches", {}).get("clutch_wins", 0) + p.get(
                "clutches", {}
            ).get("clutch_losses", 0)
            if clutch_attempts >= 2 and clutch_pct > best_clutch_pct:
                best_clutch_pct = clutch_pct
                top["clutch"] = {
                    "steam_id": sid,
                    "name": p["name"],
                    "value": clutch_pct,
                }

        # Best aim (lowest CP error)
        best_cp = 999
        for sid, p in players.items():
            cp = p.get("advanced", {}).get("cp_median_error_deg", 0)
            if 0 < cp < best_cp:
                best_cp = cp
                top["aim"] = {"steam_id": sid, "name": p["name"], "value": cp}

        return top

    def _get_ttd_rating(self, ttd_ms: float) -> str:
        """Get descriptive rating for TTD (Time to Damage)."""
        if ttd_ms < 200:
            return "Elite reaction time"
        elif ttd_ms < 300:
            return "Fast reactions"
        elif ttd_ms < 400:
            return "Average reactions"
        elif ttd_ms < 500:
            return "Slow reactions"
        else:
            return "Very slow - needs work"

    def _get_cp_rating(self, cp_deg: float) -> str:
        """Get descriptive rating for Crosshair Placement."""
        if cp_deg < 3:
            return "Pro-level placement"
        elif cp_deg < 6:
            return "Excellent placement"
        elif cp_deg < 10:
            return "Good placement"
        elif cp_deg < 15:
            return "Average placement"
        else:
            return "Needs improvement"

    def _detect_role_detailed(self, player: dict, match_stats: dict) -> tuple[str, str]:
        """
        Detect player role with confidence level using behavioral scoring.

        Uses the unified Role Scoring Engine approach:
        1. AWPer check (35%+ kills with sniper) - overrides everything
        2. Entry score (high opening attempts + first contact patterns)
        3. Support score (high utility + low first contact)
        4. Lurker score (high isolation + impact)
        5. Rifler (default high-frag role)

        Returns (role, confidence) where confidence is 'high', 'medium', or 'low'.
        """
        advanced = player.get("advanced", {})
        utility = player.get("utility", {})
        stats = player.get("stats", {})
        entry = player.get("entry", {})
        trades = player.get("trades", {})
        weapons = player.get("weapons", {})

        # Extract metrics
        kills = stats.get("kills", 0)
        deaths = stats.get("deaths", 0)
        rounds_played = stats.get("rounds_played", 1) or 1
        hs_pct = stats.get("headshot_pct", 0)
        adr = stats.get("adr", 0)

        entry_attempts = entry.get("entry_attempts", 0)
        entry_kills = entry.get("entry_kills", 0)
        entry_success_pct = entry.get("entry_success_pct", 0)

        flashes_thrown = utility.get("flashbangs_thrown", 0)
        smokes_thrown = utility.get("smokes_thrown", 0)
        he_thrown = utility.get("he_thrown", 0)
        molotovs_thrown = utility.get("molotovs_thrown", 0)
        flash_assists = utility.get("flash_assists", 0)
        effective_flashes = utility.get("effective_flashes", 0)

        untraded_deaths = (
            trades.get("untraded_deaths", 0) or advanced.get("untraded_deaths", 0) or 0
        )

        # AWP kills from weapon breakdown
        awp_kills = weapons.get("awp", 0) + weapons.get("AWP", 0)
        ssg_kills = weapons.get("ssg08", 0) + weapons.get("SSG08", 0)
        sniper_kills = awp_kills + ssg_kills

        # Score each role (0-100 scale)
        scores = {
            "entry": 0.0,
            "support": 0.0,
            "rifler": 0.0,
            "awper": 0.0,
            "lurker": 0.0,
        }

        # =====================================================================
        # STEP 1: AWPer Detection (Highest Priority)
        # If 35%+ of kills are with AWP/SSG08, this defines the player's role
        # =====================================================================
        if kills > 0:
            awp_kill_pct = (sniper_kills / kills) * 100
            if awp_kill_pct >= 35:
                scores["awper"] = 85.0 + min(awp_kill_pct - 35, 15)  # 85-100

        # =====================================================================
        # STEP 2: Entry Score
        # Based on: opening duel attempts (NOT just kills), aggression patterns
        # =====================================================================
        entry_score = 0.0

        # Opening duel ATTEMPTS (shows aggression, not just success)
        attempts_per_round = entry_attempts / rounds_played
        entry_score += min(attempts_per_round / 0.3, 1.0) * 40  # Up to 40 points

        # Entry success rate (reward winning, not just attempting)
        if entry_attempts >= 3:
            entry_score += min(entry_success_pct / 50, 1.0) * 25  # Up to 25 points

        # Opening kills as proxy for first contact
        opening_kill_rate = entry_kills / rounds_played * 100
        entry_score += min(opening_kill_rate / 25, 1.0) * 20  # Up to 20 points

        # Bonus: High effective flashes (utility-supported entries)
        if effective_flashes > 3:
            entry_score += 10

        scores["entry"] = min(entry_score, 100)

        # =====================================================================
        # STEP 3: Support Score
        # Based on: utility effectiveness, flash assists, passive positioning
        # =====================================================================
        support_score = 0.0

        # Effective flashes (shows intentional team support)
        support_score += min(effective_flashes / 8, 1.0) * 25  # Up to 25 points

        # Flash assists (direct team contribution)
        support_score += min(flash_assists / 4, 1.0) * 25  # Up to 25 points

        # Total utility usage
        total_utility = flashes_thrown + smokes_thrown + he_thrown + molotovs_thrown
        utility_per_round = total_utility / rounds_played
        support_score += min(utility_per_round / 3, 1.0) * 20  # Up to 20 points

        # LOW entry attempts = passive player (support indicator)
        if attempts_per_round < 0.15:
            support_score += 15

        # Trade attempt rate (being in position to support)
        trade_opps = trades.get("trade_kill_opportunities", 0)
        trade_attempts = trades.get("trade_kill_attempts", 0)
        if trade_opps > 0:
            trade_attempt_rate = trade_attempts / trade_opps
            support_score += trade_attempt_rate * 15  # Up to 15 points

        scores["support"] = min(support_score, 100)

        # =====================================================================
        # STEP 4: Lurker Score
        # Based on: high isolation (untraded deaths), but WITH IMPACT
        # =====================================================================
        lurker_score = 0.0

        # Isolation indicator: untraded deaths
        if deaths > 3:
            isolation_rate = untraded_deaths / deaths
            if isolation_rate > 0.5:
                lurker_score += (isolation_rate - 0.5) * 2 * 35  # Up to 35 points

        # LOW entry attempts = not taking first contact
        if entry_attempts <= 2 and rounds_played >= 10:
            lurker_score += 20

        # CRITICAL: Must have IMPACT to be a lurker, not a feeder
        kpr = kills / rounds_played
        hltv_rating = stats.get("hltv_rating", 0) or player.get("rating", {}).get("hltv_rating", 0)
        has_impact = kpr >= 0.5 or hltv_rating >= 0.9

        if has_impact:
            lurker_score += 25  # Impact bonus
        elif lurker_score > 30:
            lurker_score *= 0.3  # Penalize if no impact (feeding, not lurking)

        # Multi-kills (lurkers often catch rotations)
        multikills = (
            stats.get("2k", 0) + stats.get("3k", 0) + stats.get("4k", 0) + stats.get("5k", 0)
        )
        if multikills >= 3:
            lurker_score += 15

        scores["lurker"] = min(lurker_score, 100)

        # =====================================================================
        # STEP 5: Rifler Score (Default High-Frag Role)
        # Based on: high kills, high HS%, consistent damage
        # =====================================================================
        rifler_score = 0.0

        # Kill rate
        rifler_score += min(kpr / 0.8, 1.0) * 30  # Up to 30 points for 0.8+ KPR

        # ADR (consistent damage)
        rifler_score += min(adr / 85, 1.0) * 25  # Up to 25 points

        # Headshot percentage
        rifler_score += min(hs_pct / 50, 1.0) * 20  # Up to 20 points

        # Multi-kills (weighted: 2k=1, 3k=2, 4k=3, 5k=4)
        mk_score = (
            stats.get("2k", 0)
            + stats.get("3k", 0) * 2
            + stats.get("4k", 0) * 3
            + stats.get("5k", 0) * 4
        )
        rifler_score += min(mk_score / 8, 1.0) * 15  # Up to 15 points

        # HLTV rating bonus
        if hltv_rating >= 1.15:
            rifler_score += 10

        scores["rifler"] = min(rifler_score, 100)

        # =====================================================================
        # Determine primary role
        # =====================================================================
        best_role = max(scores, key=scores.get)
        best_score = scores[best_role]

        # Confidence based on score
        if best_score >= 70:
            confidence = "high"
        elif best_score >= 40:
            confidence = "medium"
        else:
            confidence = "low"

        # Check for tie (within 10% of each other) - default to flex
        second_best_score = sorted(scores.values(), reverse=True)[1]
        if best_score > 0 and best_score < 20:
            best_role = "flex"
            confidence = "low"
        elif best_score > 0 and (best_score - second_best_score) / best_score < 0.10:
            best_role = "flex"
            confidence = "medium"

        return best_role, confidence

    def _determine_player_identity(
        self, player: dict, match_stats: dict, top_performers: dict
    ) -> dict:
        """
        Determine player identity/archetype like Leetify's system.
        Returns identity name and top stats.
        """
        stats = player.get("stats", {})
        advanced = player.get("advanced", {})
        rating = player.get("rating", {})
        entry = player.get("entry", {})
        utility = player.get("utility", {})
        clutches = player.get("clutches", {})
        trades = player.get("trades", {})

        # Collect notable stats
        notable_stats = []

        # ADR
        adr = stats.get("adr", 0)
        if adr >= 85:
            notable_stats.append(("damage_dealer", adr, f"{adr:.0f} ADR"))

        # Entry kills
        entry_kills = entry.get("entry_kills", 0)
        if entry_kills >= 5:
            notable_stats.append(("entry_fragger", entry_kills, f"{entry_kills} Opening Kills"))

        # Clutches
        clutch_wins = clutches.get("clutch_wins", 0)
        if clutch_wins >= 2:
            notable_stats.append(("clutch_player", clutch_wins, f"{clutch_wins} Clutches Won"))

        # Trade kills
        trade_kills = trades.get("trade_kills", 0) or player.get("duels", {}).get("trade_kills", 0)
        if trade_kills >= 5:
            notable_stats.append(("team_player", trade_kills, f"{trade_kills} Trade Kills"))

        # Flash assists
        flash_assists = utility.get("flash_assists", 0)
        if flash_assists >= 3:
            notable_stats.append(
                ("support_master", flash_assists, f"{flash_assists} Flash Assists")
            )

        # HS%
        hs_pct = stats.get("headshot_pct", 0)
        if hs_pct >= 50:
            notable_stats.append(("headshot_machine", hs_pct, f"{hs_pct:.0f}% HS"))

        # Multi-kills
        multikills = stats.get("3k", 0) + stats.get("4k", 0) + stats.get("5k", 0)
        if multikills >= 3:
            notable_stats.append(("round_winner", multikills, f"{multikills} Multi-kills"))

        # KAST
        kast = rating.get("kast_percentage", 0)
        if kast >= 80:
            notable_stats.append(("consistent", kast, f"{kast:.0f}% KAST"))

        # TTD (lower is better)
        ttd = advanced.get("ttd_median_ms", 0)
        if 0 < ttd < 250:
            notable_stats.append(("fast_reactions", 1000 - ttd, f"{ttd:.0f}ms TTD"))

        # CP (lower is better)
        cp = advanced.get("cp_median_error_deg", 0)
        if 0 < cp < 5:
            notable_stats.append(("precise_aim", 100 - cp, f"{cp:.1f}Â° CP"))

        # Sort by value (highest first) and pick top identity
        notable_stats.sort(key=lambda x: x[1], reverse=True)

        # Identity mapping
        identity_names = {
            "damage_dealer": "The Damage Dealer",
            "entry_fragger": "The Entry Fragger",
            "clutch_player": "The Clutch Master",
            "team_player": "The Team Player",
            "support_master": "The Support",
            "headshot_machine": "The Headshot Machine",
            "round_winner": "The Round Winner",
            "consistent": "The Consistent One",
            "fast_reactions": "The Reactor",
            "precise_aim": "The Precise",
        }

        if notable_stats:
            top_identity = notable_stats[0][0]
            return {
                "name": identity_names.get(top_identity, "The Player"),
                "top_stats": [{"label": s[2], "category": s[0]} for s in notable_stats[:5]],
            }
        else:
            return {
                "name": "The Contributor",
                "top_stats": [],
            }

    def _detect_role(self, player: dict) -> str:
        """
        Detect player role from stats using behavioral scoring.

        Priority:
        1. AWPer (35%+ kills with sniper)
        2. Entry (high opening attempts)
        3. Support (high utility + flash assists)
        4. Rifler (high kills + HS%)
        5. Flex (default)
        """
        utility = player.get("utility", {})
        stats = player.get("stats", {})
        entry = player.get("entry", {})
        weapons = player.get("weapons", {})

        kills = stats.get("kills", 0)
        rounds_played = stats.get("rounds_played", 1) or 1

        # AWPer check first (highest priority)
        awp_kills = weapons.get("awp", 0) + weapons.get("AWP", 0)
        ssg_kills = weapons.get("ssg08", 0) + weapons.get("SSG08", 0)
        if kills > 0 and (awp_kills + ssg_kills) / kills >= 0.35:
            return "awper"

        # Entry: high opening attempts (behavior, not just kills)
        entry_attempts = entry.get("entry_attempts", 0)
        if entry_attempts >= 4 or (entry_attempts >= 3 and entry.get("entry_success_pct", 0) >= 50):
            return "entry"

        # Support: high utility + flash assists (team contribution)
        flash_assists = utility.get("flash_assists", 0)
        effective_flashes = utility.get("effective_flashes", 0)
        total_utility = (
            utility.get("flashbangs_thrown", 0)
            + utility.get("smokes_thrown", 0)
            + utility.get("he_thrown", 0)
            + utility.get("molotovs_thrown", 0)
        )
        if (
            flash_assists >= 3
            or effective_flashes >= 5
            or (total_utility / rounds_played >= 2.5 and entry_attempts <= 2)
        ):
            return "support"

        # Rifler: high kills + consistent fragging
        if kills >= 15 and stats.get("headshot_pct", 0) >= 40:
            return "rifler"

        return "flex"

    def _get_tactical_summary(self, demo_data, analysis) -> dict:
        """Get tactical analysis summary."""
        try:
            from opensight.analysis.tactical_service import TacticalAnalysisService

            service = TacticalAnalysisService(demo_data)
            summary = service.analyze()

            # Serialize team analysis to dict
            team1_dict = {
                "team_name": summary.team1_analysis.team_name,
                "team_side": summary.team1_analysis.team_side,
                "key_insights": summary.team1_analysis.key_insights,
                "recommendations": summary.team1_analysis.recommendations,
                "strengths": summary.team1_analysis.strengths,
                "weaknesses": summary.team1_analysis.weaknesses,
                "star_player": summary.team1_analysis.star_player,
                "star_player_role": summary.team1_analysis.star_player_role,
                "coordination_score": summary.team1_analysis.coordination_score,
            }

            team2_dict = {
                "team_name": summary.team2_analysis.team_name,
                "team_side": summary.team2_analysis.team_side,
                "key_insights": summary.team2_analysis.key_insights,
                "recommendations": summary.team2_analysis.recommendations,
                "strengths": summary.team2_analysis.strengths,
                "weaknesses": summary.team2_analysis.weaknesses,
                "star_player": summary.team2_analysis.star_player,
                "star_player_role": summary.team2_analysis.star_player_role,
                "coordination_score": summary.team2_analysis.coordination_score,
            }

            return {
                "key_insights": summary.key_insights,
                "t_stats": summary.t_stats,
                "ct_stats": summary.ct_stats,
                "t_executes": summary.t_executes,
                "buy_patterns": summary.buy_patterns,
                "t_strengths": summary.t_strengths,
                "t_weaknesses": summary.t_weaknesses,
                "ct_strengths": summary.ct_strengths,
                "ct_weaknesses": summary.ct_weaknesses,
                "team_recommendations": summary.team_recommendations,
                "practice_drills": summary.practice_drills,
                "team1_analysis": team1_dict,
                "team2_analysis": team2_dict,
            }
        except Exception as e:
            logger.warning(f"Tactical analysis failed: {e}")
            return {
                "key_insights": ["Demo analysis complete"],
                "team_recommendations": ["Review round-by-round for specific improvements"],
            }


# Convenience function
def analyze_demo(demo_path: Path, *, force: bool = False) -> dict:
    """
    Convenience function to analyze a demo file.

    Uses cache by default; pass force=True to re-analyze.

    Args:
        demo_path: Path to demo file
        force: Skip cache and re-analyze

    Returns:
        Complete analysis results dict
    """
    orchestrator = DemoOrchestrator()
    return orchestrator.analyze(demo_path, force=force)

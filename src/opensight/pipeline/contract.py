"""
OpenSight Output Contract — the single source of truth.

Defines the exact JSON structure that DemoOrchestrator.analyze() returns.
Every field name, nesting level, and type is locked here.

Rules:
  1. The orchestrator MUST produce output matching PLAYER_CONTRACT.
  2. API routes MUST read fields using the paths defined here.
  3. The frontend MUST access fields using the paths defined here.
  4. Any new field goes here FIRST, then gets wired through all layers.

Validated by: tests/test_contract.py (runtime schema check)
"""

from __future__ import annotations

# ─── Top-level result shape ───────────────────────────────────────────
RESULT_CONTRACT: dict = {
    "demo_info": {
        "map": str,
        "rounds": int,
        "duration_minutes": (int, float),
        "score": str,  # "13 - 7"
        "score_ct": int,
        "score_t": int,
        "total_kills": int,
        "team1_name": str,
        "team2_name": str,
    },
    "players": dict,  # keyed by steam_id string -> PLAYER_CONTRACT
    "mvp": {
        "name": str,
        "rating": (int, float),
    },
    "round_timeline": list,
    "kill_matrix": list,
    "heatmap_data": dict,
    "coaching": (dict, list),
    "tactical": dict,
    "synergy": dict,
    "timeline_graph": dict,
    "analyzed_at": str,
}

# ─── Per-player shape (keyed by steam_id string) ─────────────────────
PLAYER_CONTRACT: dict = {
    "steam_id": str,
    "name": str,
    "team": str,
    # Top-level convenience fields (frontend flat-access fallbacks)
    "rounds_played": int,
    "total_damage": int,
    # ── stats: basic scoreboard numbers ──
    "stats": {
        "kills": int,
        "deaths": int,
        "assists": int,
        "rounds_played": int,
        "adr": (int, float),
        "headshot_pct": (int, float),
        "kd_ratio": (int, float),
        "total_damage": int,
        "2k": int,
        "3k": int,
        "4k": int,
        "5k": int,
    },
    # ── rating: composite scores ──
    "rating": {
        "hltv_rating": (int, float),
        "kast_percentage": (int, float),
        "aim_rating": (int, float),
        "utility_rating": (int, float),
        "impact_rating": (int, float),
    },
    # ── advanced: TTD, CP, opening duel summary ──
    "advanced": {
        "ttd_median_ms": (int, float),
        "ttd_mean_ms": (int, float),
        "ttd_95th_ms": (int, float),
        "cp_median_error_deg": (int, float),
        "cp_mean_error_deg": (int, float),
        "prefire_kills": int,
        "opening_kills": int,
        "opening_deaths": int,
    },
    # ── utility: grenade usage and effectiveness ──
    "utility": {
        "flashbangs_thrown": int,
        "smokes_thrown": int,
        "he_thrown": int,
        "molotovs_thrown": int,
        "flash_assists": int,
        "enemies_flashed": int,
        "teammates_flashed": int,
        "he_damage": int,
        "molotov_damage": int,
        "enemies_flashed_per_round": (int, float),
        "friends_flashed_per_round": (int, float),
        "avg_blind_time": (int, float),
        "avg_he_damage": (int, float),
        "flash_effectiveness_pct": (int, float),
        "flash_assist_pct": (int, float),
        "he_team_damage": int,
        "unused_utility_value": int,
        "utility_quality_rating": (int, float),
        "utility_quantity_rating": (int, float),
        "effective_flashes": int,
        "total_blind_time": (int, float),
        "times_blinded": int,
        "total_time_blinded": (int, float),
        "avg_time_blinded": (int, float),
    },
    # ── aim_stats: accuracy and shooting mechanics ──
    "aim_stats": {
        "shots_fired": int,
        "shots_hit": int,
        "accuracy_all": (int, float),
        "headshot_hits": int,
        "head_accuracy": (int, float),
        "spray_shots_fired": int,
        "spray_shots_hit": int,
        "spray_accuracy": (int, float),
        "shots_stationary": int,
        "shots_with_velocity": int,
        "counter_strafe_pct": (int, float),
        "time_to_damage_ms": (int, float),
        "crosshair_placement_deg": (int, float),
    },
    # ── duels: quick-reference trade/clutch/opening summary ──
    "duels": {
        "trade_kills": int,
        "traded_deaths": int,
        "clutch_wins": int,
        "clutch_attempts": int,
        "opening_kills": int,
        "opening_deaths": int,
        "opening_wins": int,  # alias for opening_kills (frontend compat)
        "opening_losses": int,  # alias for opening_deaths (frontend compat)
        "opening_win_rate": (int, float),
    },
    # ── spray_transfers ──
    "spray_transfers": {
        "double_sprays": int,
        "triple_sprays": int,
        "quad_sprays": int,
        "ace_sprays": int,
        "total_sprays": int,
        "total_spray_kills": int,
        "avg_spray_time_ms": (int, float),
        "avg_kills_per_spray": (int, float),
    },
    # ── entry: opening duel detail (FACEIT style) ──
    "entry": {
        "entry_attempts": int,
        "entry_kills": int,
        "entry_deaths": int,
        "entry_diff": int,
        "entry_attempts_pct": (int, float),
        "entry_success_pct": (int, float),
    },
    # ── trades: Leetify-style trade detail ──
    "trades": {
        "trade_kill_opportunities": int,
        "trade_kill_attempts": int,
        "trade_kill_attempts_pct": (int, float),
        "trade_kill_success": int,
        "trade_kill_success_pct": (int, float),
        "traded_death_opportunities": int,
        "traded_death_attempts": int,
        "traded_death_attempts_pct": (int, float),
        "traded_death_success": int,
        "traded_death_success_pct": (int, float),
        "trade_kills": int,
        "deaths_traded": int,
        "traded_entry_kills": int,
        "traded_entry_deaths": int,
        "untraded_deaths": int,
    },
    # ── clutches: 1vX detail ──
    "clutches": {
        "clutch_wins": int,
        "clutch_losses": int,
        "clutch_success_pct": (int, float),
        "total_situations": int,
        "v1_wins": int,
        "v1_attempts": int,
        "v2_wins": int,
        "v2_attempts": int,
        "v3_wins": int,
        "v3_attempts": int,
        "v4_wins": int,
        "v4_attempts": int,
        "v5_wins": int,
        "v5_attempts": int,
    },
    # ── rws: round win shares ──
    "rws": {
        "avg_rws": (int, float),
        "total_rws": (int, float),
        "rounds_won": int,
        "rounds_played": int,
        "damage_per_round": (int, float),
        "objective_completions": int,
    },
    # ── economy: per-player spending efficiency ──
    "economy": {
        "avg_equipment_value": (int, float),
        "eco_rounds": int,
        "force_rounds": int,
        "full_buy_rounds": int,
        "damage_per_dollar": (int, float),
        "kills_per_dollar": (int, float),
    },
    # ── discipline: re-peek and positioning discipline ──
    "discipline": {
        "discipline_rating": (int, float),
        "greedy_repeeks": int,
    },
    # ── side_stats: CT/T side performance breakdown ──
    "side_stats": {
        "ct": {
            "kills": int,
            "deaths": int,
            "assists": int,
            "damage": int,
            "rounds_played": int,
            "kd_ratio": (int, float),
            "adr": (int, float),
        },
        "t": {
            "kills": int,
            "deaths": int,
            "assists": int,
            "damage": int,
            "rounds_played": int,
            "kd_ratio": (int, float),
            "adr": (int, float),
        },
    },
    # ── mistakes: team damage, team kills, flashes ──
    "mistakes": {
        "team_damage": int,
        "team_kills": int,
        "teammates_flashed": int,
        "suicides": int,
        "total_mistakes": int,
    },
    # ── lurk: lurk behavior analysis ──
    "lurk": {
        "kills": int,
        "deaths": int,
        "rounds_lurking": int,
    },
}


def validate_player(player_data: dict, errors: list[str] | None = None) -> list[str]:
    """Validate a player dict against the contract. Returns list of errors."""
    if errors is None:
        errors = []
    _validate_dict(player_data, PLAYER_CONTRACT, "player", errors)
    return errors


def validate_result(result: dict) -> list[str]:
    """Validate a full orchestrator result dict. Returns list of errors."""
    errors: list[str] = []
    # Check top-level keys
    for key in RESULT_CONTRACT:
        if key not in result:
            errors.append(f"MISSING top-level key: {key}")

    # Validate each player
    players = result.get("players", {})
    if isinstance(players, dict):
        for sid, pdata in players.items():
            _validate_dict(pdata, PLAYER_CONTRACT, f"players[{sid}]", errors)
    else:
        errors.append(f"players must be dict, got {type(players).__name__}")

    return errors


def _validate_dict(data: dict, contract: dict, path: str, errors: list[str]) -> None:
    """Recursively validate data against contract schema."""
    if not isinstance(data, dict):
        errors.append(f"{path}: expected dict, got {type(data).__name__}")
        return

    for key, expected_type in contract.items():
        full_path = f"{path}.{key}"
        if key not in data:
            errors.append(f"MISSING {full_path}")
            continue

        value = data[key]

        # If expected_type is a dict, recurse
        if isinstance(expected_type, dict):
            _validate_dict(value, expected_type, full_path, errors)
        # If expected_type is a tuple of types, check isinstance
        elif isinstance(expected_type, tuple):
            if value is not None and not isinstance(value, expected_type):
                errors.append(
                    f"TYPE {full_path}: expected {expected_type}, "
                    f"got {type(value).__name__} = {value!r}"
                )
        # Single type check
        elif expected_type is not None:
            if value is not None and not isinstance(value, expected_type):
                errors.append(
                    f"TYPE {full_path}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__} = {value!r}"
                )

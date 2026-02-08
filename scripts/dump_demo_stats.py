"""Dump all player stats from a CS2 demo file to JSON.

Usage:
    python scripts/dump_demo_stats.py path/to/demo.dem

Outputs:
    - Console: Player summary (Name, K/D/A, HLTV Rating)
    - File: scripts/demo_dump.json with ALL available stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def safe_get(obj, attr, default=None):
    """Safely get an attribute, returning default on any error."""
    try:
        val = getattr(obj, attr, default)
        return val
    except Exception as e:
        return f"ERROR: {e}"


def safe_prop(func, default=None):
    """Safely call a property/function, returning default on error."""
    try:
        val = func()
        return val
    except Exception as e:
        return f"ERROR: {e}"


def build_clutch_list(clutch_stats):
    """Build individual clutch event list from ClutchStats."""
    if clutch_stats is None:
        return []
    clutches = getattr(clutch_stats, "clutches", [])
    result = []
    for c in clutches:
        try:
            result.append({
                "round_num": getattr(c, "round_number", None),
                "disadvantage": getattr(c, "type", None),
                "kills": getattr(c, "enemies_killed", 0),
                "won": getattr(c, "outcome", "") == "WON",
            })
        except Exception as e:
            result.append({"error": str(e)})
    return result


def build_player_dump(p, orchestrator_player_dict):
    """Build comprehensive stats dict for one player.

    Args:
        p: PlayerMatchStats dataclass instance
        orchestrator_player_dict: The orchestrator's serialized dict for this player
    """
    opd = orchestrator_player_dict  # shorthand

    player = {}

    # === Identity ===
    player["name"] = p.name
    player["steam_id"] = str(p.steam_id)
    player["team"] = p.team

    # === Core Stats ===
    player["kills"] = p.kills
    player["deaths"] = p.deaths
    player["assists"] = p.assists
    player["headshots"] = p.headshots
    player["adr"] = round(p.adr, 1) if p.adr is not None else None
    player["kast_percentage"] = (
        round(p.kast_percentage, 1) if p.kast_percentage is not None else None
    )
    player["headshot_pct"] = (
        round(p.headshot_percentage, 1) if p.headshot_percentage is not None else None
    )
    player["kd_ratio"] = round(p.kd_ratio, 2) if p.kd_ratio is not None else None
    player["kd_diff"] = safe_prop(lambda: p.kd_diff)

    # === Ratings ===
    player["hltv_rating"] = safe_prop(lambda: round(p.hltv_rating, 2))
    player["impact_rating"] = safe_prop(lambda: round(p.impact_rating, 2))
    player["impact_plus_minus"] = safe_prop(lambda: round(p.impact_plus_minus, 2))
    player["aim_rating"] = safe_prop(lambda: round(p.aim_rating, 1))
    player["utility_rating"] = safe_prop(lambda: round(p.utility_rating, 1))
    player["utility_quality_rating"] = safe_prop(
        lambda: round(p.utility_quality_rating, 1)
    )
    player["utility_quantity_rating"] = safe_prop(
        lambda: round(p.utility_quantity_rating, 1)
    )

    # === Rounds ===
    player["rounds_played"] = p.rounds_played
    player["rounds_survived"] = p.rounds_survived
    player["rounds_survived_pct"] = safe_prop(lambda: round(p.survival_rate, 1))
    player["kills_per_round"] = safe_prop(lambda: p.kills_per_round)
    player["deaths_per_round"] = safe_prop(lambda: p.deaths_per_round)

    # === Multi-kills ===
    mk = p.multi_kills
    player["multi_kills"] = {
        "2k": mk.rounds_with_2k if mk else 0,
        "3k": mk.rounds_with_3k if mk else 0,
        "4k": mk.rounds_with_4k if mk else 0,
        "5k": mk.rounds_with_5k if mk else 0,
    }

    # === Trades ===
    trades = p.trades
    if trades is not None:
        player["trades"] = {
            "trade_kill_opportunities": trades.trade_kill_opportunities,
            "trade_kill_attempts": trades.trade_kill_attempts,
            "trade_kill_success": trades.trade_kill_success,
            "trade_kill_success_pct": safe_prop(
                lambda: round(trades.trade_kill_success_pct, 1)
            ),
            "traded_death_opportunities": trades.traded_death_opportunities,
            "traded_death_attempts": trades.traded_death_attempts,
            "traded_death_success": trades.traded_death_success,
            "traded_death_success_pct": safe_prop(
                lambda: round(trades.traded_death_success_pct, 1)
            ),
            "kills_traded": trades.kills_traded,
            "deaths_traded": trades.deaths_traded,
        }
    else:
        player["trades"] = None

    # === Entry / Opening Duels ===
    od = p.opening_duels
    if od is not None:
        player["entry"] = {
            "attempts": od.attempts,
            "wins": od.wins,
            "losses": od.losses,
            "win_rate": safe_prop(lambda: round(od.win_rate, 1)),
            "t_side_entries": od.t_side_entries,
            "ct_side_entries": od.ct_side_entries,
            "dry_peek_rate": safe_prop(lambda: round(od.dry_peek_rate, 1)),
        }
    else:
        player["entry"] = None

    # === Clutches ===
    cl = p.clutches
    if cl is not None:
        player["clutches"] = {
            "total_situations": cl.total_situations,
            "total_wins": cl.total_wins,
            "win_rate": safe_prop(lambda: round(cl.win_rate, 1)),
            "v1_attempts": cl.v1_attempts,
            "v1_wins": cl.v1_wins,
            "v2_attempts": cl.v2_attempts,
            "v2_wins": cl.v2_wins,
            "v3_attempts": cl.v3_attempts,
            "v3_wins": cl.v3_wins,
            "v4_attempts": cl.v4_attempts,
            "v4_wins": cl.v4_wins,
            "v5_attempts": cl.v5_attempts,
            "v5_wins": cl.v5_wins,
            "individual_clutches": build_clutch_list(cl),
        }
    else:
        player["clutches"] = None

    # === Utility ===
    util = p.utility
    if util is not None:
        player["utility"] = {
            "flashbangs_thrown": util.flashbangs_thrown,
            "smokes_thrown": util.smokes_thrown,
            "he_thrown": util.he_thrown,
            "molotovs_thrown": util.molotovs_thrown,
            "total_utility": safe_prop(lambda: util.total_utility),
            "enemies_flashed": util.enemies_flashed,
            "teammates_flashed": util.teammates_flashed,
            "flash_assists": util.flash_assists,
            "effective_flashes": util.effective_flashes,
            "he_damage": util.he_damage,
            "molotov_damage": util.molotov_damage,
            "he_team_damage": util.he_team_damage,
            "unused_utility_value": util.unused_utility_value,
            "utility_quality_rating": safe_prop(
                lambda: round(util.utility_quality_rating, 1)
            ),
            "utility_quantity_rating": safe_prop(
                lambda: round(util.utility_quantity_rating, 1)
            ),
            "flash_effectiveness_pct": safe_prop(
                lambda: round(util.flash_effectiveness_pct, 1)
            ),
            "avg_blind_time": safe_prop(lambda: round(util.avg_blind_time, 2)),
            "total_blind_time": round(util.total_blind_time, 2),
        }
    else:
        player["utility"] = None

    # === Advanced (TTD, CP) ===
    player["advanced"] = {
        "ttd_median_ms": (
            round(p.ttd_median_ms, 1) if p.ttd_median_ms is not None else None
        ),
        "ttd_mean_ms": (
            round(p.ttd_mean_ms, 1) if p.ttd_mean_ms is not None else None
        ),
        "cp_median_error_deg": (
            round(p.cp_median_error_deg, 1)
            if p.cp_median_error_deg is not None
            else None
        ),
        "cp_mean_error_deg": (
            round(p.cp_mean_error_deg, 1)
            if p.cp_mean_error_deg is not None
            else None
        ),
        "reaction_time_median_ms": safe_prop(
            lambda: (
                round(p.reaction_time_median_ms, 1)
                if p.reaction_time_median_ms is not None
                else None
            )
        ),
        "prefire_count": p.prefire_count,
        "prefire_percentage": safe_prop(lambda: round(p.prefire_percentage, 1)),
    }

    # === Aim Stats ===
    player["aim_stats"] = {
        "shots_fired": p.shots_fired,
        "shots_hit": p.shots_hit,
        "accuracy_all": safe_prop(lambda: round(p.accuracy, 1)),
        "headshot_hits": p.headshot_hits,
        "head_hit_rate": safe_prop(lambda: round(p.head_hit_rate, 1)),
        "spray_shots_fired": p.spray_shots_fired,
        "spray_shots_hit": p.spray_shots_hit,
        "spray_accuracy": safe_prop(lambda: round(p.spray_accuracy, 1)),
        "shots_stationary": p.shots_stationary,
        "shots_with_velocity": p.shots_with_velocity,
        "counter_strafe_pct": safe_prop(lambda: round(p.counter_strafe_pct, 1)),
    }

    # === Damage ===
    player["damage"] = {
        "total_damage": p.total_damage,
        "shots_fired": p.shots_fired,
        "damage_per_shot": (
            round(p.total_damage / p.shots_fired, 1) if p.shots_fired > 0 else 0
        ),
    }

    # === Economy ===
    player["economy"] = {
        "avg_equipment_value": round(p.avg_equipment_value, 0),
        "eco_rounds": p.eco_rounds,
        "force_rounds": p.force_rounds,
        "full_buy_rounds": p.full_buy_rounds,
        "damage_per_dollar": round(p.damage_per_dollar, 4),
        "kills_per_dollar": round(p.kills_per_dollar, 4),
    }

    # === CT / T Side Stats ===
    ct = p.ct_stats
    if ct is not None:
        player["ct_stats"] = {
            "kills": ct.kills,
            "deaths": ct.deaths,
            "assists": ct.assists,
            "damage": ct.damage,
            "rounds_played": ct.rounds_played,
            "kd_ratio": safe_prop(lambda: ct.kd_ratio),
            "adr": safe_prop(lambda: ct.adr),
        }
    else:
        player["ct_stats"] = None

    t = p.t_stats
    if t is not None:
        player["t_stats"] = {
            "kills": t.kills,
            "deaths": t.deaths,
            "assists": t.assists,
            "damage": t.damage,
            "rounds_played": t.rounds_played,
            "kd_ratio": safe_prop(lambda: t.kd_ratio),
            "adr": safe_prop(lambda: t.adr),
        }
    else:
        player["t_stats"] = None

    # === Spray Transfers ===
    st = p.spray_transfers
    if st is not None:
        player["spray_transfers"] = {
            "double_sprays": st.double_sprays,
            "triple_sprays": st.triple_sprays,
            "quad_sprays": st.quad_sprays,
            "ace_sprays": st.ace_sprays,
            "total_sprays": safe_prop(lambda: st.total_sprays),
            "total_spray_kills": st.total_spray_kills,
        }
    else:
        player["spray_transfers"] = None

    # === Discipline ===
    player["discipline"] = {
        "greedy_repeeks": p.greedy_repeeks,
        "discipline_rating": round(p.discipline_rating, 1),
        "untraded_deaths": p.untraded_deaths,
    }

    # === RWS ===
    player["rws"] = {
        "avg_rws": round(p.rws, 2),
        "damage_in_won_rounds": p.damage_in_won_rounds,
        "rounds_won": p.rounds_won,
    }

    # === Weapon Kills ===
    player["weapon_kills"] = p.weapon_kills or {}

    return player


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/dump_demo_stats.py <path_to_demo.dem>")
        sys.exit(1)

    demo_path = Path(sys.argv[1])
    if not demo_path.exists():
        print(f"Error: Demo file not found: {demo_path}")
        sys.exit(1)

    print(f"Analyzing: {demo_path.name}")
    print("This may take a minute...")
    print()

    # Run the full orchestrator pipeline (same as web UI)
    import os

    os.environ.setdefault("PYTHONPATH", "src")

    from opensight.analysis.analytics import DemoAnalyzer
    from opensight.core.parser import DemoParser
    from opensight.pipeline.orchestrator import DemoOrchestrator

    # Parse and analyze
    orchestrator = DemoOrchestrator(use_cache=False)
    result = orchestrator.analyze(demo_path, force=True)

    # Also get the raw analysis for direct dataclass access
    parser = DemoParser(demo_path)
    demo_data = parser.parse()
    analyzer = DemoAnalyzer(demo_data)
    analysis = analyzer.analyze()

    # === Build output ===
    output = {}

    # Match-level info
    demo_info = result.get("demo_info", {})
    output["match_info"] = {
        "map_name": demo_info.get("map", "unknown"),
        "total_rounds": demo_info.get("rounds", 0),
        "score": demo_info.get("score", "0 - 0"),
        "score_ct": demo_info.get("score_ct", 0),
        "score_t": demo_info.get("score_t", 0),
        "team1_name": demo_info.get("team1_name", "Counter-Terrorists"),
        "team2_name": demo_info.get("team2_name", "Terrorists"),
        "total_kills": demo_info.get("total_kills", 0),
        "duration_minutes": demo_info.get("duration_minutes", None),
        "rounds_in_timeline": len(result.get("round_timeline", [])),
        "knife_round_detected": any(
            r.get("is_knife_round", False)
            for r in result.get("round_timeline", [])
        ),
    }

    # MVP
    output["mvp"] = result.get("mvp", None)

    # Players
    output["players"] = {}
    for sid, p in analysis.players.items():
        sid_str = str(sid)
        orch_player = result.get("players", {}).get(sid_str, {})
        try:
            output["players"][sid_str] = build_player_dump(p, orch_player)
        except Exception as e:
            output["players"][sid_str] = {
                "name": getattr(p, "name", "unknown"),
                "error": f"Failed to dump: {e}",
            }

    # Sort players by HLTV rating for display
    sorted_players = sorted(
        output["players"].values(),
        key=lambda x: x.get("hltv_rating", 0) if not isinstance(x.get("hltv_rating"), str) else 0,
        reverse=True,
    )

    # === Console Summary ===
    print("=" * 70)
    print(
        f"  Map: {output['match_info']['map_name']}  |  "
        f"Score: {output['match_info']['score']}  |  "
        f"Rounds: {output['match_info']['total_rounds']}"
    )
    print("=" * 70)
    print(f"  {'Player':<20} {'Team':<5} {'K':>3} {'D':>3} {'A':>3} "
          f"{'ADR':>6} {'KAST':>6} {'Rating':>7}")
    print("-" * 70)
    for pl in sorted_players:
        name = pl.get("name", "?")[:20]
        team = pl.get("team", "?")[:4]
        k = pl.get("kills", 0)
        d = pl.get("deaths", 0)
        a = pl.get("assists", 0)
        adr = pl.get("adr", 0)
        kast = pl.get("kast_percentage", 0)
        rating = pl.get("hltv_rating", 0)
        if isinstance(rating, str):
            rating = 0
        if isinstance(adr, str):
            adr = 0
        if isinstance(kast, str):
            kast = 0
        print(
            f"  {name:<20} {team:<5} {k:>3} {d:>3} {a:>3} "
            f"{adr:>6.1f} {kast:>5.1f}% {rating:>7.2f}"
        )
    print("=" * 70)

    # === Write JSON ===
    out_path = Path(__file__).parent / "demo_dump.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nFull dump written to: {out_path}")
    print(f"Players: {len(output['players'])}")
    print(f"JSON size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

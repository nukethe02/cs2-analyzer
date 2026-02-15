"""
Economy and KAST computation methods extracted from DemoAnalyzer.

Contains:
- calculate_kast: KAST% (Kill/Assist/Survived/Traded) per player
- calculate_side_stats: CT vs T side performance breakdown
- calculate_economy_history: Round-by-round economy timeline
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from opensight.core.constants import TRADE_WINDOW_SECONDS
from opensight.core.parser import DemoData, safe_int

if TYPE_CHECKING:
    from opensight.analysis.analytics import DemoAnalyzer

logger = logging.getLogger(__name__)


def _normalize_sid_series(series: pd.Series) -> pd.Series:
    """Normalize steamid Series to clean strings (no float precision loss)."""
    s = series.astype(str).str.strip()
    # Remove trailing .0 from float-stored steamids (e.g. "76561198048426444.0")
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": "", "None": "", "0": ""})
    return s


def _normalize_sid(sid) -> str:
    """Normalize a single steamid to a clean string."""
    s = str(sid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def calculate_kast(analyzer: DemoAnalyzer) -> None:
    """Calculate KAST (Kill/Assist/Survived/Traded) for each player.

    Uses STRING-based steamid matching to avoid float64 precision loss
    on 17-digit SteamIDs (>2^53), which causes set-lookup failures when
    using pd.to_numeric or float().
    """
    kills_df = analyzer.data.kills_df
    if (
        kills_df.empty
        or not analyzer._round_col
        or not analyzer._att_id_col
        or not analyzer._vic_id_col
    ):
        logger.info("Skipping KAST calculation - missing columns")
        return

    trade_window_ticks = int(TRADE_WINDOW_SECONDS * analyzer.TICK_RATE)

    # Work on a copy — normalize steamids to clean strings
    kdf = kills_df.copy()
    att_col = analyzer._att_id_col
    vic_col = analyzer._vic_id_col
    kdf[att_col] = _normalize_sid_series(kdf[att_col])
    kdf[vic_col] = _normalize_sid_series(kdf[vic_col])

    # Create lookup sets: which players got K/A/Died in each round (string-based)
    kills_by_round = kdf.groupby(analyzer._round_col)[att_col].apply(set).to_dict()
    deaths_by_round = kdf.groupby(analyzer._round_col)[vic_col].apply(set).to_dict()

    assists_by_round: dict = {}
    if "assister_steamid" in kdf.columns:
        kdf["assister_steamid"] = _normalize_sid_series(kdf["assister_steamid"])
        valid_assists = kdf[kdf["assister_steamid"] != ""]
        if not valid_assists.empty:
            assists_by_round = (
                valid_assists.groupby(analyzer._round_col)["assister_steamid"].apply(set).to_dict()
            )

    # Pre-compute trade lookup (who was traded in each round)
    traded_by_round: dict[int, set] = {}
    if analyzer._att_side_col and analyzer._att_side_col in kdf.columns:
        # Build player team lookup (string keys)
        player_teams = {
            _normalize_sid(sid): player.team for sid, player in analyzer._players.items()
        }

        for round_num in kdf[analyzer._round_col].unique():
            round_kills = kdf[kdf[analyzer._round_col] == round_num].sort_values(by="tick")
            traded_players: set = set()

            for _idx, death in round_kills.iterrows():
                victim_id = str(death.get(vic_col, "")).strip()
                if not victim_id or victim_id == "nan":
                    continue

                death_tick = safe_int(death.get("tick"))
                killer_id = str(death.get(att_col, "")).strip()
                victim_team = player_teams.get(victim_id, "")

                if not killer_id or killer_id in ("", "nan") or not victim_team:
                    continue

                # Check if death was traded (killer killed within trade window)
                trade_mask = (
                    (round_kills["tick"] > death_tick)
                    & (round_kills["tick"] <= death_tick + trade_window_ticks)
                    & (round_kills[vic_col] == killer_id)
                )

                potential_trades = round_kills[trade_mask]
                for _, trade_kill in potential_trades.iterrows():
                    trader_id = str(trade_kill.get(att_col, "")).strip()
                    trader_team = player_teams.get(trader_id, "")
                    if trader_team == victim_team:
                        traded_players.add(victim_id)
                        break

            traded_by_round[round_num] = traded_players

    # Get unique round numbers
    round_nums = sorted(kdf[analyzer._round_col].unique())

    # Calculate KAST for each player
    for steam_id, player in analyzer._players.items():
        sid = _normalize_sid(steam_id)
        kast_count = 0
        survived_count = 0

        for round_num in round_nums:
            kast_this_round = False

            # K - Got a kill
            if sid in kills_by_round.get(round_num, set()):
                kast_this_round = True

            # A - Got an assist
            if not kast_this_round and sid in assists_by_round.get(round_num, set()):
                kast_this_round = True

            # S - Survived (didn't die)
            if sid not in deaths_by_round.get(round_num, set()):
                kast_this_round = True
                survived_count += 1

            # T - Was traded
            if not kast_this_round and sid in traded_by_round.get(round_num, set()):
                kast_this_round = True

            if kast_this_round:
                kast_count += 1

        player.kast_rounds = kast_count
        player.rounds_survived = survived_count

    logger.info(
        f"KAST calculation complete for {len(analyzer._players)} players "
        f"over {len(round_nums)} rounds"
    )


def _is_ct_side(series: pd.Series) -> pd.Series:
    """Return boolean mask for CT-side rows, handling both numeric and string columns."""
    if pd.api.types.is_numeric_dtype(series):
        return series == 3  # CS2: 3 = CT
    upper = series.astype(str).str.upper()
    return upper.str.contains("CT", na=False) | upper.str.contains("COUNTER", na=False)


def _is_t_side(series: pd.Series) -> pd.Series:
    """Return boolean mask for T-side rows, handling both numeric and string columns."""
    if pd.api.types.is_numeric_dtype(series):
        return series == 2  # CS2: 2 = T
    upper = series.astype(str).str.upper()
    is_ct = upper.str.contains("CT", na=False) | upper.str.contains("COUNTER", na=False)
    return upper.str.contains("T", na=False) & ~is_ct


def calculate_side_stats(analyzer: DemoAnalyzer) -> None:
    """Calculate CT-side vs T-side performance breakdown."""
    kills_df = analyzer.data.kills_df
    damages_df = analyzer.data.damages_df

    if kills_df.empty or not analyzer._att_id_col or not analyzer._att_side_col:
        logger.info("Skipping side stats - missing columns")
        return

    for steam_id, player in analyzer._players.items():
        # Count CT-side kills
        if analyzer._att_side_col:
            ct_kills_df = kills_df[
                (kills_df[analyzer._att_id_col] == steam_id)
                & _is_ct_side(kills_df[analyzer._att_side_col])
            ]
            player.ct_stats.kills = len(ct_kills_df)

            t_kills_df = kills_df[
                (kills_df[analyzer._att_id_col] == steam_id)
                & _is_t_side(kills_df[analyzer._att_side_col])
            ]
            player.t_stats.kills = len(t_kills_df)

        # Count CT-side deaths
        if analyzer._vic_id_col and analyzer._vic_side_col:
            ct_deaths_df = kills_df[
                (kills_df[analyzer._vic_id_col] == steam_id)
                & _is_ct_side(kills_df[analyzer._vic_side_col])
            ]
            player.ct_stats.deaths = len(ct_deaths_df)

            t_deaths_df = kills_df[
                (kills_df[analyzer._vic_id_col] == steam_id)
                & _is_t_side(kills_df[analyzer._vic_side_col])
            ]
            player.t_stats.deaths = len(t_deaths_df)

        # Estimate rounds per side (typically half each)
        total_rounds = max(analyzer.data.num_rounds, 1)
        half_rounds = total_rounds // 2
        player.ct_stats.rounds_played = half_rounds
        player.t_stats.rounds_played = total_rounds - half_rounds

        # Calculate side-specific damage
        if not damages_df.empty:
            dmg_att_col = analyzer._find_col(damages_df, analyzer.ATT_ID_COLS)
            dmg_att_side = analyzer._find_col(damages_df, analyzer.ATT_SIDE_COLS)
            dmg_col = analyzer._find_col(damages_df, ["dmg_health", "damage", "dmg"])

            if dmg_att_col and dmg_att_side and dmg_col:
                ct_dmg = damages_df[
                    (damages_df[dmg_att_col] == steam_id) & _is_ct_side(damages_df[dmg_att_side])
                ]
                player.ct_stats.damage = int(ct_dmg[dmg_col].sum())

                t_dmg = damages_df[
                    (damages_df[dmg_att_col] == steam_id) & _is_t_side(damages_df[dmg_att_side])
                ]
                player.t_stats.damage = int(t_dmg[dmg_col].sum())

    logger.info("Calculated side-based stats")


def calculate_economy_history(match_data: DemoData) -> list[dict]:
    """
    Calculate round-by-round economy history for both teams.

    Uses equipment values from kill/damage events to estimate team wealth
    per round. This enables economy flow visualization for coaching.

    Args:
        match_data: Parsed demo data (DemoData) from DemoParser

    Returns:
        List of dicts with round economy data:
        [
            {"round": 1, "team_t_val": 3500, "team_ct_val": 4500,
             "t_buy": "pistol", "ct_buy": "pistol"},
            {"round": 2, "team_t_val": 8000, "team_ct_val": 12000,
             "t_buy": "eco", "ct_buy": "full"},
            ...
        ]

    Example:
        >>> from opensight.core.parser import DemoParser
        >>> from opensight.analysis.compute_economy import calculate_economy_history
        >>> parser = DemoParser("match.dem")
        >>> data = parser.parse()
        >>> economy = calculate_economy_history(data)
        >>> for round_data in economy:
        ...     print(f"Round {round_data['round']}: "
        ...           f"T=${round_data['team_t_val']}, "
        ...           f"CT=${round_data['team_ct_val']}")
    """
    try:
        from opensight.domains.economy import EconomyAnalyzer
    except ImportError:
        logger.warning("Economy module not available, returning empty history")
        return []

    try:
        econ_analyzer = EconomyAnalyzer(match_data)
        stats = econ_analyzer.analyze()
    except Exception as e:
        logger.warning(f"Economy analysis failed: {e}")
        return []

    # Build round-by-round history from team economies
    # Team 2 = T, Team 3 = CT (CS2 convention)
    t_rounds = {tr.round_num: tr for tr in stats.team_economies.get(2, [])}
    ct_rounds = {tr.round_num: tr for tr in stats.team_economies.get(3, [])}

    # Get all round numbers
    all_rounds = sorted(set(t_rounds.keys()) | set(ct_rounds.keys()))

    if not all_rounds:
        # Fallback: Generate basic data from round count if no economy data
        logger.info("No detailed economy data, generating estimates from round count")
        history = []
        for round_num in range(1, match_data.num_rounds + 1):
            is_pistol = round_num in [
                1,
                13,
                16,
                28,
            ]  # Pistol rounds (MR12/MR15)
            is_second = round_num in [
                2,
                14,
                17,
                29,
            ]  # Often eco after pistol loss

            if is_pistol:
                t_val, ct_val = 800, 800
                t_buy, ct_buy = "pistol", "pistol"
            elif is_second:
                t_val, ct_val = 2000, 2000
                t_buy, ct_buy = "eco", "eco"
            else:
                # Estimate mid-game average
                t_val, ct_val = 20000, 22000
                t_buy, ct_buy = "full", "full"

            history.append(
                {
                    "round": round_num,
                    "team_t_val": t_val,
                    "team_ct_val": ct_val,
                    "t_buy": t_buy,
                    "ct_buy": ct_buy,
                }
            )
        return history

    history = []
    for round_num in all_rounds:
        t_round = t_rounds.get(round_num)
        ct_round = ct_rounds.get(round_num)

        t_val = t_round.total_equipment if t_round else 0
        ct_val = ct_round.total_equipment if ct_round else 0

        # Get buy type string
        t_buy = t_round.buy_type.value if t_round else "unknown"
        ct_buy = ct_round.buy_type.value if ct_round else "unknown"

        history.append(
            {
                "round": round_num,
                "team_t_val": t_val,
                "team_ct_val": ct_val,
                "t_buy": t_buy,
                "ct_buy": ct_buy,
            }
        )

    return history

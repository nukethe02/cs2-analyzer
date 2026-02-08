"""
Natural language query interface for match data.
Users ask questions in plain English -> system queries structured data ->
returns answers with supporting evidence.

Architecture: User query -> LLM classifies intent + extracts parameters ->
Python queries the data -> LLM formats the answer.

Uses ModelTier.STANDARD (Haiku) -- fast responses for interactive use.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from opensight.ai.llm_client import LLMClient, ModelTier

logger = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class QueryResult:
    """Result from a natural language query."""

    question: str
    answer: str
    data: dict[str, Any]
    confidence: str  # "high", "medium", "low"
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "data": self.data,
            "confidence": self.confidence,
            "sources": self.sources,
        }


# =============================================================================
# QueryInterface
# =============================================================================


class QueryInterface:
    """Natural language interface to match data."""

    SUPPORTED_QUERIES = [
        "player_stats",
        "team_performance",
        "opponent_scouting",
        "economy_analysis",
        "comparison",
        "trend",
        "round_specific",
        "tactical",
    ]

    def __init__(self) -> None:
        self.llm = LLMClient()

    def query(
        self,
        question: str,
        available_data: dict[str, Any],
    ) -> QueryResult:
        """
        Process a natural language query.

        Two-step LLM approach:
        1. CLASSIFY: Haiku classifies the query type and extracts parameters
        2. ANSWER: Python queries the data, then Haiku formats the response

        Args:
            question: Natural language question.
            available_data: Orchestrator result dict (players, round_timeline, etc.)
                Optionally includes "scouting" and "tracking" keys.

        Returns:
            QueryResult with answer and supporting data.
        """
        # Step 1: Classify query
        classification = self._classify_query(question)

        # Step 2: Execute data lookup based on classification
        data = self._execute_query(classification, available_data)

        # Step 3: Format answer
        answer = self._format_answer(question, data)

        return QueryResult(
            question=question,
            answer=answer,
            data=data,
            confidence=classification.get("confidence", "medium"),
            sources=data.get("sources", []),
        )

    # -------------------------------------------------------------------------
    # Step 1: Classification
    # -------------------------------------------------------------------------

    def _classify_query(self, question: str) -> dict[str, Any]:
        """
        Use Haiku to classify the query and extract parameters.

        Falls back to keyword-based classification when no API key is set.

        Returns dict with:
            type, player, map, metric, time_range, round_number,
            comparison_target, confidence
        """
        prompt = self._build_classification_prompt(question)

        if self.llm.api_key:
            try:
                client = self.llm._get_client()
                response = client.messages.create(
                    model=ModelTier.STANDARD.value,
                    max_tokens=200,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                return _parse_json(text)
            except Exception as e:
                logger.warning("LLM classification failed, using keyword fallback: %s", e)

        return self._classify_by_keywords(question)

    def _build_classification_prompt(self, question: str) -> str:
        """Build the classification prompt for LLM."""
        return f"""Classify this CS2 analytics query and extract parameters.

Query: "{question}"

Respond in JSON only. Fields:
- type: one of {self.SUPPORTED_QUERIES}
- player: player name if mentioned (null if team-wide)
- map: map name if mentioned, with de_ prefix (null if all maps)
- metric: specific metric if mentioned (null if general)
- time_range: "last_N" or "all" (default "all")
- round_number: specific round number if mentioned (null otherwise)
- comparison_target: second player name if comparison query (null otherwise)
- confidence: "high", "medium", or "low" based on how clear the query is

JSON:"""

    def _classify_by_keywords(self, question: str) -> dict[str, Any]:
        """Keyword-based query classification fallback when no LLM available."""
        q = question.lower()

        result: dict[str, Any] = {
            "type": "player_stats",
            "player": None,
            "map": None,
            "metric": None,
            "time_range": "all",
            "round_number": None,
            "comparison_target": None,
            "confidence": "medium",
        }

        # Detect query type by keywords
        if any(w in q for w in ["compare", "versus", " vs ", "better than", "worse than"]):
            result["type"] = "comparison"
        elif any(
            w in q
            for w in [
                "trend",
                "improving",
                "getting better",
                "getting worse",
                "over time",
                "progress",
            ]
        ):
            result["type"] = "trend"
        elif re.search(r"\bround\s+\d+", q):
            result["type"] = "round_specific"
            match = re.search(r"\bround\s+(\d+)", q)
            if match:
                result["round_number"] = int(match.group(1))
        elif any(w in q for w in ["economy", "eco ", "force buy", "save round", "money"]):
            result["type"] = "economy_analysis"
        elif any(w in q for w in ["strat", "execute", "setup", "default", "bombsite"]):
            result["type"] = "tactical"
        elif any(w in q for w in ["team ", "our ", "we ", "ct side", "t side"]):
            result["type"] = "team_performance"
        elif any(w in q for w in ["opponent", "enemy", "they ", "their "]):
            result["type"] = "opponent_scouting"

        # Detect map
        for map_suffix in [
            "mirage",
            "inferno",
            "nuke",
            "ancient",
            "anubis",
            "dust2",
            "vertigo",
            "overpass",
        ]:
            if map_suffix in q:
                result["map"] = f"de_{map_suffix}"
                break

        # Detect time range
        match = re.search(r"last\s*(\d+)", q)
        if match:
            result["time_range"] = f"last_{match.group(1)}"

        # Detect metric keywords
        _METRIC_MAP = {
            "k/d": "kd_ratio",
            "kill": "kills",
            "death": "deaths",
            "adr": "adr",
            "rating": "hltv_rating",
            "headshot": "headshot_pct",
            "kast": "kast_pct",
            "opening duel": "opening_duel_win_rate",
            "entry": "opening_duel_win_rate",
            "clutch": "clutch_win_rate",
            "trade": "trade_rate",
            "flash": "flash_assists",
            "utility": "utility_damage",
            "ttd": "ttd_median_ms",
            "crosshair": "cp_median_error_deg",
            "pistol": "pistol_round",
        }
        for keyword, metric in _METRIC_MAP.items():
            if keyword in q:
                result["metric"] = metric
                break

        return result

    # -------------------------------------------------------------------------
    # Step 2: Data execution
    # -------------------------------------------------------------------------

    def _execute_query(
        self, classification: dict[str, Any], available_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the classified query against actual data."""
        query_type = classification.get("type", "player_stats")

        handlers = {
            "player_stats": self._query_player_stats,
            "team_performance": self._query_team_performance,
            "opponent_scouting": self._query_opponent_scouting,
            "economy_analysis": self._query_economy,
            "comparison": self._query_comparison,
            "trend": self._query_trend,
            "round_specific": self._query_round,
            "tactical": self._query_tactical,
        }

        handler = handlers.get(query_type, self._query_player_stats)
        try:
            return handler(classification, available_data)
        except Exception as e:
            logger.warning("Query execution failed for type %s: %s", query_type, e)
            return {
                "error": f"Could not execute query: {e}",
                "type": query_type,
                "sources": [],
            }

    # -- Query handlers -------------------------------------------------------

    def _query_player_stats(
        self, classification: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Query player statistics from orchestrator results."""
        players = data.get("players", {})
        player_name = classification.get("player")
        map_filter = classification.get("map")
        metric = classification.get("metric")

        demo_info = data.get("demo_info", {})
        current_map = demo_info.get("map", "unknown")

        # If map filter specified and doesn't match current demo
        if map_filter and current_map != map_filter and current_map != "unknown":
            return {
                "message": f"Current demo is on {current_map}, not {map_filter}",
                "available_map": current_map,
                "sources": [current_map],
            }

        # Find player by name (case-insensitive)
        matched_player = None
        if player_name:
            for _sid, pdata in players.items():
                if pdata.get("name", "").lower() == player_name.lower():
                    matched_player = pdata
                    break

        if player_name and not matched_player:
            available = [p.get("name", sid) for sid, p in players.items()]
            return {
                "message": f"Player '{player_name}' not found",
                "available_players": available,
                "sources": [],
            }

        if matched_player:
            player_info = _extract_player_info(matched_player, current_map)
            if metric and metric in player_info:
                player_info["requested_metric"] = metric
                player_info["requested_value"] = player_info[metric]
            return player_info

        # No specific player — return all players summary
        all_players = []
        for _sid, pdata in players.items():
            stats = pdata.get("stats", {})
            rating_data = pdata.get("rating", {})
            all_players.append(
                {
                    "name": pdata.get("name"),
                    "kills": stats.get("kills", 0),
                    "deaths": stats.get("deaths", 0),
                    "adr": stats.get("adr", 0.0),
                    "hltv_rating": rating_data.get("hltv_rating", 0.0),
                }
            )

        return {"players": all_players, "map": current_map, "sources": [current_map]}

    def _query_team_performance(
        self, classification: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Aggregate team performance from round timeline."""
        timeline = data.get("round_timeline", [])
        demo_info = data.get("demo_info", {})
        players = data.get("players", {})

        if not timeline:
            return {"message": "No round data available", "sources": []}

        ct_wins = 0
        t_wins = 0
        pistol_results: list[dict[str, Any]] = []
        first_kill_ct = 0
        first_kill_t = 0
        rounds_with_fk = 0

        for rdata in timeline:
            winner = rdata.get("winner", "")
            if winner == "CT":
                ct_wins += 1
            elif winner == "T":
                t_wins += 1

            rnum = rdata.get("round_num", 0)
            if rnum in (1, 13):
                pistol_results.append({"round": rnum, "winner": winner})

            kills = rdata.get("kills", [])
            if kills:
                fk_team = kills[0].get("killer_team", "")
                if fk_team:
                    rounds_with_fk += 1
                    if fk_team == "CT":
                        first_kill_ct += 1
                    else:
                        first_kill_t += 1

        # Team totals
        team_stats: dict[str, dict[str, Any]] = {}
        for _sid, pdata in players.items():
            team = pdata.get("team", "Unknown")
            if team not in team_stats:
                team_stats[team] = {"kills": 0, "deaths": 0, "players": []}
            stats = pdata.get("stats", {})
            team_stats[team]["kills"] += stats.get("kills", 0)
            team_stats[team]["deaths"] += stats.get("deaths", 0)
            team_stats[team]["players"].append(pdata.get("name", ""))

        current_map = demo_info.get("map", "unknown")
        return {
            "map": current_map,
            "total_rounds": len(timeline),
            "score": f"{demo_info.get('score_ct', 0)}-{demo_info.get('score_t', 0)}",
            "ct_rounds_won": ct_wins,
            "t_rounds_won": t_wins,
            "pistol_results": pistol_results,
            "first_kill_rate": {
                "ct": first_kill_ct / rounds_with_fk if rounds_with_fk > 0 else 0,
                "t": first_kill_t / rounds_with_fk if rounds_with_fk > 0 else 0,
            },
            "teams": team_stats,
            "sources": [current_map],
        }

    def _query_opponent_scouting(
        self, classification: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Query scouting data about an opponent."""
        scouting = data.get("scouting", {})
        if scouting:
            return {"scouting": scouting, "sources": ["scouting_report"]}
        # Fall back to regular match analysis
        return self._query_team_performance(classification, data)

    def _query_economy(
        self, classification: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Query economy data from round timeline."""
        timeline = data.get("round_timeline", [])
        if not timeline:
            return {"message": "No round data available", "sources": []}

        eco_rounds: dict[str, list[dict]] = {"ct": [], "t": []}
        force_rounds: dict[str, list[dict]] = {"ct": [], "t": []}
        total_ct_buy = 0
        total_t_buy = 0

        for rdata in timeline:
            economy = rdata.get("economy", {})
            rnum = rdata.get("round_num", 0)
            winner = rdata.get("winner", "")

            for side in ("ct", "t"):
                side_econ = economy.get(side, {})
                buy_type = side_econ.get("buy_type", "")
                equipment = side_econ.get("equipment", 0)

                if side == "ct":
                    total_ct_buy += equipment
                else:
                    total_t_buy += equipment

                won = winner == side.upper()
                entry = {
                    "round": rnum,
                    "buy_type": buy_type,
                    "equipment": equipment,
                    "won": won,
                }

                if buy_type in ("eco", "semi_eco"):
                    eco_rounds[side].append(entry)
                elif buy_type in ("force", "half_buy"):
                    force_rounds[side].append(entry)

        ct_eco_wins = sum(1 for r in eco_rounds["ct"] if r["won"])
        t_eco_wins = sum(1 for r in eco_rounds["t"] if r["won"])
        ct_force_wins = sum(1 for r in force_rounds["ct"] if r["won"])
        t_force_wins = sum(1 for r in force_rounds["t"] if r["won"])

        demo_info = data.get("demo_info", {})
        return {
            "map": demo_info.get("map", "unknown"),
            "eco_round_stats": {
                "ct": {
                    "total": len(eco_rounds["ct"]),
                    "wins": ct_eco_wins,
                    "win_rate": ct_eco_wins / len(eco_rounds["ct"]) if eco_rounds["ct"] else 0,
                },
                "t": {
                    "total": len(eco_rounds["t"]),
                    "wins": t_eco_wins,
                    "win_rate": t_eco_wins / len(eco_rounds["t"]) if eco_rounds["t"] else 0,
                },
            },
            "force_buy_stats": {
                "ct": {
                    "total": len(force_rounds["ct"]),
                    "wins": ct_force_wins,
                    "win_rate": ct_force_wins / len(force_rounds["ct"])
                    if force_rounds["ct"]
                    else 0,
                },
                "t": {
                    "total": len(force_rounds["t"]),
                    "wins": t_force_wins,
                    "win_rate": t_force_wins / len(force_rounds["t"]) if force_rounds["t"] else 0,
                },
            },
            "avg_equipment": {
                "ct": total_ct_buy / len(timeline) if timeline else 0,
                "t": total_t_buy / len(timeline) if timeline else 0,
            },
            "sources": [demo_info.get("map", "unknown")],
        }

    def _query_comparison(
        self, classification: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare two players' stats."""
        players = data.get("players", {})
        player_a_name = classification.get("player")
        player_b_name = classification.get("comparison_target")

        if not player_a_name or not player_b_name:
            available = [p.get("name", sid) for sid, p in players.items()]
            return {
                "message": "Comparison requires two player names",
                "available_players": available,
                "sources": [],
            }

        player_a = None
        player_b = None
        for _sid, pdata in players.items():
            name_lower = pdata.get("name", "").lower()
            if name_lower == player_a_name.lower():
                player_a = pdata
            elif name_lower == player_b_name.lower():
                player_b = pdata

        if not player_a or not player_b:
            return {
                "message": f"Could not find both players: {player_a_name}, {player_b_name}",
                "sources": [],
            }

        a_stats = _extract_comparison_stats(player_a)
        b_stats = _extract_comparison_stats(player_b)

        # Compute diffs
        diffs: dict[str, float] = {}
        for key in (
            "kills",
            "deaths",
            "adr",
            "hltv_rating",
            "kast_pct",
            "headshot_pct",
            "opening_duel_win_rate",
        ):
            diffs[key] = a_stats.get(key, 0) - b_stats.get(key, 0)

        demo_info = data.get("demo_info", {})
        return {
            "player_a": a_stats,
            "player_b": b_stats,
            "diffs": diffs,
            "map": demo_info.get("map", "unknown"),
            "sources": [f"{a_stats['name']} vs {b_stats['name']}"],
        }

    def _query_trend(self, classification: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Query trend data from player tracker."""
        tracking = data.get("tracking", {})
        if tracking:
            return {"trends": tracking, "sources": ["player_tracker"]}
        return {
            "message": (
                "No trend data available. "
                "Use /api/player-tracking/{steam_id}/trends for historical trends."
            ),
            "sources": [],
        }

    def _query_round(self, classification: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Query specific round data from timeline."""
        timeline = data.get("round_timeline", [])
        round_num = classification.get("round_number")

        if round_num is None:
            return {
                "message": "No round number specified",
                "total_rounds": len(timeline),
                "sources": [],
            }

        for rdata in timeline:
            if rdata.get("round_num") == round_num:
                kills = rdata.get("kills", [])
                kill_feed = [
                    {
                        "killer": k.get("killer", "Unknown"),
                        "victim": k.get("victim", "Unknown"),
                        "weapon": k.get("weapon", ""),
                        "headshot": k.get("headshot", False),
                    }
                    for k in kills
                ]

                demo_info = data.get("demo_info", {})
                return {
                    "round_num": round_num,
                    "winner": rdata.get("winner", ""),
                    "win_reason": rdata.get("win_reason", ""),
                    "kills": kill_feed,
                    "total_kills": len(kills),
                    "economy": rdata.get("economy", {}),
                    "map": demo_info.get("map", "unknown"),
                    "sources": [f"Round {round_num}"],
                }

        return {
            "message": f"Round {round_num} not found",
            "total_rounds": len(timeline),
            "sources": [],
        }

    def _query_tactical(
        self, classification: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Query tactical data (bomb sites, utility usage)."""
        timeline = data.get("round_timeline", [])
        demo_info = data.get("demo_info", {})

        if not timeline:
            return {"message": "No tactical data available", "sources": []}

        # Bomb site preferences
        site_attempts: dict[str, int] = {"A": 0, "B": 0}
        site_wins: dict[str, int] = {"A": 0, "B": 0}

        for rdata in timeline:
            for ev in rdata.get("events", []):
                if ev.get("type") == "bomb_plant":
                    site = ev.get("site", "")
                    if site in site_attempts:
                        site_attempts[site] += 1
                        if rdata.get("winner") == "T":
                            site_wins[site] += 1

        # Utility summary
        total_utility = 0
        utility_before_entry = 0
        for rdata in timeline:
            kills = rdata.get("kills", [])
            first_tick = kills[0].get("tick", 0) if kills else float("inf")
            for util in rdata.get("utility", []):
                total_utility += 1
                if util.get("tick", 0) <= first_tick:
                    utility_before_entry += 1

        current_map = demo_info.get("map", "unknown")
        return {
            "map": current_map,
            "bomb_site_stats": {
                site: {
                    "attempts": site_attempts[site],
                    "wins": site_wins[site],
                    "win_rate": site_wins[site] / site_attempts[site]
                    if site_attempts[site] > 0
                    else 0,
                }
                for site in ("A", "B")
            },
            "utility_summary": {
                "total_thrown": total_utility,
                "before_entry": utility_before_entry,
                "entry_util_rate": utility_before_entry / total_utility if total_utility > 0 else 0,
            },
            "total_rounds": len(timeline),
            "sources": [current_map],
        }

    # -------------------------------------------------------------------------
    # Step 3: Answer formatting
    # -------------------------------------------------------------------------

    def _format_answer(self, question: str, data: dict[str, Any]) -> str:
        """Use Haiku to format the data into a natural language answer."""
        data_for_llm = {k: v for k, v in data.items() if k != "sources"}
        prompt = self._build_format_prompt(question, data_for_llm)

        if self.llm.api_key:
            try:
                client = self.llm._get_client()
                response = client.messages.create(
                    model=ModelTier.STANDARD.value,
                    max_tokens=300,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()
            except Exception as e:
                logger.warning("LLM answer formatting failed, using fallback: %s", e)

        return _format_fallback(question, data)

    @staticmethod
    def _build_format_prompt(question: str, data: dict[str, Any]) -> str:
        """Build the answer-formatting prompt for LLM."""
        return f"""Answer this CS2 question using ONLY the data provided.

Question: "{question}"

Data:
{json.dumps(data, indent=2, default=str)}

Rules:
- Be specific -- cite numbers
- If data is insufficient, say so honestly
- Keep answer under 150 words
- Do NOT make up statistics

Answer:"""


# =============================================================================
# Helpers (module-level, tested independently)
# =============================================================================


def _extract_player_info(pdata: dict[str, Any], map_name: str) -> dict[str, Any]:
    """Extract a flat stats dict from orchestrator player data."""
    stats = pdata.get("stats", {})
    rating_data = pdata.get("rating", {})
    advanced = pdata.get("advanced", {})
    entry = pdata.get("entry", {})
    trades = pdata.get("trades", {})
    clutches = pdata.get("clutches", {})
    utility = pdata.get("utility", {})

    return {
        "name": pdata.get("name"),
        "team": pdata.get("team"),
        "kills": stats.get("kills", 0),
        "deaths": stats.get("deaths", 0),
        "assists": stats.get("assists", 0),
        "adr": stats.get("adr", 0.0),
        "hltv_rating": rating_data.get("hltv_rating", 0.0),
        "kast_pct": rating_data.get("kast_percentage", 0.0),
        "headshot_pct": stats.get("headshot_pct", 0.0),
        "opening_duel_wins": entry.get("entry_kills", 0),
        "opening_duel_attempts": entry.get("entry_attempts", 0),
        "opening_duel_win_rate": entry.get("entry_success_pct", 0.0),
        "clutch_wins": clutches.get("clutch_wins", 0),
        "clutch_attempts": clutches.get("total_situations", 0),
        "trade_kill_success": trades.get("trade_kill_success", 0),
        "trade_kill_success_pct": trades.get("trade_kill_success_pct", 0.0),
        "flash_assists": utility.get("flash_assists", 0),
        "utility_damage": utility.get("he_damage", 0) + utility.get("molotov_damage", 0),
        "ttd_median_ms": advanced.get("ttd_median_ms"),
        "cp_median_error_deg": advanced.get("cp_median_error_deg"),
        "map": map_name,
        "sources": [f"{pdata.get('name')} on {map_name}"],
    }


def _extract_comparison_stats(pdata: dict[str, Any]) -> dict[str, Any]:
    """Extract stats subset for player comparison."""
    stats = pdata.get("stats", {})
    rating_data = pdata.get("rating", {})
    entry = pdata.get("entry", {})
    return {
        "name": pdata.get("name"),
        "kills": stats.get("kills", 0),
        "deaths": stats.get("deaths", 0),
        "adr": stats.get("adr", 0.0),
        "hltv_rating": rating_data.get("hltv_rating", 0.0),
        "kast_pct": rating_data.get("kast_percentage", 0.0),
        "headshot_pct": stats.get("headshot_pct", 0.0),
        "opening_duel_win_rate": entry.get("entry_success_pct", 0.0),
    }


def _parse_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response text."""
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # First { ... } block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
    return {"type": "player_stats", "confidence": "low"}


def _format_fallback(question: str, data: dict[str, Any]) -> str:
    """Format data into text answer without LLM."""
    if "error" in data:
        return f"Unable to answer: {data['error']}"
    if "message" in data:
        return str(data["message"])

    parts: list[str] = []

    if "name" in data and "kills" in data:
        # Player stats
        parts.append(f"**{data['name']}** on {data.get('map', 'unknown')}:")
        parts.append(f"- K/D: {data.get('kills', 0)}/{data.get('deaths', 0)}")
        parts.append(f"- ADR: {data.get('adr', 0):.1f}")
        parts.append(f"- HLTV Rating: {data.get('hltv_rating', 0):.2f}")
        parts.append(f"- KAST: {data.get('kast_pct', 0):.0f}%")
        if data.get("requested_metric"):
            parts.append(f"\nRequested ({data['requested_metric']}): {data['requested_value']}")
    elif "player_a" in data:
        # Comparison
        a = data["player_a"]
        b = data["player_b"]
        parts.append(f"**{a['name']}** vs **{b['name']}**:")
        parts.append(f"- Rating: {a['hltv_rating']:.2f} vs {b['hltv_rating']:.2f}")
        parts.append(f"- K/D: {a['kills']}/{a['deaths']} vs {b['kills']}/{b['deaths']}")
        parts.append(f"- ADR: {a['adr']:.1f} vs {b['adr']:.1f}")
    elif "round_num" in data:
        # Round specific
        parts.append(
            f"**Round {data['round_num']}**: "
            f"{data.get('winner', '?')} wins ({data.get('win_reason', '')})"
        )
        parts.append(f"Kills: {data.get('total_kills', 0)}")
        for k in data.get("kills", []):
            hs = " (HS)" if k.get("headshot") else ""
            parts.append(f"  - {k['killer']} -> {k['victim']} [{k.get('weapon', '')}]{hs}")
    elif "eco_round_stats" in data:
        # Economy
        eco = data["eco_round_stats"]
        parts.append("**Economy Analysis:**")
        for side in ("ct", "t"):
            s = eco[side]
            parts.append(
                f"- {side.upper()} eco rounds: {s['wins']}/{s['total']} won ({s['win_rate']:.0%})"
            )
    elif "total_rounds" in data and "score" in data:
        # Team performance
        parts.append(f"Score: {data.get('score', '?')} on {data.get('map', '?')}")
        parts.append(
            f"CT rounds: {data.get('ct_rounds_won', 0)}, T rounds: {data.get('t_rounds_won', 0)}"
        )
    else:
        # Generic
        parts.append(json.dumps(data, indent=2, default=str)[:500])

    return "\n".join(parts)


# =============================================================================
# Singleton
# =============================================================================

_query_interface_instance: QueryInterface | None = None


def get_query_interface() -> QueryInterface:
    """Get or create singleton QueryInterface instance."""
    global _query_interface_instance
    if _query_interface_instance is None:
        _query_interface_instance = QueryInterface()
    return _query_interface_instance

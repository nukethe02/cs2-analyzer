"""Tests for the natural language query interface."""

import json
from unittest.mock import MagicMock, patch

import pytest

from opensight.ai.query_interface import (
    QueryInterface,
    QueryResult,
    _extract_comparison_stats,
    _extract_player_info,
    _format_fallback,
    _parse_json,
    get_query_interface,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def qi():
    """QueryInterface with no API key (uses keyword fallback)."""
    with patch.dict("os.environ", {}, clear=False):
        instance = QueryInterface()
        instance.llm.api_key = None
        return instance


@pytest.fixture()
def demo_data():
    """Realistic orchestrator result dict."""
    return {
        "demo_info": {
            "map": "de_inferno",
            "score_ct": 13,
            "score_t": 11,
            "rounds": 24,
        },
        "players": {
            "76561198001": {
                "name": "kix",
                "team": "CT",
                "stats": {
                    "kills": 25,
                    "deaths": 16,
                    "assists": 5,
                    "adr": 82.3,
                    "headshot_pct": 48.0,
                },
                "rating": {"hltv_rating": 1.28, "kast_percentage": 75.0},
                "advanced": {"ttd_median_ms": 280, "cp_median_error_deg": 8.5},
                "entry": {
                    "entry_attempts": 10,
                    "entry_kills": 6,
                    "entry_deaths": 4,
                    "entry_success_pct": 60.0,
                },
                "trades": {
                    "trade_kill_success": 4,
                    "trade_kill_success_pct": 57.1,
                },
                "clutches": {"clutch_wins": 2, "total_situations": 3},
                "utility": {
                    "flash_assists": 5,
                    "flashbangs_thrown": 12,
                    "he_damage": 80,
                    "molotov_damage": 120,
                },
            },
            "76561198002": {
                "name": "foe",
                "team": "CT",
                "stats": {
                    "kills": 18,
                    "deaths": 19,
                    "assists": 7,
                    "adr": 68.1,
                    "headshot_pct": 35.0,
                },
                "rating": {"hltv_rating": 0.92, "kast_percentage": 62.0},
                "advanced": {"ttd_median_ms": 350, "cp_median_error_deg": 14.2},
                "entry": {
                    "entry_attempts": 5,
                    "entry_kills": 2,
                    "entry_deaths": 3,
                    "entry_success_pct": 40.0,
                },
                "trades": {
                    "trade_kill_success": 2,
                    "trade_kill_success_pct": 33.3,
                },
                "clutches": {"clutch_wins": 0, "total_situations": 1},
                "utility": {
                    "flash_assists": 8,
                    "flashbangs_thrown": 18,
                    "he_damage": 40,
                    "molotov_damage": 60,
                },
            },
            "76561198003": {
                "name": "rival",
                "team": "T",
                "stats": {
                    "kills": 20,
                    "deaths": 17,
                    "assists": 3,
                    "adr": 74.5,
                    "headshot_pct": 55.0,
                },
                "rating": {"hltv_rating": 1.10, "kast_percentage": 70.0},
                "advanced": {"ttd_median_ms": 310, "cp_median_error_deg": 10.0},
                "entry": {
                    "entry_attempts": 8,
                    "entry_kills": 5,
                    "entry_deaths": 3,
                    "entry_success_pct": 62.5,
                },
                "trades": {
                    "trade_kill_success": 3,
                    "trade_kill_success_pct": 50.0,
                },
                "clutches": {"clutch_wins": 1, "total_situations": 2},
                "utility": {
                    "flash_assists": 3,
                    "flashbangs_thrown": 8,
                    "he_damage": 50,
                    "molotov_damage": 30,
                },
            },
        },
        "round_timeline": [
            {
                "round_num": 1,
                "winner": "CT",
                "win_reason": "elimination",
                "kills": [
                    {
                        "killer": "kix",
                        "victim": "rival",
                        "killer_team": "CT",
                        "weapon": "usp_silencer",
                        "headshot": True,
                        "tick": 1000,
                    },
                    {
                        "killer": "foe",
                        "victim": "enemy2",
                        "killer_team": "CT",
                        "weapon": "usp_silencer",
                        "headshot": False,
                        "tick": 1200,
                    },
                ],
                "economy": {
                    "ct": {"buy_type": "pistol", "equipment": 800},
                    "t": {"buy_type": "pistol", "equipment": 800},
                },
                "utility": [{"tick": 900, "type": "flashbang"}],
                "events": [],
            },
            {
                "round_num": 2,
                "winner": "CT",
                "win_reason": "elimination",
                "kills": [
                    {
                        "killer": "kix",
                        "victim": "rival",
                        "killer_team": "CT",
                        "weapon": "m4a1_silencer",
                        "headshot": False,
                        "tick": 2000,
                    },
                ],
                "economy": {
                    "ct": {"buy_type": "full", "equipment": 4500},
                    "t": {"buy_type": "eco", "equipment": 1200},
                },
                "utility": [],
                "events": [],
            },
            {
                "round_num": 3,
                "winner": "T",
                "win_reason": "bomb_exploded",
                "kills": [
                    {
                        "killer": "rival",
                        "victim": "foe",
                        "killer_team": "T",
                        "weapon": "glock",
                        "headshot": True,
                        "tick": 3000,
                    },
                ],
                "economy": {
                    "ct": {"buy_type": "full", "equipment": 5200},
                    "t": {"buy_type": "force", "equipment": 3000},
                },
                "utility": [
                    {"tick": 2800, "type": "smoke"},
                    {"tick": 2900, "type": "flashbang"},
                ],
                "events": [{"type": "bomb_plant", "site": "B"}],
            },
            {
                "round_num": 13,
                "winner": "T",
                "win_reason": "elimination",
                "kills": [],
                "economy": {
                    "ct": {"buy_type": "pistol", "equipment": 800},
                    "t": {"buy_type": "pistol", "equipment": 800},
                },
                "utility": [],
                "events": [],
            },
        ],
    }


# =============================================================================
# QueryResult
# =============================================================================


class TestQueryResult:
    def test_construction(self):
        qr = QueryResult(
            question="test?",
            answer="answer",
            data={"key": "val"},
            confidence="high",
            sources=["demo1"],
        )
        assert qr.question == "test?"
        assert qr.confidence == "high"

    def test_to_dict(self):
        qr = QueryResult(
            question="q?",
            answer="a",
            data={"x": 1},
            confidence="medium",
            sources=["s"],
        )
        d = qr.to_dict()
        assert d["question"] == "q?"
        assert d["answer"] == "a"
        assert d["data"] == {"x": 1}
        assert d["confidence"] == "medium"
        assert d["sources"] == ["s"]

    def test_default_sources(self):
        qr = QueryResult(question="q?", answer="a", data={}, confidence="low")
        assert qr.sources == []


# =============================================================================
# JSON parsing
# =============================================================================


class TestParseJson:
    def test_direct_json(self):
        result = _parse_json('{"type": "player_stats", "player": "kix"}')
        assert result["type"] == "player_stats"
        assert result["player"] == "kix"

    def test_code_block(self):
        text = '```json\n{"type": "economy_analysis"}\n```'
        result = _parse_json(text)
        assert result["type"] == "economy_analysis"

    def test_embedded_json(self):
        text = 'Here is the classification: {"type": "round_specific", "round_number": 14}'
        result = _parse_json(text)
        assert result["type"] == "round_specific"

    def test_unparseable_returns_default(self):
        result = _parse_json("This is not JSON at all")
        assert result["type"] == "player_stats"
        assert result["confidence"] == "low"


# =============================================================================
# Keyword classification
# =============================================================================


class TestKeywordClassification:
    def test_player_stats_default(self, qi):
        c = qi._classify_by_keywords("What is kix's K/D?")
        assert c["type"] == "player_stats"
        assert c["metric"] == "kd_ratio"

    def test_comparison(self, qi):
        c = qi._classify_by_keywords("Compare kix vs foe")
        assert c["type"] == "comparison"

    def test_trend(self, qi):
        c = qi._classify_by_keywords("Am I improving over time?")
        assert c["type"] == "trend"

    def test_round_specific(self, qi):
        c = qi._classify_by_keywords("What happened in round 14?")
        assert c["type"] == "round_specific"
        assert c["round_number"] == 14

    def test_economy(self, qi):
        c = qi._classify_by_keywords("How often do we force buy and win?")
        assert c["type"] == "economy_analysis"

    def test_team_performance(self, qi):
        c = qi._classify_by_keywords("How does our team do on CT side?")
        assert c["type"] == "team_performance"

    def test_opponent(self, qi):
        c = qi._classify_by_keywords("What does the opponent do on pistol?")
        assert c["type"] == "opponent_scouting"

    def test_tactical(self, qi):
        c = qi._classify_by_keywords("What's our most successful execute on default setup?")
        assert c["type"] == "tactical"

    def test_map_detection(self, qi):
        c = qi._classify_by_keywords("What is my rating on inferno?")
        assert c["map"] == "de_inferno"

    def test_time_range(self, qi):
        c = qi._classify_by_keywords("Show stats from last 20 matches")
        assert c["time_range"] == "last_20"

    def test_metric_adr(self, qi):
        c = qi._classify_by_keywords("What is my adr?")
        assert c["metric"] == "adr"

    def test_metric_opening_duel(self, qi):
        c = qi._classify_by_keywords("How often does kix win opening duels?")
        assert c["metric"] == "opening_duel_win_rate"

    def test_metric_clutch(self, qi):
        c = qi._classify_by_keywords("What's my clutch win rate?")
        assert c["metric"] == "clutch_win_rate"

    def test_dust2_map(self, qi):
        c = qi._classify_by_keywords("Stats on dust2")
        assert c["map"] == "de_dust2"


# =============================================================================
# _classify_query (with LLM mock)
# =============================================================================


class TestClassifyQueryLLM:
    def test_llm_path(self, qi):
        """When API key is set, uses LLM classification."""
        qi.llm.api_key = "test-key"
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"type": "player_stats", "player": "kix", "confidence": "high"}'
            )
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch.object(qi.llm, "_get_client", return_value=mock_client):
            result = qi._classify_query("What is kix's rating?")

        assert result["type"] == "player_stats"
        assert result["player"] == "kix"
        assert result["confidence"] == "high"

    def test_llm_fallback_on_error(self, qi):
        """Falls back to keywords if LLM fails."""
        qi.llm.api_key = "test-key"
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API error")

        with patch.object(qi.llm, "_get_client", return_value=mock_client):
            result = qi._classify_query("What is my adr on inferno?")

        # Falls back to keyword classification
        assert result["metric"] == "adr"
        assert result["map"] == "de_inferno"


# =============================================================================
# Player stats query
# =============================================================================


class TestQueryPlayerStats:
    def test_specific_player(self, qi, demo_data):
        classification = {"player": "kix", "map": None, "metric": None}
        result = qi._query_player_stats(classification, demo_data)
        assert result["name"] == "kix"
        assert result["kills"] == 25
        assert result["deaths"] == 16
        assert result["adr"] == 82.3
        assert result["hltv_rating"] == 1.28

    def test_player_not_found(self, qi, demo_data):
        classification = {"player": "unknown_player", "map": None, "metric": None}
        result = qi._query_player_stats(classification, demo_data)
        assert "not found" in result["message"]
        assert "available_players" in result

    def test_all_players_summary(self, qi, demo_data):
        classification = {"player": None, "map": None, "metric": None}
        result = qi._query_player_stats(classification, demo_data)
        assert "players" in result
        assert len(result["players"]) == 3

    def test_specific_metric(self, qi, demo_data):
        classification = {"player": "kix", "map": None, "metric": "adr"}
        result = qi._query_player_stats(classification, demo_data)
        assert result["requested_metric"] == "adr"
        assert result["requested_value"] == 82.3

    def test_map_filter_mismatch(self, qi, demo_data):
        classification = {"player": "kix", "map": "de_mirage", "metric": None}
        result = qi._query_player_stats(classification, demo_data)
        assert "de_inferno" in result.get("message", "") or result.get("available_map") == "de_inferno"

    def test_map_filter_match(self, qi, demo_data):
        classification = {"player": "kix", "map": "de_inferno", "metric": None}
        result = qi._query_player_stats(classification, demo_data)
        assert result["name"] == "kix"

    def test_case_insensitive_player_name(self, qi, demo_data):
        classification = {"player": "KIX", "map": None, "metric": None}
        result = qi._query_player_stats(classification, demo_data)
        assert result["name"] == "kix"


# =============================================================================
# Team performance query
# =============================================================================


class TestQueryTeamPerformance:
    def test_basic(self, qi, demo_data):
        classification = {}
        result = qi._query_team_performance(classification, demo_data)
        assert result["map"] == "de_inferno"
        assert result["total_rounds"] == 4
        assert result["ct_rounds_won"] == 2
        assert result["t_rounds_won"] == 2

    def test_pistol_results(self, qi, demo_data):
        classification = {}
        result = qi._query_team_performance(classification, demo_data)
        assert len(result["pistol_results"]) == 2
        assert result["pistol_results"][0]["round"] == 1
        assert result["pistol_results"][0]["winner"] == "CT"

    def test_empty_timeline(self, qi):
        data = {"round_timeline": [], "demo_info": {}, "players": {}}
        result = qi._query_team_performance({}, data)
        assert result["message"] == "No round data available"

    def test_first_kill_rate(self, qi, demo_data):
        classification = {}
        result = qi._query_team_performance(classification, demo_data)
        # Rounds 1, 2, 3 have kills with killer_team; round 13 has no kills
        # Rounds 1,2: CT first kill, Round 3: T first kill
        assert result["first_kill_rate"]["ct"] == pytest.approx(2 / 3)
        assert result["first_kill_rate"]["t"] == pytest.approx(1 / 3)


# =============================================================================
# Economy query
# =============================================================================


class TestQueryEconomy:
    def test_eco_rounds(self, qi, demo_data):
        classification = {}
        result = qi._query_economy(classification, demo_data)
        assert result["map"] == "de_inferno"
        # Round 2 T side is eco
        assert result["eco_round_stats"]["t"]["total"] == 1
        # T lost that eco (CT won round 2)
        assert result["eco_round_stats"]["t"]["wins"] == 0

    def test_force_rounds(self, qi, demo_data):
        classification = {}
        result = qi._query_economy(classification, demo_data)
        # Round 3 T side is force
        assert result["force_buy_stats"]["t"]["total"] == 1
        # T won round 3
        assert result["force_buy_stats"]["t"]["wins"] == 1

    def test_empty_timeline(self, qi):
        data = {"round_timeline": [], "demo_info": {}}
        result = qi._query_economy({}, data)
        assert result["message"] == "No round data available"


# =============================================================================
# Comparison query
# =============================================================================


class TestQueryComparison:
    def test_basic(self, qi, demo_data):
        classification = {"player": "kix", "comparison_target": "foe"}
        result = qi._query_comparison(classification, demo_data)
        assert result["player_a"]["name"] == "kix"
        assert result["player_b"]["name"] == "foe"
        assert result["diffs"]["kills"] == 7  # 25 - 18

    def test_missing_player(self, qi, demo_data):
        classification = {"player": "kix", "comparison_target": "nobody"}
        result = qi._query_comparison(classification, demo_data)
        assert "Could not find" in result["message"]

    def test_no_names_provided(self, qi, demo_data):
        classification = {"player": None, "comparison_target": None}
        result = qi._query_comparison(classification, demo_data)
        assert "requires two player names" in result["message"]
        assert "available_players" in result

    def test_diffs_correct(self, qi, demo_data):
        classification = {"player": "kix", "comparison_target": "foe"}
        result = qi._query_comparison(classification, demo_data)
        assert result["diffs"]["hltv_rating"] == pytest.approx(0.36)
        assert result["diffs"]["adr"] == pytest.approx(14.2)


# =============================================================================
# Round-specific query
# =============================================================================


class TestQueryRound:
    def test_specific_round(self, qi, demo_data):
        classification = {"round_number": 1}
        result = qi._query_round(classification, demo_data)
        assert result["round_num"] == 1
        assert result["winner"] == "CT"
        assert result["total_kills"] == 2

    def test_kill_feed(self, qi, demo_data):
        classification = {"round_number": 1}
        result = qi._query_round(classification, demo_data)
        assert len(result["kills"]) == 2
        assert result["kills"][0]["killer"] == "kix"
        assert result["kills"][0]["headshot"] is True

    def test_round_not_found(self, qi, demo_data):
        classification = {"round_number": 99}
        result = qi._query_round(classification, demo_data)
        assert "not found" in result["message"]

    def test_no_round_number(self, qi, demo_data):
        classification = {"round_number": None}
        result = qi._query_round(classification, demo_data)
        assert "No round number" in result["message"]
        assert result["total_rounds"] == 4


# =============================================================================
# Tactical query
# =============================================================================


class TestQueryTactical:
    def test_bomb_sites(self, qi, demo_data):
        classification = {}
        result = qi._query_tactical(classification, demo_data)
        # Round 3 has bomb_plant at B, T won
        assert result["bomb_site_stats"]["B"]["attempts"] == 1
        assert result["bomb_site_stats"]["B"]["wins"] == 1
        assert result["bomb_site_stats"]["A"]["attempts"] == 0

    def test_utility_summary(self, qi, demo_data):
        classification = {}
        result = qi._query_tactical(classification, demo_data)
        # Round 1: 1 util (tick 900) before first kill (tick 1000)
        # Round 3: 2 utils (tick 2800, 2900) before first kill (tick 3000)
        assert result["utility_summary"]["total_thrown"] == 3
        assert result["utility_summary"]["before_entry"] == 3

    def test_empty_timeline(self, qi):
        data = {"round_timeline": [], "demo_info": {}}
        result = qi._query_tactical({}, data)
        assert result["message"] == "No tactical data available"


# =============================================================================
# Trend query
# =============================================================================


class TestQueryTrend:
    def test_with_tracking_data(self, qi):
        data = {"tracking": {"hltv_rating": [1.0, 1.1, 1.2]}}
        result = qi._query_trend({}, data)
        assert result["sources"] == ["player_tracker"]

    def test_no_tracking_data(self, qi, demo_data):
        result = qi._query_trend({}, demo_data)
        assert "No trend data" in result["message"]


# =============================================================================
# Opponent scouting query
# =============================================================================


class TestQueryOpponentScouting:
    def test_with_scouting_data(self, qi):
        data = {"scouting": {"team": "Evil Corp", "tendencies": []}}
        result = qi._query_opponent_scouting({}, data)
        assert result["scouting"]["team"] == "Evil Corp"

    def test_fallback_to_team_performance(self, qi, demo_data):
        result = qi._query_opponent_scouting({}, demo_data)
        # Falls back to team_performance since no scouting key
        assert "total_rounds" in result


# =============================================================================
# Format fallback
# =============================================================================


class TestFormatFallback:
    def test_error_message(self):
        result = _format_fallback("q?", {"error": "something broke"})
        assert "something broke" in result

    def test_info_message(self):
        result = _format_fallback("q?", {"message": "No data"})
        assert result == "No data"

    def test_player_stats(self):
        data = {
            "name": "kix",
            "kills": 25,
            "deaths": 16,
            "adr": 82.3,
            "hltv_rating": 1.28,
            "kast_pct": 75,
            "map": "de_inferno",
        }
        result = _format_fallback("q?", data)
        assert "kix" in result
        assert "25/16" in result
        assert "82.3" in result
        assert "1.28" in result

    def test_comparison(self):
        data = {
            "player_a": {
                "name": "kix",
                "hltv_rating": 1.28,
                "kills": 25,
                "deaths": 16,
                "adr": 82.3,
            },
            "player_b": {
                "name": "foe",
                "hltv_rating": 0.92,
                "kills": 18,
                "deaths": 19,
                "adr": 68.1,
            },
        }
        result = _format_fallback("q?", data)
        assert "kix" in result
        assert "foe" in result
        assert "1.28" in result

    def test_round_specific(self):
        data = {
            "round_num": 5,
            "winner": "CT",
            "win_reason": "elimination",
            "total_kills": 3,
            "kills": [
                {
                    "killer": "kix",
                    "victim": "rival",
                    "weapon": "ak47",
                    "headshot": True,
                }
            ],
        }
        result = _format_fallback("q?", data)
        assert "Round 5" in result
        assert "kix" in result
        assert "(HS)" in result

    def test_economy(self):
        data = {
            "eco_round_stats": {
                "ct": {"total": 3, "wins": 1, "win_rate": 0.333},
                "t": {"total": 2, "wins": 0, "win_rate": 0.0},
            }
        }
        result = _format_fallback("q?", data)
        assert "Economy" in result
        assert "CT" in result

    def test_team_performance(self):
        data = {
            "total_rounds": 24,
            "score": "13-11",
            "map": "de_inferno",
            "ct_rounds_won": 8,
            "t_rounds_won": 5,
        }
        result = _format_fallback("q?", data)
        assert "13-11" in result


# =============================================================================
# Helper functions
# =============================================================================


class TestExtractPlayerInfo:
    def test_basic(self):
        pdata = {
            "name": "kix",
            "team": "CT",
            "stats": {"kills": 25, "deaths": 16, "assists": 5, "adr": 82.3, "headshot_pct": 48.0},
            "rating": {"hltv_rating": 1.28, "kast_percentage": 75.0},
            "advanced": {"ttd_median_ms": 280, "cp_median_error_deg": 8.5},
            "entry": {"entry_kills": 6, "entry_attempts": 10, "entry_success_pct": 60.0},
            "trades": {"trade_kill_success": 4, "trade_kill_success_pct": 57.1},
            "clutches": {"clutch_wins": 2, "total_situations": 3},
            "utility": {"flash_assists": 5, "he_damage": 80, "molotov_damage": 120},
        }
        result = _extract_player_info(pdata, "de_inferno")
        assert result["name"] == "kix"
        assert result["kills"] == 25
        assert result["utility_damage"] == 200
        assert result["map"] == "de_inferno"

    def test_missing_sub_dicts(self):
        pdata = {"name": "empty_player"}
        result = _extract_player_info(pdata, "de_dust2")
        assert result["name"] == "empty_player"
        assert result["kills"] == 0
        assert result["hltv_rating"] == 0.0


class TestExtractComparisonStats:
    def test_basic(self):
        pdata = {
            "name": "kix",
            "stats": {"kills": 25, "deaths": 16, "adr": 82.3, "headshot_pct": 48.0},
            "rating": {"hltv_rating": 1.28, "kast_percentage": 75.0},
            "entry": {"entry_success_pct": 60.0},
        }
        result = _extract_comparison_stats(pdata)
        assert result["name"] == "kix"
        assert result["kills"] == 25
        assert result["opening_duel_win_rate"] == 60.0


# =============================================================================
# _execute_query error handling
# =============================================================================


class TestExecuteQueryErrors:
    def test_unknown_type_falls_back(self, qi, demo_data):
        classification = {"type": "nonexistent_type", "player": None, "map": None, "metric": None}
        result = qi._execute_query(classification, demo_data)
        # Falls back to player_stats handler
        assert "players" in result or "name" in result

    def test_handler_exception_caught(self, qi):
        """If a handler raises, _execute_query returns an error dict."""
        classification = {"type": "round_specific", "round_number": None}
        # Pass data that will cause issues
        data = {"round_timeline": None}  # None instead of list
        result = qi._execute_query(classification, data)
        assert "error" in result


# =============================================================================
# Full query integration (no LLM)
# =============================================================================


class TestFullQuery:
    def test_player_query(self, qi, demo_data):
        result = qi.query("What is kix's K/D on inferno?", demo_data)
        assert isinstance(result, QueryResult)
        assert result.question == "What is kix's K/D on inferno?"
        assert result.data.get("name") == "kix" or "kix" in result.answer

    def test_economy_query(self, qi, demo_data):
        result = qi.query("How does our economy look?", demo_data)
        assert isinstance(result, QueryResult)
        assert result.confidence in ("high", "medium", "low")

    def test_round_query(self, qi, demo_data):
        result = qi.query("What happened in round 3?", demo_data)
        assert isinstance(result, QueryResult)
        assert result.data.get("round_num") == 3

    def test_comparison_query(self, qi, demo_data):
        """Comparison without LLM won't have player names extracted, falls back gracefully."""
        result = qi.query("Compare kix vs foe", demo_data)
        assert isinstance(result, QueryResult)
        # Keyword classifier detects "comparison" type but can't extract names
        # So it returns a message about needing two player names
        assert result.data is not None

    def test_trend_query(self, qi, demo_data):
        result = qi.query("Am I improving over time?", demo_data)
        assert isinstance(result, QueryResult)
        assert "trend" in result.answer.lower() or "No trend" in result.answer


# =============================================================================
# Format answer with LLM
# =============================================================================


class TestFormatAnswerLLM:
    def test_llm_formatting(self, qi):
        qi.llm.api_key = "test-key"
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="kix has a K/D of 25/16 (1.56) with 82.3 ADR on Inferno.")
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        data = {"name": "kix", "kills": 25, "deaths": 16, "adr": 82.3}
        with patch.object(qi.llm, "_get_client", return_value=mock_client):
            answer = qi._format_answer("What is kix's K/D?", data)

        assert "kix" in answer
        assert "25/16" in answer

    def test_llm_error_falls_back(self, qi):
        qi.llm.api_key = "test-key"
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API error")

        data = {"name": "kix", "kills": 25, "deaths": 16, "adr": 82.3, "hltv_rating": 1.28, "kast_pct": 75, "map": "de_inferno"}
        with patch.object(qi.llm, "_get_client", return_value=mock_client):
            answer = qi._format_answer("What is kix's stats?", data)

        # Falls back to _format_fallback
        assert "kix" in answer


# =============================================================================
# Classification prompt building
# =============================================================================


class TestBuildPrompts:
    def test_classification_prompt(self, qi):
        prompt = qi._build_classification_prompt("What is kix's rating?")
        assert '"What is kix\'s rating?"' in prompt
        assert "player_stats" in prompt
        assert "JSON" in prompt

    def test_format_prompt(self):
        prompt = QueryInterface._build_format_prompt(
            "What is kix's K/D?", {"kills": 25, "deaths": 16}
        )
        assert "kix" in prompt
        assert "25" in prompt
        assert "150 words" in prompt


# =============================================================================
# Singleton
# =============================================================================


class TestSingleton:
    def test_get_query_interface(self):
        import opensight.ai.query_interface as mod

        mod._query_interface_instance = None
        qi1 = get_query_interface()
        qi2 = get_query_interface()
        assert qi1 is qi2
        mod._query_interface_instance = None  # cleanup


# =============================================================================
# Supported queries constant
# =============================================================================


class TestSupportedQueries:
    def test_eight_types(self):
        assert len(QueryInterface.SUPPORTED_QUERIES) == 8

    def test_known_types(self):
        expected = {
            "player_stats",
            "team_performance",
            "opponent_scouting",
            "economy_analysis",
            "comparison",
            "trend",
            "round_specific",
            "tactical",
        }
        assert set(QueryInterface.SUPPORTED_QUERIES) == expected


# =============================================================================
# API endpoint
# =============================================================================


class TestQueryAPI:
    def test_missing_question(self):
        from fastapi.testclient import TestClient

        from opensight.api import app

        client = TestClient(app)
        resp = client.post("/api/query", json={"question": "", "job_id": "abc123def456"})
        assert resp.status_code == 400

    def test_missing_job_id(self):
        from fastapi.testclient import TestClient

        from opensight.api import app

        client = TestClient(app)
        resp = client.post("/api/query", json={"question": "test?"})
        assert resp.status_code == 422  # missing required field

    def test_job_not_found(self):
        from fastapi.testclient import TestClient

        from opensight.api import app

        client = TestClient(app)
        resp = client.post(
            "/api/query",
            json={"question": "What is my K/D?", "job_id": "nonexistent12345"},
        )
        assert resp.status_code == 404

    def test_success(self):
        from unittest.mock import patch as mock_patch

        from fastapi.testclient import TestClient

        from opensight.api import app
        from opensight.api.shared import Job

        client = TestClient(app)

        fake_result = {
            "demo_info": {"map": "de_inferno", "score_ct": 13, "score_t": 11},
            "players": {
                "123": {
                    "name": "kix",
                    "team": "CT",
                    "stats": {"kills": 25, "deaths": 16, "adr": 82.3},
                    "rating": {"hltv_rating": 1.28},
                    "advanced": {},
                    "entry": {},
                    "trades": {},
                    "clutches": {},
                    "utility": {},
                }
            },
            "round_timeline": [],
        }

        mock_job = Job(
            job_id="testjob12345678",
            filename="test.dem",
            size=1000,
            status="completed",
            result=fake_result,
        )

        with mock_patch(
            "opensight.api.routes_match._get_job_store"
        ) as mock_store:
            mock_store.return_value.get_job.return_value = mock_job
            resp = client.post(
                "/api/query",
                json={
                    "question": "What is kix's rating?",
                    "job_id": "testjob12345678",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "What is kix's rating?"
        assert "answer" in body
        assert "data" in body
        assert "confidence" in body

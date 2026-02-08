"""
Tests for AI Game Plan Generator (Prompt 6 — Capstone).

Covers:
  - StratCall, EconomyPlan, GamePlan dataclass construction & to_dict
  - Matchup analysis computation with mock data
  - Economy plan generation (heuristic, no LLM)
  - Team average computation
  - Opponent stats extraction
  - Prompt building produces expected structure
  - LLM response parsing into GamePlan
  - Confidence computation
  - Plan caching
  - API endpoint integration
  - Edge cases: empty data, missing fields, malformed JSON
"""

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Fixtures — realistic orchestrator + scouting data
# =============================================================================

STEAM_ID_1 = "76561198000000001"
STEAM_ID_2 = "76561198000000002"
STEAM_ID_3 = "76561198000000003"
STEAM_ID_4 = "76561198000000004"
STEAM_ID_5 = "76561198000000005"


def _make_player(
    name: str,
    kills: int = 20,
    deaths: int = 15,
    adr: float = 80.0,
    rating: float = 1.10,
    kast: float = 72.0,
    hs_pct: float = 50.0,
    opening_kills: int = 3,
    opening_deaths: int = 2,
    clutch_wins: int = 1,
    clutch_attempts: int = 3,
    trade_kills: int = 4,
    trade_opps: int = 8,
    flash_assists: int = 3,
    he_damage: int = 120,
    flashbangs_thrown: int = 6,
    smokes_thrown: int = 4,
    he_thrown: int = 3,
    molotovs_thrown: int = 2,
    rounds_played: int = 25,
) -> dict:
    return {
        "name": name,
        "team": "TeamAlpha",
        "stats": {
            "kills": kills,
            "deaths": deaths,
            "assists": 5,
            "adr": adr,
            "headshot_pct": hs_pct,
            "rounds_played": rounds_played,
            "total_damage": int(adr * rounds_played),
        },
        "rating": {
            "hltv_rating": rating,
            "kast_percentage": kast,
            "aim_rating": 65.0,
            "utility_rating": 55.0,
            "impact_rating": 1.0,
        },
        "advanced": {
            "ttd_median_ms": 300,
            "cp_median_error_deg": 12.0,
        },
        "duels": {
            "opening_kills": opening_kills,
            "opening_deaths": opening_deaths,
            "clutch_wins": clutch_wins,
            "clutch_attempts": clutch_attempts,
            "trade_kills": trade_kills,
            "trade_kill_opportunities": trade_opps,
        },
        "utility": {
            "flashbangs_thrown": flashbangs_thrown,
            "smokes_thrown": smokes_thrown,
            "he_thrown": he_thrown,
            "molotovs_thrown": molotovs_thrown,
            "flash_assists": flash_assists,
            "he_damage": he_damage,
            "enemies_flashed": 5,
        },
    }


def _make_orchestrator_result(players: dict | None = None) -> dict:
    """Build a realistic orchestrator result dict."""
    if players is None:
        players = {
            STEAM_ID_1: _make_player("foe", kills=22, deaths=14, rating=1.20, adr=85.0),
            STEAM_ID_2: _make_player(
                "kix", kills=25, deaths=12, rating=1.30, adr=90.0, opening_kills=5, opening_deaths=2
            ),
            STEAM_ID_3: _make_player("ace", kills=18, deaths=16, rating=1.05, adr=72.0),
            STEAM_ID_4: _make_player(
                "nova", kills=15, deaths=18, rating=0.90, adr=65.0, flash_assists=6
            ),
            STEAM_ID_5: _make_player(
                "zen", kills=12, deaths=17, rating=0.85, adr=58.0, clutch_wins=3, clutch_attempts=5
            ),
        }
    return {
        "demo_info": {
            "map": "de_ancient",
            "rounds": 25,
            "score_ct": 13,
            "score_t": 12,
            "team1_name": "TeamAlpha",
            "team2_name": "TeamBravo",
        },
        "players": players,
        "round_timeline": [],
    }


def _make_scouting_data() -> dict:
    """Build realistic opponent scouting data (TeamScoutReport.to_dict() format)."""
    return {
        "team_name": "OpponentTeam",
        "demos_analyzed": 3,
        "total_rounds": 75,
        "confidence_level": "medium",
        "maps_analyzed": ["de_ancient", "de_mirage"],
        "players": [
            {
                "name": "StarPlayer",
                "steamid": "765611980000000A1",
                "play_style": "aggressive",
                "aggression_score": 78,
                "consistency_score": 70,
                "avg_kills_per_round": 0.92,
                "avg_deaths_per_round": 0.68,
                "kd_ratio": 1.35,
                "avg_adr": 88.0,
                "avg_kast": 74.0,
                "headshot_rate": 55.0,
                "entry_attempt_rate": 22.0,
                "entry_success_rate": 58.0,
                "opening_duel_win_rate": 60.0,
                "awp_usage_rate": 5.0,
                "awp_kills_per_awp_round": 0.2,
                "clutch_attempts": 4,
                "clutch_wins": 2,
                "clutch_win_rate": 50.0,
                "weapon_preferences": [
                    {"weapon": "ak47", "kills": 30, "usage_rate": 40},
                ],
                "demos_analyzed": 3,
                "rounds_analyzed": 75,
            },
            {
                "name": "AWPGod",
                "steamid": "765611980000000A2",
                "play_style": "passive",
                "aggression_score": 30,
                "consistency_score": 85,
                "avg_kills_per_round": 0.75,
                "avg_deaths_per_round": 0.55,
                "kd_ratio": 1.36,
                "avg_adr": 70.0,
                "avg_kast": 72.0,
                "headshot_rate": 35.0,
                "entry_attempt_rate": 5.0,
                "entry_success_rate": 40.0,
                "opening_duel_win_rate": 55.0,
                "awp_usage_rate": 45.0,
                "awp_kills_per_awp_round": 1.2,
                "clutch_attempts": 2,
                "clutch_wins": 1,
                "clutch_win_rate": 50.0,
                "weapon_preferences": [
                    {"weapon": "awp", "kills": 25, "usage_rate": 45},
                ],
                "demos_analyzed": 3,
                "rounds_analyzed": 75,
            },
            {
                "name": "Support1",
                "steamid": "765611980000000A3",
                "play_style": "passive",
                "aggression_score": 25,
                "consistency_score": 60,
                "avg_kills_per_round": 0.55,
                "avg_deaths_per_round": 0.70,
                "kd_ratio": 0.79,
                "avg_adr": 58.0,
                "avg_kast": 65.0,
                "headshot_rate": 42.0,
                "entry_attempt_rate": 8.0,
                "entry_success_rate": 30.0,
                "opening_duel_win_rate": 40.0,
                "awp_usage_rate": 2.0,
                "awp_kills_per_awp_round": 0.1,
                "clutch_attempts": 1,
                "clutch_wins": 0,
                "clutch_win_rate": 0.0,
                "weapon_preferences": [],
                "demos_analyzed": 3,
                "rounds_analyzed": 75,
            },
        ],
        "map_tendencies": [
            {
                "map_name": "de_ancient",
                "t_side": {
                    "default_setup": "3-1-1 mid control",
                    "common_executes": ["A split", "B main rush"],
                    "aggression": 65.0,
                },
                "ct_side": {
                    "default_setup": "2-1-2",
                    "rotation_speed": "slow",
                    "aggression": 30.0,
                },
                "timing": {
                    "avg_first_contact": 22.5,
                    "avg_execute_time": 35.0,
                },
            },
        ],
        "economy": {
            "tendency": "balanced",
            "force_buy_rate": 25.0,
            "eco_round_rate": 18.0,
        },
        "anti_strats": ["Counter their A split with smoke + molly"],
    }


@pytest.fixture
def your_demos():
    """Two orchestrator results for your team."""
    return [_make_orchestrator_result(), _make_orchestrator_result()]


@pytest.fixture
def opponent_scouting():
    """Opponent scouting data."""
    return _make_scouting_data()


@pytest.fixture
def roster():
    """Team roster with roles."""
    return {
        "foe": "igl",
        "kix": "entry",
        "ace": "rifler",
        "nova": "support",
        "zen": "lurker",
    }


# =============================================================================
# TestStratCall
# =============================================================================


class TestStratCall:
    def test_construction(self):
        from opensight.ai.game_plan import StratCall

        sc = StratCall(
            name="A Split",
            description="Split through main and short",
            utility_sequence=["smoke CT", "flash main"],
            player_assignments={"kix": "entry main", "foe": "flash support"},
            when_to_call="Default round 4+",
            confidence=0.75,
            evidence="Opponent rotates slow (8.2s avg)",
            expected_success_rate="~60%",
        )
        assert sc.name == "A Split"
        assert sc.confidence == 0.75
        assert len(sc.utility_sequence) == 2

    def test_to_dict(self):
        from opensight.ai.game_plan import StratCall

        sc = StratCall(name="B Rush", description="Fast B")
        d = sc.to_dict()
        assert d["name"] == "B Rush"
        assert d["description"] == "Fast B"
        assert isinstance(d["utility_sequence"], list)
        assert isinstance(d["player_assignments"], dict)
        assert d["confidence"] == 0.5  # default

    def test_defaults(self):
        from opensight.ai.game_plan import StratCall

        sc = StratCall(name="test", description="test")
        assert sc.utility_sequence == []
        assert sc.player_assignments == {}
        assert sc.when_to_call == ""
        assert sc.evidence == ""


# =============================================================================
# TestEconomyPlan
# =============================================================================


class TestEconomyPlan:
    def test_construction(self):
        from opensight.ai.game_plan import EconomyPlan

        ep = EconomyPlan(
            pistol_round_buy="kevlar + p250",
            force_buy_threshold=3200,
        )
        assert ep.force_buy_threshold == 3200
        assert ep.pistol_round_buy == "kevlar + p250"

    def test_to_dict(self):
        from opensight.ai.game_plan import EconomyPlan

        ep = EconomyPlan()
        d = ep.to_dict()
        assert "pistol_round_buy" in d
        assert "anti_eco_setup" in d
        assert "force_buy_threshold" in d
        assert "save_triggers" in d
        assert "double_eco_into_full" in d

    def test_defaults(self):
        from opensight.ai.game_plan import EconomyPlan

        ep = EconomyPlan()
        assert ep.force_buy_threshold == 3200
        assert isinstance(ep.save_triggers, list)


# =============================================================================
# TestGamePlan
# =============================================================================


class TestGamePlan:
    def test_construction(self):
        from opensight.ai.game_plan import GamePlan, StratCall

        plan = GamePlan(
            plan_id="test-123",
            opponent="TeamBravo",
            map_name="de_ancient",
            generated_at="2026-02-07T12:00:00",
            confidence_overall=0.75,
            executive_summary="Test plan",
            key_advantage="Strong entry fragger",
            key_risk="Opponent AWPer controls mid",
            ct_default=StratCall(name="2-1-2", description="Standard CT"),
            t_default=StratCall(name="Default", description="Mid control"),
        )
        assert plan.opponent == "TeamBravo"
        assert plan.confidence_overall == 0.75

    def test_to_dict_structure(self):
        from opensight.ai.game_plan import GamePlan, StratCall

        plan = GamePlan(
            plan_id="abc",
            opponent="Opp",
            map_name="de_dust2",
            ct_default=StratCall(name="CT Default", description="Hold"),
            t_default=StratCall(name="T Default", description="Push"),
            t_executes=[StratCall(name="A Exec", description="Rush A")],
            if_losing=["Switch to aggressive"],
            if_winning=["Play standard"],
            opponent_exploits=["Slow rotations"],
        )
        d = plan.to_dict()
        assert d["plan_id"] == "abc"
        assert d["opponent"] == "Opp"
        assert d["ct_side"]["default"]["name"] == "CT Default"
        assert d["t_side"]["default"]["name"] == "T Default"
        assert len(d["t_side"]["executes"]) == 1
        assert d["situational"]["if_losing"] == ["Switch to aggressive"]
        assert d["opponent_exploits"] == ["Slow rotations"]

    def test_to_dict_no_defaults(self):
        from opensight.ai.game_plan import GamePlan

        plan = GamePlan()
        d = plan.to_dict()
        assert d["ct_side"]["default"] is None
        assert d["t_side"]["default"] is None
        assert d["t_side"]["executes"] == []

    def test_economy_plan_in_dict(self):
        from opensight.ai.game_plan import EconomyPlan, GamePlan

        plan = GamePlan(economy_plan=EconomyPlan(force_buy_threshold=2800))
        d = plan.to_dict()
        assert d["economy_plan"]["force_buy_threshold"] == 2800


# =============================================================================
# TestComputeTeamAverages
# =============================================================================


class TestComputeTeamAverages:
    def test_basic(self, your_demos):
        from opensight.ai.game_plan import _compute_team_averages

        avgs = _compute_team_averages(your_demos)
        assert avgs["avg_rating"] > 0
        assert avgs["avg_adr"] > 0
        assert 0 <= avgs["opening_duel_win_rate"] <= 1
        assert 0 <= avgs["trade_success_rate"] <= 1

    def test_empty_demos(self):
        from opensight.ai.game_plan import _compute_team_averages

        avgs = _compute_team_averages([])
        assert avgs["kills"] == 0

    def test_empty_players(self):
        from opensight.ai.game_plan import _compute_team_averages

        result = {"players": {}, "demo_info": {}}
        avgs = _compute_team_averages([result])
        assert avgs["kills"] == 0

    def test_single_player(self):
        from opensight.ai.game_plan import _compute_team_averages

        result = _make_orchestrator_result(
            {
                STEAM_ID_1: _make_player("solo", kills=30, deaths=10, rating=1.50, adr=100.0),
            }
        )
        avgs = _compute_team_averages([result])
        assert avgs["avg_kills"] == 30
        assert avgs["avg_rating"] == 1.50
        assert avgs["avg_adr"] == 100.0


# =============================================================================
# TestExtractOpponentStats
# =============================================================================


class TestExtractOpponentStats:
    def test_basic(self, opponent_scouting):
        from opensight.ai.game_plan import _extract_opponent_stats

        stats = _extract_opponent_stats(opponent_scouting)
        assert stats["avg_kpr"] > 0
        assert stats["avg_adr"] > 0
        assert stats["force_buy_rate"] == 25.0

    def test_empty_players(self):
        from opensight.ai.game_plan import _extract_opponent_stats

        stats = _extract_opponent_stats({"players": []})
        assert stats == {}

    def test_missing_players(self):
        from opensight.ai.game_plan import _extract_opponent_stats

        stats = _extract_opponent_stats({})
        assert stats == {}

    def test_economy_data(self, opponent_scouting):
        from opensight.ai.game_plan import _extract_opponent_stats

        stats = _extract_opponent_stats(opponent_scouting)
        assert stats["eco_round_rate"] == 18.0


# =============================================================================
# TestBuildMatchupAnalysis
# =============================================================================


class TestBuildMatchupAnalysis:
    def test_produces_structured_text(self, your_demos, opponent_scouting):
        from opensight.ai.game_plan import build_matchup_analysis

        text = build_matchup_analysis(your_demos, opponent_scouting)
        assert "<matchup_analysis>" in text
        assert "</matchup_analysis>" in text

    def test_contains_key_sections(self, your_demos, opponent_scouting):
        from opensight.ai.game_plan import build_matchup_analysis

        text = build_matchup_analysis(your_demos, opponent_scouting)
        # Should mention opening duels
        assert "Opening duels" in text or "opening duels" in text.lower()
        # Should mention trade
        assert "Trade" in text or "trade" in text.lower()
        # Should mention utility
        assert "Utility" in text or "utility" in text.lower()

    def test_identifies_high_threat_players(self, your_demos, opponent_scouting):
        from opensight.ai.game_plan import build_matchup_analysis

        text = build_matchup_analysis(your_demos, opponent_scouting)
        # StarPlayer has 0.92 KPR — should be flagged
        assert "StarPlayer" in text
        assert "HIGH-THREAT" in text

    def test_empty_your_data(self, opponent_scouting):
        from opensight.ai.game_plan import build_matchup_analysis

        text = build_matchup_analysis([], opponent_scouting)
        assert "Insufficient data" in text

    def test_empty_opponent_data(self, your_demos):
        from opensight.ai.game_plan import build_matchup_analysis

        text = build_matchup_analysis(your_demos, {})
        assert "Insufficient data" in text

    def test_economy_exploit_detected(self):
        from opensight.ai.game_plan import build_matchup_analysis

        # High force buy rate should be flagged
        opp = _make_scouting_data()
        opp["economy"]["force_buy_rate"] = 40.0
        your = [_make_orchestrator_result()]
        text = build_matchup_analysis(your, opp)
        assert "EXPLOIT" in text
        assert "40%" in text


# =============================================================================
# TestBuildEconomyPlan
# =============================================================================


class TestBuildEconomyPlan:
    def test_basic(self, your_demos, opponent_scouting):
        from opensight.ai.game_plan import _build_economy_plan

        plan = _build_economy_plan(your_demos, opponent_scouting)
        assert plan.force_buy_threshold > 0
        assert len(plan.save_triggers) >= 3
        assert plan.double_eco_into_full != ""

    def test_aggressive_opponent(self, your_demos):
        from opensight.ai.game_plan import _build_economy_plan

        opp = _make_scouting_data()
        opp["economy"]["force_buy_rate"] = 40.0
        plan = _build_economy_plan(your_demos, opp)
        # Should mention opponent force-buying in save triggers
        triggers_text = " ".join(plan.save_triggers)
        assert "force" in triggers_text.lower()

    def test_high_eco_opponent(self, your_demos):
        from opensight.ai.game_plan import _build_economy_plan

        opp = _make_scouting_data()
        opp["economy"]["eco_round_rate"] = 30.0
        plan = _build_economy_plan(your_demos, opp)
        assert "eco" in plan.anti_eco_setup.lower()

    def test_to_dict(self, your_demos, opponent_scouting):
        from opensight.ai.game_plan import _build_economy_plan

        plan = _build_economy_plan(your_demos, opponent_scouting)
        d = plan.to_dict()
        assert "force_buy_threshold" in d
        assert isinstance(d["save_triggers"], list)

    def test_missing_economy_data(self, your_demos):
        from opensight.ai.game_plan import _build_economy_plan

        plan = _build_economy_plan(your_demos, {})
        assert plan.force_buy_threshold > 0


# =============================================================================
# TestBuildGamePlanPrompt
# =============================================================================


class TestBuildGamePlanPrompt:
    def test_contains_sections(self, roster):
        from opensight.ai.game_plan import _build_game_plan_prompt

        prompt = _build_game_plan_prompt(
            matchup_analysis="<matchup_analysis>Test</matchup_analysis>",
            opponent_scouting_text="Opponent data",
            your_team_text="Your team data",
            map_name="de_ancient",
            roster=roster,
        )
        assert "<your_team>" in prompt
        assert "<opponent>" in prompt
        assert "<task>" in prompt
        assert "de_ancient" in prompt

    def test_roster_names_in_prompt(self, roster):
        from opensight.ai.game_plan import _build_game_plan_prompt

        prompt = _build_game_plan_prompt(
            matchup_analysis="test",
            opponent_scouting_text="test",
            your_team_text="test",
            map_name="de_dust2",
            roster=roster,
        )
        for name in roster:
            assert name in prompt

    def test_includes_matchup(self, roster):
        from opensight.ai.game_plan import _build_game_plan_prompt

        prompt = _build_game_plan_prompt(
            matchup_analysis="<matchup_analysis>ADVANTAGE — Opening duels</matchup_analysis>",
            opponent_scouting_text="test",
            your_team_text="test",
            map_name="de_mirage",
            roster=roster,
        )
        assert "ADVANTAGE — Opening duels" in prompt


# =============================================================================
# TestBuildYourTeamSummary
# =============================================================================


class TestBuildYourTeamSummary:
    def test_basic(self, your_demos):
        from opensight.ai.game_plan import _build_your_team_summary

        text = _build_your_team_summary(your_demos)
        assert "Average Rating" in text
        assert "Average ADR" in text
        assert "Player Breakdown" in text

    def test_empty(self):
        from opensight.ai.game_plan import _build_your_team_summary

        text = _build_your_team_summary([])
        assert "Average Rating" in text

    def test_player_names_present(self, your_demos):
        from opensight.ai.game_plan import _build_your_team_summary

        text = _build_your_team_summary(your_demos)
        # At least one player name should be present
        assert any(name in text for name in ["foe", "kix", "ace", "nova", "zen"])


# =============================================================================
# TestConfidenceComputation
# =============================================================================


class TestConfidenceComputation:
    def test_full_confidence(self):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        conf = gen._compute_confidence(your_demos=5, opponent_demos=4)
        assert conf == 1.0

    def test_zero_demos(self):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        conf = gen._compute_confidence(your_demos=0, opponent_demos=0)
        assert conf == 0.0

    def test_partial_your_demos(self):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        conf = gen._compute_confidence(your_demos=2, opponent_demos=4)
        # 2/5 * 0.5 + 4/4 * 0.5 = 0.2 + 0.5 = 0.7
        assert abs(conf - 0.7) < 0.01

    def test_partial_opponent_demos(self):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        conf = gen._compute_confidence(your_demos=5, opponent_demos=2)
        # 5/5 * 0.5 + 2/4 * 0.5 = 0.5 + 0.25 = 0.75
        assert abs(conf - 0.75) < 0.01

    def test_excess_demos_capped(self):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        conf = gen._compute_confidence(your_demos=100, opponent_demos=100)
        assert conf == 1.0


# =============================================================================
# TestParseStratCall
# =============================================================================


class TestParseStratCall:
    def test_basic(self):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        sc = gen._parse_strat_call(
            {
                "name": "B Rush",
                "description": "Fast B take",
                "utility_sequence": ["flash B main", "smoke CT"],
                "player_assignments": {"kix": "entry", "foe": "support"},
                "when_to_call": "After eco win",
                "confidence": 0.85,
                "evidence": "Opponent rotates slow",
                "expected_success_rate": "~70%",
            }
        )
        assert sc.name == "B Rush"
        assert sc.confidence == 0.85
        assert len(sc.utility_sequence) == 2
        assert sc.player_assignments["kix"] == "entry"

    def test_missing_fields(self):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        sc = gen._parse_strat_call({})
        assert sc.name == ""
        assert sc.confidence == 0.5  # default


# =============================================================================
# TestPopulatePlanFromLLM
# =============================================================================


class TestPopulatePlanFromLLM:
    def _make_llm_response(self) -> str:
        import json

        return json.dumps(
            {
                "executive_summary": "Strong plan for de_ancient",
                "key_advantage": "Entry fragger dominance",
                "key_risk": "AWPer controls mid",
                "ct_default": {
                    "name": "2-1-2 Default",
                    "description": "Standard CT hold",
                    "confidence": 0.85,
                    "evidence": "Opponent splits A 60% of rounds",
                },
                "ct_adjustments": [
                    {
                        "name": "B Stack",
                        "description": "3 B when they rush",
                        "when_to_call": "After 2 B rushes",
                        "confidence": 0.70,
                    },
                ],
                "ct_retake_priorities": {"A": "Rotate 2 from B", "B": "3 from A retake"},
                "t_default": {
                    "name": "Mid Control",
                    "description": "Take mid first",
                    "confidence": 0.80,
                },
                "t_executes": [
                    {
                        "name": "A Split",
                        "description": "Split through A main + short",
                        "confidence": 0.75,
                        "utility_sequence": ["smoke CT", "flash main"],
                    },
                    {
                        "name": "B Rush",
                        "description": "Fast B with utility",
                        "confidence": 0.65,
                    },
                ],
                "t_read_based": [
                    {
                        "name": "Late B",
                        "description": "If A is smoked, rotate B late",
                        "confidence": 0.60,
                    },
                ],
                "player_roles": {"foe": "IGL/anchor B", "kix": "entry A"},
                "player_matchups": ["Put kix vs their weakest player on A"],
                "if_losing": ["Down 0-3: switch to aggressive T pushes"],
                "if_winning": ["Up 5-0: play standard, don't force"],
                "timeout_triggers": ["After 3-round losing streak"],
                "opponent_exploits": ["Slow CT rotations — execute fast"],
            }
        )

    def test_full_parse(self):
        from opensight.ai.game_plan import GamePlan, GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        plan = GamePlan()
        gen._populate_plan_from_llm(plan, self._make_llm_response(), {"foe": "igl"})

        assert plan.executive_summary == "Strong plan for de_ancient"
        assert plan.key_advantage == "Entry fragger dominance"
        assert plan.key_risk == "AWPer controls mid"
        assert plan.ct_default is not None
        assert plan.ct_default.name == "2-1-2 Default"
        assert len(plan.ct_adjustments) == 1
        assert plan.t_default is not None
        assert len(plan.t_executes) == 2
        assert len(plan.t_read_based) == 1
        assert len(plan.if_losing) == 1
        assert len(plan.opponent_exploits) == 1

    def test_markdown_fenced_json(self):
        from opensight.ai.game_plan import GamePlan, GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        plan = GamePlan()
        fenced = "```json\n" + self._make_llm_response() + "\n```"
        gen._populate_plan_from_llm(plan, fenced, {"foe": "igl"})
        assert plan.executive_summary == "Strong plan for de_ancient"

    def test_invalid_json(self):
        from opensight.ai.game_plan import GamePlan, GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        plan = GamePlan()
        gen._populate_plan_from_llm(plan, "not valid json at all", {"foe": "igl"})
        assert "JSON parse error" in plan.generation_error
        # Falls back to putting raw text in summary
        assert plan.executive_summary != ""

    def test_partial_json(self):
        import json

        from opensight.ai.game_plan import GamePlan, GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        plan = GamePlan()
        partial = json.dumps(
            {
                "executive_summary": "Partial plan",
                "key_advantage": "Test",
            }
        )
        gen._populate_plan_from_llm(plan, partial, {"foe": "igl"})
        assert plan.executive_summary == "Partial plan"
        assert plan.ct_default is None  # not provided


# =============================================================================
# TestGenerateNoApiKey
# =============================================================================


class TestGenerateNoApiKey:
    def test_returns_plan_without_llm(self, your_demos, opponent_scouting, roster):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        plan = gen.generate(
            your_team_demos=your_demos,
            opponent_scouting=opponent_scouting,
            map_name="de_ancient",
            roster=roster,
        )

        assert plan.opponent == "OpponentTeam"
        assert plan.map_name == "de_ancient"
        assert "ANTHROPIC_API_KEY not configured" in plan.generation_error
        assert plan.economy_plan.force_buy_threshold > 0
        assert plan.player_roles == roster
        assert plan.plan_id != ""
        assert plan.generated_at != ""

    def test_confidence_computed(self, your_demos, opponent_scouting, roster):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        plan = gen.generate(
            your_team_demos=your_demos,
            opponent_scouting=opponent_scouting,
            map_name="de_mirage",
            roster=roster,
        )
        # 2 demos / 5 * 0.5 + 3 demos / 4 * 0.5 = 0.2 + 0.375 = 0.575
        assert 0.5 < plan.confidence_overall < 0.6

    def test_empty_demos(self, opponent_scouting, roster):
        from opensight.ai.game_plan import GamePlanGenerator

        gen = GamePlanGenerator(api_key=None)
        plan = gen.generate(
            your_team_demos=[],
            opponent_scouting=opponent_scouting,
            map_name="de_dust2",
            roster=roster,
        )
        assert plan.confidence_overall < 0.5  # low confidence


# =============================================================================
# TestPlanCaching
# =============================================================================


class TestPlanCaching:
    def test_cache_and_retrieve(self):
        from opensight.ai.game_plan import GamePlan, cache_plan, get_cached_plan

        plan = GamePlan(plan_id="cache-test-1", opponent="TestTeam")
        cache_plan(plan)
        retrieved = get_cached_plan("cache-test-1")
        assert retrieved is not None
        assert retrieved.opponent == "TestTeam"

    def test_missing_plan(self):
        from opensight.ai.game_plan import get_cached_plan

        assert get_cached_plan("nonexistent-id") is None

    def test_cache_limit(self):
        from opensight.ai.game_plan import GamePlan, _plan_cache, cache_plan

        # Clear and fill cache
        _plan_cache.clear()
        for i in range(55):
            cache_plan(GamePlan(plan_id=f"limit-test-{i}"))
        assert len(_plan_cache) <= 50


# =============================================================================
# TestSafeDiv
# =============================================================================


class TestSafeDiv:
    def test_normal(self):
        from opensight.ai.game_plan import _safe_div

        assert _safe_div(10, 5) == 2.0

    def test_zero_denominator(self):
        from opensight.ai.game_plan import _safe_div

        assert _safe_div(10, 0) == 0.0

    def test_custom_default(self):
        from opensight.ai.game_plan import _safe_div

        assert _safe_div(10, 0, default=1.0) == 1.0


# =============================================================================
# TestGamePlanAPI
# =============================================================================


class TestGamePlanAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from opensight.api import app

        return TestClient(app)

    def test_get_plan_not_found(self, client):
        resp = client.get("/api/game-plan/nonexistent-id")
        assert resp.status_code == 404

    def test_get_cached_plan(self, client):
        from opensight.ai.game_plan import GamePlan, cache_plan

        plan = GamePlan(
            plan_id="api-test-get",
            opponent="TestTeam",
            map_name="de_dust2",
        )
        cache_plan(plan)

        resp = client.get("/api/game-plan/api-test-get")
        assert resp.status_code == 200
        data = resp.json()
        assert data["plan_id"] == "api-test-get"
        assert data["opponent"] == "TestTeam"

    def test_generate_no_job_ids(self, client):
        resp = client.post(
            "/api/game-plan/generate",
            json={
                "your_team_job_ids": [],
                "opponent_scouting": _make_scouting_data(),
                "map": "de_ancient",
                "roster": {"foe": "igl"},
            },
        )
        assert resp.status_code == 400
        assert "At least one job ID" in resp.json()["detail"]

    def test_generate_no_roster(self, client):
        resp = client.post(
            "/api/game-plan/generate",
            json={
                "your_team_job_ids": ["some-id"],
                "opponent_scouting": _make_scouting_data(),
                "map": "de_ancient",
                "roster": {},
            },
        )
        assert resp.status_code == 400
        assert "Roster" in resp.json()["detail"]

    def test_generate_job_not_found(self, client):
        resp = client.post(
            "/api/game-plan/generate",
            json={
                "your_team_job_ids": ["nonexistent-job-id"],
                "opponent_scouting": _make_scouting_data(),
                "map": "de_ancient",
                "roster": {"foe": "igl"},
            },
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    @patch("opensight.api.routes_match._get_job_store")
    def test_generate_job_not_completed(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "processing"
        mock_job.result = None
        mock_store.get_job.return_value = mock_job
        mock_get_store.return_value = mock_store

        resp = client.post(
            "/api/game-plan/generate",
            json={
                "your_team_job_ids": ["job-123"],
                "opponent_scouting": _make_scouting_data(),
                "map": "de_ancient",
                "roster": {"foe": "igl"},
            },
        )
        assert resp.status_code == 400
        assert "not completed" in resp.json()["detail"]

    @patch("opensight.api.routes_match._get_job_store")
    def test_generate_success_no_api_key(self, mock_get_store, client):
        """Full generate flow without API key — returns data-only plan."""
        mock_store = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.result = _make_orchestrator_result()
        mock_store.get_job.return_value = mock_job
        mock_get_store.return_value = mock_store

        resp = client.post(
            "/api/game-plan/generate",
            json={
                "your_team_job_ids": ["job-abc"],
                "opponent_scouting": _make_scouting_data(),
                "map": "de_ancient",
                "roster": {"foe": "igl", "kix": "entry"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["opponent"] == "OpponentTeam"
        assert data["map_name"] == "de_ancient"
        assert data["plan_id"] != ""
        assert "ANTHROPIC_API_KEY" in data["generation_error"]
        assert data["economy_plan"]["force_buy_threshold"] > 0
        assert data["player_roles"]["foe"] == "igl"

"""
Tests for AI Anti-Strat Report Generator.

Tests the full pipeline:
  - Data extraction from TeamScoutReport.to_dict()
  - Player threat classification (role, threat level)
  - Scouting prompt construction
  - LLM response parsing into structured report
  - API endpoint integration
  - Edge cases (empty data, missing fields, malformed JSON)
"""

import json
from unittest.mock import Mock, patch

import pytest

# =============================================================================
# Fixtures — realistic scouting data matching TeamScoutReport.to_dict() format
# =============================================================================


@pytest.fixture
def scout_report_dict():
    """
    Realistic TeamScoutReport.to_dict() output with 3 players.

    All values match the exact format from PlayerScoutProfile.to_dict():
    - Rates are already multiplied by 100 (e.g., headshot_rate: 48.5 means 48.5%)
    - KPR/DPR are raw per-round values
    """
    return {
        "team_name": "TestTeam",
        "demos_analyzed": 3,
        "total_rounds": 87,
        "confidence_level": "medium",
        "maps_analyzed": ["de_mirage", "de_inferno"],
        "players": [
            {
                "steamid": "76561198000001",
                "name": "StarPlayer",
                "play_style": "aggressive",
                "aggression_score": 72.5,
                "consistency_score": 85.0,
                "avg_kills_per_round": 0.92,
                "avg_deaths_per_round": 0.55,
                "kd_ratio": 1.67,
                "avg_adr": 88.3,
                "avg_kast": 76.2,
                "headshot_rate": 48.5,
                "entry_attempt_rate": 22.0,
                "entry_success_rate": 58.0,
                "opening_duel_win_rate": 60.0,
                "awp_usage_rate": 5.0,
                "awp_kills_per_awp_round": 0.5,
                "avg_first_kill_time_seconds": 15.2,
                "avg_rotation_time_seconds": 0,
                "favorite_positions": {},
                "weapon_preferences": [
                    {"weapon": "AK-47", "usage_rate": 45.0, "kills": 38},
                    {"weapon": "M4A1-S", "usage_rate": 20.0, "kills": 16},
                    {"weapon": "Desert Eagle", "usage_rate": 10.0, "kills": 8},
                ],
                "clutch_attempts": 6,
                "clutch_wins": 3,
                "clutch_win_rate": 50.0,
                "demos_analyzed": 3,
                "rounds_analyzed": 87,
            },
            {
                "steamid": "76561198000002",
                "name": "AWPGod",
                "play_style": "passive",
                "aggression_score": 35.0,
                "consistency_score": 70.0,
                "avg_kills_per_round": 0.78,
                "avg_deaths_per_round": 0.52,
                "kd_ratio": 1.50,
                "avg_adr": 72.1,
                "avg_kast": 71.0,
                "headshot_rate": 22.0,
                "entry_attempt_rate": 8.0,
                "entry_success_rate": 40.0,
                "opening_duel_win_rate": 55.0,
                "awp_usage_rate": 45.0,
                "awp_kills_per_awp_round": 1.3,
                "avg_first_kill_time_seconds": 22.0,
                "avg_rotation_time_seconds": 0,
                "favorite_positions": {},
                "weapon_preferences": [
                    {"weapon": "AWP", "usage_rate": 55.0, "kills": 28},
                    {"weapon": "USP-S", "usage_rate": 15.0, "kills": 8},
                ],
                "clutch_attempts": 4,
                "clutch_wins": 1,
                "clutch_win_rate": 25.0,
                "demos_analyzed": 3,
                "rounds_analyzed": 87,
            },
            {
                "steamid": "76561198000003",
                "name": "Struggler",
                "play_style": "mixed",
                "aggression_score": 45.0,
                "consistency_score": 40.0,
                "avg_kills_per_round": 0.48,
                "avg_deaths_per_round": 0.68,
                "kd_ratio": 0.71,
                "avg_adr": 55.2,
                "avg_kast": 58.0,
                "headshot_rate": 35.0,
                "entry_attempt_rate": 5.0,
                "entry_success_rate": 30.0,
                "opening_duel_win_rate": 40.0,
                "awp_usage_rate": 2.0,
                "awp_kills_per_awp_round": 0.0,
                "avg_first_kill_time_seconds": 30.0,
                "avg_rotation_time_seconds": 0,
                "favorite_positions": {},
                "weapon_preferences": [
                    {"weapon": "M4A4", "usage_rate": 40.0, "kills": 18},
                ],
                "clutch_attempts": 2,
                "clutch_wins": 0,
                "clutch_win_rate": 0.0,
                "demos_analyzed": 3,
                "rounds_analyzed": 87,
            },
        ],
        "map_tendencies": [
            {
                "map_name": "de_mirage",
                "demos_analyzed": 2,
                "rounds_analyzed": 58,
                "t_side": {
                    "default_setup": "Standard default with lurk",
                    "common_executes": ["A execute", "B split"],
                    "aggression": 65.0,
                },
                "ct_side": {
                    "default_setup": "Standard 2-1-2",
                    "rotation_speed": "medium",
                    "aggression": 40.0,
                },
                "timing": {
                    "avg_first_contact": 18.5,
                    "avg_execute_time": 45.0,
                },
            },
        ],
        "economy": {
            "tendency": "aggressive",
            "force_buy_rate": 35.0,
            "eco_round_rate": 15.0,
        },
        "anti_strats": ["Some existing anti-strat"],
    }


@pytest.fixture
def mock_llm_response():
    """Realistic LLM JSON response for anti-strat generation."""
    return json.dumps(
        {
            "player_assessments": [
                {
                    "name": "StarPlayer",
                    "role": "entry",
                    "threat_level": "high",
                    "counter_strategy": "StarPlayer wins 58% of entries with 0.92 KPR. Set up crossfire on his peek angles. Use flash + off-angle combo.",
                    "key_weakness": "Aggressive style (72.5 aggression) makes him predictable. Bait his peeks with jiggle and punish with trade.",
                },
                {
                    "name": "AWPGod",
                    "role": "awper",
                    "threat_level": "high",
                    "counter_strategy": "AWPGod uses AWP 45% of rounds with 1.3 kills per AWP round. Smoke common angles and double-peek.",
                    "key_weakness": "Passive style (35 aggression) means predictable positioning. Flash and rush his angles.",
                },
                {
                    "name": "Struggler",
                    "role": "support",
                    "threat_level": "low",
                    "counter_strategy": "Weakest link at 0.71 K/D. Target his site on CT side for easy entry.",
                    "key_weakness": "Low consistency (40) and 0.48 KPR. Struggle to hold under pressure.",
                },
            ],
            "map_counters": [
                {
                    "map_name": "de_mirage",
                    "opponent_tendency": "65% T-side aggression with early contact at 18.5s. Common A execute and B split.",
                    "counter_plan": "Hold passive angles on CT. Delay utility until 1:15. Stack B when they show A presence.",
                    "required_utility": ["Smoke A ramp", "Flash connector", "Molly palace"],
                    "key_positions": ["Stairs", "Jungle", "B Van"],
                },
            ],
            "economy_exploits": [
                {
                    "pattern": "Aggressive economy — 35% force buy rate",
                    "exploit": "Expect upgraded pistols on eco rounds. Hold longer angles where pistols are weak.",
                    "trigger_rounds": "After pistol loss (rounds 2-3, 14-15) and after gun round losses",
                },
            ],
            "t_side_game_plan": "Default with mid control. Exploit Struggler's site. Fast executes to beat AWPGod's positioning.",
            "ct_side_game_plan": "Passive setup to punish StarPlayer's aggression. Crossfire B site, stack A against their A execute.",
            "pistol_round_plan": "Buy armor + P250. Rush the site Struggler holds for quick plant.",
            "anti_eco_plan": "Hold long angles with rifles. They force buy 35% — expect deagles and SMGs.",
            "veto_recommendation": "Ban de_inferno (only 1 demo). Pick de_mirage where we have 2 demos of data.",
        }
    )


# =============================================================================
# Test: Dataclass Construction
# =============================================================================


class TestAntiStratDataclasses:
    """Test that dataclasses can be constructed and serialized."""

    def test_player_threat_construction(self):
        from opensight.ai.antistrat_report import PlayerThreat

        pt = PlayerThreat(
            name="Test",
            steamid="123",
            role="entry",
            threat_level="high",
            avg_kills_per_round=0.85,
            entry_success_rate=0.55,
            awp_usage_rate=0.05,
            clutch_win_rate=0.40,
            headshot_rate=0.45,
            play_style="aggressive",
        )
        assert pt.name == "Test"
        assert pt.threat_level == "high"
        assert pt.counter_strategy == ""  # Not yet populated

    def test_map_counter_construction(self):
        from opensight.ai.antistrat_report import MapCounter

        mc = MapCounter(
            map_name="de_mirage",
            opponent_tendency="High T aggression",
            counter_plan="Hold passive",
            required_utility=["Smoke A ramp"],
            key_positions=["Stairs"],
        )
        assert mc.map_name == "de_mirage"
        assert len(mc.required_utility) == 1

    def test_economy_exploit_construction(self):
        from opensight.ai.antistrat_report import EconomyExploit

        ee = EconomyExploit(
            pattern="Force buy 35%",
            exploit="Hold long angles",
            trigger_rounds="After pistol loss",
        )
        assert ee.pattern == "Force buy 35%"

    def test_antistrat_report_to_dict(self, scout_report_dict):
        from opensight.ai.antistrat_report import AntiStratReport, PlayerThreat

        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=[
                PlayerThreat(
                    name="Star",
                    steamid="123",
                    role="entry",
                    threat_level="high",
                    avg_kills_per_round=0.9,
                    entry_success_rate=0.55,
                    awp_usage_rate=0.05,
                    clutch_win_rate=0.4,
                    headshot_rate=0.48,
                    play_style="aggressive",
                    counter_strategy="Set up crossfire",
                    key_weakness="Predictable aggression",
                )
            ],
            t_side_game_plan="Fast A exec",
            ct_side_game_plan="Passive B hold",
        )

        d = report.to_dict()
        assert d["team_name"] == "TestTeam"
        assert d["demos_analyzed"] == 3
        assert len(d["player_threats"]) == 1
        assert d["player_threats"][0]["counter_strategy"] == "Set up crossfire"
        assert d["game_plans"]["t_side"] == "Fast A exec"
        assert d["game_plans"]["ct_side"] == "Passive B hold"


# =============================================================================
# Test: Player Threat Extraction (Data-Only, No LLM)
# =============================================================================


class TestPlayerThreatExtraction:
    """Test _extract_player_threats — pure data processing from scouting stats."""

    def test_extracts_correct_player_count(self, scout_report_dict):
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        assert len(threats) == 3

    def test_entry_fragger_classified(self, scout_report_dict):
        """StarPlayer has 22% entry rate → should be classified as entry."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)

        star = next(t for t in threats if t.name == "StarPlayer")
        assert star.role == "entry"

    def test_awper_classified(self, scout_report_dict):
        """AWPGod has 45% AWP usage → should be classified as awper."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)

        awper = next(t for t in threats if t.name == "AWPGod")
        assert awper.role == "awper"

    def test_support_classified(self, scout_report_dict):
        """Struggler has low entry rate + low KPR → should be support."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)

        support = next(t for t in threats if t.name == "Struggler")
        assert support.role == "support"

    def test_high_threat_star_player(self, scout_report_dict):
        """StarPlayer has 1.67 K/D and 0.92 KPR → high threat."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)

        star = next(t for t in threats if t.name == "StarPlayer")
        assert star.threat_level == "high"

    def test_low_threat_struggler(self, scout_report_dict):
        """Struggler has 0.71 K/D and 0.48 KPR → low threat."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)

        struggler = next(t for t in threats if t.name == "Struggler")
        assert struggler.threat_level == "low"

    def test_rates_converted_from_percentage(self, scout_report_dict):
        """to_dict() returns rates as percentages; PlayerThreat stores as 0-1."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)

        star = next(t for t in threats if t.name == "StarPlayer")
        # entry_success_rate in dict is 58.0 (percentage), should be 0.58
        assert abs(star.entry_success_rate - 0.58) < 0.01
        # headshot_rate in dict is 48.5 (percentage), should be 0.485
        assert abs(star.headshot_rate - 0.485) < 0.01
        # awp_usage_rate in dict is 5.0 (percentage), should be 0.05
        assert abs(star.awp_usage_rate - 0.05) < 0.01

    def test_play_style_preserved(self, scout_report_dict):
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)

        star = next(t for t in threats if t.name == "StarPlayer")
        assert star.play_style == "aggressive"

        awper = next(t for t in threats if t.name == "AWPGod")
        assert awper.play_style == "passive"

    def test_empty_players_list(self):
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats({"players": []})
        assert threats == []

    def test_missing_players_key(self):
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats({})
        assert threats == []


# =============================================================================
# Test: Scouting Prompt Construction
# =============================================================================


class TestScoutingPrompt:
    """Test _build_scouting_prompt — data → LLM prompt text."""

    def test_prompt_contains_team_name(self, scout_report_dict):
        from opensight.ai.antistrat_report import _build_scouting_prompt

        prompt = _build_scouting_prompt(scout_report_dict)
        assert "TestTeam" in prompt

    def test_prompt_contains_player_stats(self, scout_report_dict):
        from opensight.ai.antistrat_report import _build_scouting_prompt

        prompt = _build_scouting_prompt(scout_report_dict)
        assert "StarPlayer" in prompt
        assert "0.92" in prompt  # KPR
        assert "88.3" in prompt  # ADR
        assert "AWPGod" in prompt
        assert "45.0" in prompt  # AWP usage rate

    def test_prompt_contains_map_tendencies(self, scout_report_dict):
        from opensight.ai.antistrat_report import _build_scouting_prompt

        prompt = _build_scouting_prompt(scout_report_dict)
        assert "de_mirage" in prompt
        assert "18.5" in prompt  # first contact time
        assert "65.0" in prompt  # T-side aggression

    def test_prompt_contains_economy(self, scout_report_dict):
        from opensight.ai.antistrat_report import _build_scouting_prompt

        prompt = _build_scouting_prompt(scout_report_dict)
        assert "aggressive" in prompt.lower()
        assert "35.0" in prompt  # force buy rate

    def test_prompt_contains_weapon_preferences(self, scout_report_dict):
        from opensight.ai.antistrat_report import _build_scouting_prompt

        prompt = _build_scouting_prompt(scout_report_dict)
        assert "AK-47" in prompt
        assert "AWP" in prompt

    def test_prompt_contains_clutch_stats(self, scout_report_dict):
        from opensight.ai.antistrat_report import _build_scouting_prompt

        prompt = _build_scouting_prompt(scout_report_dict)
        assert "3/6" in prompt  # StarPlayer clutches
        assert "50.0%" in prompt  # clutch rate

    def test_empty_report_doesnt_crash(self):
        from opensight.ai.antistrat_report import _build_scouting_prompt

        prompt = _build_scouting_prompt({})
        assert "Unknown" in prompt  # default team name


# =============================================================================
# Test: LLM Response Parsing
# =============================================================================


class TestLLMResponseParsing:
    """Test _populate_report_from_llm — JSON parsing into structured report."""

    def test_valid_json_parsed(self, scout_report_dict, mock_llm_response):
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )

        gen._populate_report_from_llm(report, mock_llm_response)

        # Player counter-strategies populated
        star = next(t for t in report.player_threats if t.name == "StarPlayer")
        assert "crossfire" in star.counter_strategy.lower()
        assert star.key_weakness != ""

        awper = next(t for t in report.player_threats if t.name == "AWPGod")
        assert "awp" in awper.counter_strategy.lower()

    def test_map_counters_populated(self, scout_report_dict, mock_llm_response):
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )

        gen._populate_report_from_llm(report, mock_llm_response)

        assert len(report.map_counters) == 1
        mc = report.map_counters[0]
        assert mc.map_name == "de_mirage"
        assert len(mc.required_utility) > 0
        assert len(mc.key_positions) > 0

    def test_economy_exploits_populated(self, scout_report_dict, mock_llm_response):
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )

        gen._populate_report_from_llm(report, mock_llm_response)

        assert len(report.economy_exploits) == 1
        ee = report.economy_exploits[0]
        assert "35%" in ee.pattern

    def test_game_plans_populated(self, scout_report_dict, mock_llm_response):
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )

        gen._populate_report_from_llm(report, mock_llm_response)

        assert report.t_side_game_plan != ""
        assert report.ct_side_game_plan != ""
        assert report.pistol_round_plan != ""
        assert report.anti_eco_plan != ""
        assert report.veto_recommendation != ""

    def test_markdown_fenced_json_parsed(self, scout_report_dict, mock_llm_response):
        """LLM sometimes wraps JSON in ```json fencing."""
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )

        fenced = f"```json\n{mock_llm_response}\n```"
        gen._populate_report_from_llm(report, fenced)

        # Should still parse correctly
        assert report.t_side_game_plan != ""
        assert len(report.map_counters) == 1

    def test_invalid_json_fallback(self, scout_report_dict):
        """Malformed JSON falls back to raw text in t_side_game_plan."""
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )

        gen._populate_report_from_llm(report, "This is not JSON at all")

        assert "JSON parse error" in report.generation_error
        assert report.t_side_game_plan == "This is not JSON at all"

    def test_role_updated_from_llm(self, scout_report_dict, mock_llm_response):
        """LLM can override role classification if it has better context."""
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )

        gen._populate_report_from_llm(report, mock_llm_response)

        # LLM says StarPlayer is "entry" which matches data classification
        star = next(t for t in report.player_threats if t.name == "StarPlayer")
        assert star.role == "entry"


# =============================================================================
# Test: Full Generate Pipeline (Mocked LLM)
# =============================================================================


class TestGeneratePipeline:
    """Test the full generate() method with mocked Anthropic client."""

    def test_generate_without_api_key(self, scout_report_dict):
        """Without API key, report has data-only threats but no LLM plans."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        report = gen.generate(scout_report_dict)

        assert report.team_name == "TestTeam"
        assert report.demos_analyzed == 3
        assert report.confidence_level == "medium"
        assert len(report.player_threats) == 3
        assert "not configured" in report.generation_error

        # Data-grounded fields populated
        star = next(t for t in report.player_threats if t.name == "StarPlayer")
        assert star.role == "entry"
        assert star.threat_level == "high"

        # LLM fields empty
        assert star.counter_strategy == ""
        assert report.t_side_game_plan == ""

    def test_generate_with_mocked_llm(self, scout_report_dict, mock_llm_response):
        """Full pipeline with mocked Anthropic client."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        mock_content = Mock()
        mock_content.text = mock_llm_response

        mock_usage = Mock()
        mock_usage.input_tokens = 3000
        mock_usage.output_tokens = 1500
        mock_usage.cache_read_input_tokens = 0
        mock_usage.cache_creation_input_tokens = 800

        mock_message = Mock()
        mock_message.content = [mock_content]
        mock_message.usage = mock_usage

        with patch.dict("sys.modules", {"anthropic": Mock()}):
            import sys

            mock_anthropic_module = sys.modules["anthropic"]
            mock_anthropic_class = Mock()
            mock_client_instance = Mock()
            mock_client_instance.messages.create.return_value = mock_message
            mock_anthropic_class.return_value = mock_client_instance
            mock_anthropic_module.Anthropic = mock_anthropic_class

            gen = AntiStratGenerator(api_key="test-key")
            report = gen.generate(scout_report_dict)

        # Report fully populated
        assert report.team_name == "TestTeam"
        assert len(report.player_threats) == 3
        assert len(report.map_counters) == 1
        assert len(report.economy_exploits) == 1
        assert report.t_side_game_plan != ""
        assert report.ct_side_game_plan != ""
        assert report.generation_error == ""
        assert report.model_used == "claude-sonnet-4-5-20250929"

        # Verify LLM was called with DEEP tier
        call_args = mock_client_instance.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-sonnet-4-5-20250929"

    def test_generate_handles_llm_error(self, scout_report_dict):
        """LLM failure produces report with error but still has data threats."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        with patch.dict("sys.modules", {"anthropic": Mock()}):
            import sys

            mock_anthropic_module = sys.modules["anthropic"]
            mock_anthropic_class = Mock()
            mock_client_instance = Mock()
            mock_client_instance.messages.create.side_effect = Exception("API Error")
            mock_anthropic_class.return_value = mock_client_instance
            mock_anthropic_module.Anthropic = mock_anthropic_class

            gen = AntiStratGenerator(api_key="test-key")
            report = gen.generate(scout_report_dict)

        # Data threats still present
        assert len(report.player_threats) == 3
        # LLM plans empty but error recorded
        assert "API Error" in report.generation_error
        assert report.t_side_game_plan == ""


# =============================================================================
# Test: Role Classification Edge Cases
# =============================================================================


class TestRoleClassification:
    """Test _classify_role with edge cases."""

    def test_lurker_detection(self):
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        player = {
            "play_style": "passive",
            "awp_usage_rate": 3.0,
            "entry_attempt_rate": 5.0,
            "entry_success_rate": 30.0,
            "avg_kills_per_round": 0.65,
        }
        assert gen._classify_role(player) == "lurker"

    def test_star_fragger_detection(self):
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        player = {
            "play_style": "mixed",
            "awp_usage_rate": 5.0,
            "entry_attempt_rate": 10.0,
            "entry_success_rate": 50.0,
            "avg_kills_per_round": 0.90,
        }
        assert gen._classify_role(player) == "star"

    def test_all_zeros_defaults_to_support(self):
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        player = {
            "play_style": "mixed",
            "awp_usage_rate": 0,
            "entry_attempt_rate": 0,
            "entry_success_rate": 0,
            "avg_kills_per_round": 0,
        }
        assert gen._classify_role(player) == "support"


# =============================================================================
# Test: Threat Level Assessment Edge Cases
# =============================================================================


class TestThreatAssessment:
    """Test _assess_threat_level with edge cases."""

    def test_borderline_medium(self):
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        player = {"kd_ratio": 1.0, "avg_kills_per_round": 0.65, "entry_success_rate": 45.0}
        assert gen._assess_threat_level(player) == "medium"

    def test_very_high_entry_success(self):
        """High entry success alone can make high threat."""
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        player = {"kd_ratio": 1.1, "avg_kills_per_round": 0.75, "entry_success_rate": 60.0}
        assert gen._assess_threat_level(player) == "high"

    def test_very_low_stats(self):
        from opensight.ai.antistrat_report import AntiStratGenerator

        gen = AntiStratGenerator(api_key=None)
        player = {"kd_ratio": 0.5, "avg_kills_per_round": 0.3, "entry_success_rate": 20.0}
        assert gen._assess_threat_level(player) == "low"


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_antistrat_generator_singleton(self):
        from opensight.ai.antistrat_report import get_antistrat_generator

        gen1 = get_antistrat_generator()
        gen2 = get_antistrat_generator()
        assert gen1 is gen2

    def test_generate_antistrat_report_convenience(self, scout_report_dict):
        """The convenience function works without API key (data-only)."""
        from opensight.ai import antistrat_report

        # Reset singleton to ensure clean state
        antistrat_report._generator_instance = None

        report = antistrat_report.generate_antistrat_report(scout_report_dict)
        assert report.team_name == "TestTeam"
        assert len(report.player_threats) == 3

        # Clean up singleton
        antistrat_report._generator_instance = None


# =============================================================================
# Test: to_dict Round-Trip
# =============================================================================


class TestToDict:
    """Test that to_dict() produces valid, complete output."""

    def test_to_dict_all_fields_present(self, scout_report_dict, mock_llm_response):
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )
        gen._populate_report_from_llm(report, mock_llm_response)

        d = report.to_dict()

        # Top-level keys
        assert "team_name" in d
        assert "maps_analyzed" in d
        assert "demos_analyzed" in d
        assert "confidence_level" in d
        assert "player_threats" in d
        assert "map_counters" in d
        assert "economy_exploits" in d
        assert "game_plans" in d
        assert "veto_recommendation" in d
        assert "model_used" in d

        # Game plans sub-keys
        gp = d["game_plans"]
        assert "t_side" in gp
        assert "ct_side" in gp
        assert "pistol_round" in gp
        assert "anti_eco" in gp

    def test_to_dict_json_serializable(self, scout_report_dict, mock_llm_response):
        """to_dict() output must be JSON-serializable (no dataclass objects)."""
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )
        gen._populate_report_from_llm(report, mock_llm_response)

        # This must not raise
        json_str = json.dumps(report.to_dict())
        assert len(json_str) > 100

    def test_player_threat_stats_in_dict(self, scout_report_dict):
        from opensight.ai.antistrat_report import AntiStratGenerator, AntiStratReport

        gen = AntiStratGenerator(api_key=None)
        threats = gen._extract_player_threats(scout_report_dict)
        report = AntiStratReport(
            team_name="TestTeam",
            maps_analyzed=["de_mirage"],
            demos_analyzed=3,
            confidence_level="medium",
            player_threats=threats,
        )

        d = report.to_dict()
        star_dict = next(pt for pt in d["player_threats"] if pt["name"] == "StarPlayer")
        stats = star_dict["stats"]

        assert stats["avg_kills_per_round"] == 0.92
        # entry_success_rate stored as 0-1, displayed as percentage
        assert stats["entry_success_rate"] == 58.0
        assert stats["play_style"] == "aggressive"


# =============================================================================
# Test: API Endpoints
# =============================================================================


class TestAntiStratEndpoints:
    """Test the anti-strat API endpoints via FastAPI test client."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from opensight.api import app

        return TestClient(app)

    def test_generate_no_session(self, client):
        """POST /api/antistrat/{id}/generate with invalid session → 404."""
        resp = client.post(
            "/api/antistrat/nonexistent/generate",
            json={"opponent_steamids": [123], "team_name": "Test"},
        )
        assert resp.status_code == 404

    def test_get_report_no_session(self, client):
        """GET /api/antistrat/{id}/report with no generated report → 404."""
        resp = client.get("/api/antistrat/nonexistent/report")
        assert resp.status_code == 404

    def test_generate_no_opponents(self, client):
        """POST with empty opponent list → 400."""
        # Create a scouting session first
        resp = client.post("/api/scouting/session")
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        resp = client.post(
            f"/api/antistrat/{session_id}/generate",
            json={"opponent_steamids": [], "team_name": "Test"},
        )
        assert resp.status_code == 400

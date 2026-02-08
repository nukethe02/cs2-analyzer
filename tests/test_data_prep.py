"""
Tests for the AI data preprocessing pipeline.

Tests that preprocess_match() correctly extracts and computes derived
metrics from orchestrator output, and that to_llm_prompt() generates
properly structured output for LLM consumption.
"""

import pytest

from opensight.ai.data_prep import (
    MatchSummary,
    PlayerSummary,
    RoundSummary,
    preprocess_match,
    to_llm_prompt,
)

# =============================================================================
# Fixtures — realistic orchestrator output snippets
# =============================================================================


@pytest.fixture
def minimal_result():
    """Minimal valid orchestrator result with 2 players and 3 rounds."""
    return {
        "demo_info": {
            "map": "de_mirage",
            "rounds": 3,
            "score_ct": 2,
            "score_t": 1,
            "team1_name": "Alphas",
            "team2_name": "Bravos",
        },
        "players": {
            "111": {
                "name": "PlayerA",
                "team": "CT",
                "stats": {
                    "kills": 10,
                    "deaths": 5,
                    "assists": 3,
                    "adr": 85.0,
                    "headshot_pct": 50.0,
                },
                "rating": {
                    "hltv_rating": 1.25,
                    "kast_percentage": 75.0,
                },
                "advanced": {
                    "ttd_median_ms": 250,
                    "cp_median_error_deg": 8.0,
                },
                "utility": {
                    "flashbangs_thrown": 3,
                    "smokes_thrown": 2,
                    "he_thrown": 1,
                    "molotovs_thrown": 1,
                    "flash_assists": 2,
                    "he_damage": 40,
                    "molotov_damage": 30,
                },
                "entry": {
                    "entry_attempts": 6,
                    "entry_kills": 4,
                    "entry_deaths": 2,
                },
                "trades": {
                    "trade_kill_success": 3,
                    "untraded_deaths": 1,
                    "traded_death_success": 2,
                },
                "clutches": {
                    "total_situations": 2,
                    "clutch_wins": 1,
                },
                "weapon_kills": {
                    "ak47": 6,
                    "awp": 0,
                    "usp_silencer": 2,
                },
            },
            "222": {
                "name": "PlayerB",
                "team": "T",
                "stats": {
                    "kills": 4,
                    "deaths": 8,
                    "assists": 1,
                    "adr": 52.0,
                    "headshot_pct": 30.0,
                },
                "rating": {
                    "hltv_rating": 0.72,
                    "kast_percentage": 58.0,
                },
                "advanced": {},
                "utility": {
                    "flashbangs_thrown": 2,
                    "smokes_thrown": 1,
                    "he_thrown": 0,
                    "molotovs_thrown": 0,
                    "flash_assists": 1,
                    "he_damage": 0,
                    "molotov_damage": 0,
                },
                "entry": {
                    "entry_attempts": 3,
                    "entry_kills": 1,
                    "entry_deaths": 2,
                },
                "trades": {
                    "trade_kill_success": 0,
                    "untraded_deaths": 4,
                    "traded_death_success": 3,
                },
                "clutches": {
                    "total_situations": 1,
                    "clutch_wins": 0,
                },
                "weapon_kills": {
                    "ak47": 3,
                    "awp": 1,
                },
            },
        },
        "round_timeline": [
            {
                "round_num": 1,
                "winner": "CT",
                "win_reason": "Elimination",
                "round_type": "pistol",
                "kills": [
                    {
                        "killer": "PlayerA",
                        "victim": "PlayerB",
                        "killer_team": "CT",
                        "victim_team": "T",
                        "weapon": "usp_silencer",
                        "tick": 1000,
                        "was_dry_peek": False,
                    },
                ],
                "utility": [],
                "blinds": [],
                "events": [],
                "clutches": [],
                "economy": {
                    "ct": {"buy_type": "pistol", "equipment": 800},
                    "t": {"buy_type": "pistol", "equipment": 800},
                },
            },
            {
                "round_num": 2,
                "winner": "CT",
                "win_reason": "Elimination",
                "round_type": "full_buy",
                "kills": [
                    {
                        "killer": "PlayerA",
                        "victim": "PlayerB",
                        "killer_team": "CT",
                        "victim_team": "T",
                        "weapon": "ak47",
                        "tick": 2000,
                        "was_dry_peek": True,
                    },
                ],
                "utility": [
                    {
                        "tick": 1900,
                        "type": "flashbang",
                        "player": "PlayerA",
                        "player_team": "CT",
                    },
                ],
                "blinds": [],
                "events": [],
                "clutches": [],
                "economy": {
                    "ct": {"buy_type": "full_buy", "equipment": 5400},
                    "t": {"buy_type": "eco", "equipment": 1200},
                },
            },
            {
                "round_num": 3,
                "winner": "T",
                "win_reason": "Bomb Exploded",
                "round_type": "full_buy",
                "kills": [
                    {
                        "killer": "PlayerB",
                        "victim": "PlayerA",
                        "killer_team": "T",
                        "victim_team": "CT",
                        "weapon": "ak47",
                        "tick": 3000,
                        "was_dry_peek": True,
                    },
                ],
                "utility": [],
                "blinds": [],
                "events": [
                    {
                        "type": "bomb_plant",
                        "player": "PlayerB",
                        "site": "A",
                        "tick": 2800,
                    },
                ],
                "clutches": [
                    {
                        "player": "PlayerB",
                        "scenario": "1v1",
                        "won": True,
                        "kills_in_clutch": 1,
                    },
                ],
                "economy": {
                    "ct": {"buy_type": "full_buy", "equipment": 5200},
                    "t": {"buy_type": "full_buy", "equipment": 4800},
                },
            },
        ],
    }


@pytest.fixture
def empty_result():
    """Completely empty orchestrator result."""
    return {}


# =============================================================================
# preprocess_match() tests
# =============================================================================


class TestPreprocessMatch:
    def test_match_metadata(self, minimal_result):
        summary = preprocess_match(minimal_result)
        assert summary.map_name == "de_mirage"
        assert summary.final_score == "2-1"
        assert summary.total_rounds == 3
        assert summary.winner == "CT"
        assert summary.overtime is False
        assert summary.team_a_name == "Alphas"
        assert summary.team_b_name == "Bravos"

    def test_player_count(self, minimal_result):
        summary = preprocess_match(minimal_result)
        total_players = len(summary.team_a_players) + len(summary.team_b_players)
        assert total_players == 2

    def test_player_core_stats(self, minimal_result):
        summary = preprocess_match(minimal_result)
        # Find PlayerA
        player_a = None
        for p in summary.team_a_players + summary.team_b_players:
            if p.name == "PlayerA":
                player_a = p
                break
        assert player_a is not None
        assert player_a.kills == 10
        assert player_a.deaths == 5
        assert player_a.assists == 3
        assert player_a.adr == 85.0
        assert player_a.rating == 1.25
        assert player_a.kast_pct == 75.0

    def test_opening_duel_rate(self, minimal_result):
        summary = preprocess_match(minimal_result)
        player_a = next(
            p for p in summary.team_a_players + summary.team_b_players if p.name == "PlayerA"
        )
        assert player_a.opening_duel_attempts == 6
        assert player_a.opening_duel_wins == 4
        assert abs(player_a.opening_duel_win_rate - 66.67) < 1.0

    def test_clutch_stats(self, minimal_result):
        summary = preprocess_match(minimal_result)
        player_a = next(
            p for p in summary.team_a_players + summary.team_b_players if p.name == "PlayerA"
        )
        assert player_a.clutch_attempts == 2
        assert player_a.clutch_wins == 1
        assert player_a.clutch_win_rate == 50.0

    def test_trade_stats(self, minimal_result):
        summary = preprocess_match(minimal_result)
        player_a = next(
            p for p in summary.team_a_players + summary.team_b_players if p.name == "PlayerA"
        )
        assert player_a.trades_given == 3
        assert player_a.trades_missed == 1
        # trade_rate = traded_death_success / deaths * 100 = 2/5 * 100 = 40%
        assert player_a.trade_rate == 40.0

    def test_utility_stats(self, minimal_result):
        summary = preprocess_match(minimal_result)
        player_a = next(
            p for p in summary.team_a_players + summary.team_b_players if p.name == "PlayerA"
        )
        assert player_a.utility_thrown == 7  # 3+2+1+1
        assert player_a.flash_assists == 2
        assert player_a.utility_damage == 70.0  # 40+30

    def test_dry_peek_rate(self, minimal_result):
        """PlayerA has 1 death with was_dry_peek=True out of 1 total."""
        summary = preprocess_match(minimal_result)
        player_a = next(
            p for p in summary.team_a_players + summary.team_b_players if p.name == "PlayerA"
        )
        # In round 3, PlayerA dies with was_dry_peek=True
        assert player_a.dry_peek_deaths == 1
        assert player_a.total_deaths_with_peek_data == 1
        assert player_a.dry_peek_rate == 100.0

    def test_side_kills(self, minimal_result):
        summary = preprocess_match(minimal_result)
        player_a = next(
            p for p in summary.team_a_players + summary.team_b_players if p.name == "PlayerA"
        )
        # PlayerA has 2 kills in timeline, both as CT
        assert player_a.ct_kills == 2
        assert player_a.t_kills == 0

    def test_role_guess_entry(self, minimal_result):
        """PlayerA has 4/6 opening duels (67%) → should be 'entry'."""
        summary = preprocess_match(minimal_result)
        player_a = next(
            p for p in summary.team_a_players + summary.team_b_players if p.name == "PlayerA"
        )
        assert player_a.role_guess == "entry"

    def test_round_summaries_count(self, minimal_result):
        summary = preprocess_match(minimal_result)
        assert len(summary.rounds) == 3

    def test_round_summary_fields(self, minimal_result):
        summary = preprocess_match(minimal_result)
        r1 = summary.rounds[0]
        assert r1.round_number == 1
        assert r1.winner == "CT"
        assert r1.win_method == "Elimination"
        assert r1.first_kill_player == "PlayerA"
        assert r1.first_kill_victim == "PlayerB"
        assert r1.first_kill_side == "CT"

    def test_round_with_clutch(self, minimal_result):
        summary = preprocess_match(minimal_result)
        r3 = summary.rounds[2]
        assert r3.clutch_player == "PlayerB"
        assert r3.clutch_situation == "1v1"
        assert r3.clutch_won is True

    def test_round_bomb_site(self, minimal_result):
        summary = preprocess_match(minimal_result)
        r3 = summary.rounds[2]
        assert r3.bomb_site == "A"

    def test_round_utility_before_entry(self, minimal_result):
        summary = preprocess_match(minimal_result)
        r2 = summary.rounds[1]
        # Round 2 has a flash at tick 1900 before kill at tick 2000
        assert r2.utility_used_before_entry == 1

    def test_pistol_round_tracking(self, minimal_result):
        summary = preprocess_match(minimal_result)
        # Round 1: CT wins, team_a is CT in first half → team_a wins pistol
        assert summary.team_a_pistol_wins == 1

    def test_running_score(self, minimal_result):
        summary = preprocess_match(minimal_result)
        assert summary.rounds[0].score_after_ct == 1
        assert summary.rounds[0].score_after_t == 0
        assert summary.rounds[1].score_after_ct == 2
        assert summary.rounds[1].score_after_t == 0
        assert summary.rounds[2].score_after_ct == 2
        assert summary.rounds[2].score_after_t == 1

    def test_empty_result(self, empty_result):
        """Preprocessing an empty dict should not crash."""
        summary = preprocess_match(empty_result)
        assert summary.map_name == "unknown"
        assert summary.total_rounds == 0
        assert len(summary.team_a_players) == 0
        assert len(summary.rounds) == 0

    def test_overtime_detection(self):
        result = {
            "demo_info": {"map": "de_dust2", "rounds": 32, "score_ct": 17, "score_t": 15},
            "players": {},
            "round_timeline": [],
        }
        summary = preprocess_match(result)
        assert summary.overtime is True

    def test_players_sorted_by_rating(self, minimal_result):
        summary = preprocess_match(minimal_result)
        # Within each team, players should be sorted by rating descending
        for team_players in [summary.team_a_players, summary.team_b_players]:
            for i in range(len(team_players) - 1):
                assert team_players[i].rating >= team_players[i + 1].rating


# =============================================================================
# to_llm_prompt() tests
# =============================================================================


class TestToLLMPrompt:
    def test_coaching_prompt_has_all_sections(self, minimal_result):
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="coaching")
        assert "<match_context>" in prompt
        assert "</match_context>" in prompt
        assert "<team " in prompt
        assert "</team>" in prompt
        assert "<round_flow>" in prompt
        assert "</round_flow>" in prompt
        assert "<economy_patterns>" in prompt
        assert "<analysis_instructions>" in prompt

    def test_scouting_prompt_has_instructions(self, minimal_result):
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="scouting")
        assert "exploitable patterns" in prompt

    def test_economy_focus_has_economy_section(self, minimal_result):
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="economy")
        assert "<economy_patterns>" in prompt
        assert "economy decision-making" in prompt.lower()

    def test_antistrat_focus(self, minimal_result):
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="antistrat")
        assert "counter-strategies" in prompt

    def test_prompt_contains_map_name(self, minimal_result):
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="coaching")
        assert "de_mirage" in prompt

    def test_prompt_contains_player_names(self, minimal_result):
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="coaching")
        assert "PlayerA" in prompt
        assert "PlayerB" in prompt

    def test_prompt_contains_scores(self, minimal_result):
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="coaching")
        assert "2-1" in prompt

    def test_prompt_contains_round_flow(self, minimal_result):
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="coaching")
        assert "R1:" in prompt
        assert "R2:" in prompt
        assert "R3:" in prompt

    def test_prompt_token_count_reasonable(self, minimal_result):
        """LLM prompt for a 3-round match should be well under 2000 tokens."""
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="coaching")
        est_tokens = len(prompt) / 4
        assert est_tokens < 2000, f"Prompt too long: ~{est_tokens:.0f} tokens"

    def test_economy_focus_skips_round_flow(self, minimal_result):
        """Economy focus should NOT include round_flow to save tokens."""
        summary = preprocess_match(minimal_result)
        prompt = to_llm_prompt(summary, focus="economy")
        assert "<round_flow>" not in prompt

    def test_empty_summary_produces_valid_prompt(self):
        summary = MatchSummary()
        prompt = to_llm_prompt(summary, focus="coaching")
        assert "<match_context>" in prompt
        assert "</analysis_instructions>" in prompt


# =============================================================================
# Dataclass tests
# =============================================================================


class TestDataclasses:
    def test_player_summary_defaults(self):
        ps = PlayerSummary()
        assert ps.name == ""
        assert ps.kills == 0
        assert ps.rating == 0.0
        assert ps.role_guess == "unknown"

    def test_round_summary_defaults(self):
        rs = RoundSummary()
        assert rs.round_number == 0
        assert rs.winner == ""
        assert rs.clutch_player is None

    def test_match_summary_defaults(self):
        ms = MatchSummary()
        assert ms.map_name == ""
        assert ms.total_rounds == 0
        assert ms.team_a_players == []
        assert ms.rounds == []

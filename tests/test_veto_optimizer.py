"""
Tests for Map Veto Optimizer (Prompt 7).

Covers:
  - MapStrength, VetoRecommendation, VetoAnalysis dataclass construction & to_dict
  - Map strength computation from orchestrator results
  - Recency weighting with exponential decay
  - Map strengths from scouting data
  - Pool filling for missing maps
  - BO1 veto sequence generation
  - BO3 veto sequence generation
  - Decider prediction
  - Confidence levels
  - Edge cases (empty data, single demo, all same map)
  - API endpoint integration
"""

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Fixtures — realistic orchestrator results for multiple maps
# =============================================================================


def _make_demo(
    map_name: str,
    score_ct: int = 13,
    score_t: int = 8,
    timeline: list | None = None,
) -> dict:
    """Build a minimal orchestrator result for veto testing."""
    return {
        "demo_info": {
            "map": map_name,
            "rounds": score_ct + score_t,
            "score_ct": score_ct,
            "score_t": score_t,
            "team1_name": "TeamAlpha",
            "team2_name": "TeamBravo",
        },
        "players": {},
        "round_timeline": timeline or [],
    }


def _make_round(round_num: int, winner: str) -> dict:
    """Build a minimal round_timeline entry."""
    return {"round_num": round_num, "winner": winner, "kills": [], "economy": {}}


def _demo_with_timeline(map_name: str, score_ct: int, score_t: int) -> dict:
    """Build a demo with a simple round timeline for CT/T split testing."""
    timeline = []
    ct_wins = 0
    t_wins = 0

    # First half: rounds 1-12, team1 is CT
    for r in range(1, 13):
        if ct_wins < min(score_ct, 12) and (r <= score_ct or t_wins >= 12):
            timeline.append(_make_round(r, "CT"))
            ct_wins += 1
        else:
            timeline.append(_make_round(r, "T"))
            t_wins += 1

    # Second half: rounds 13+, team1 is T
    remaining_ct = score_ct - ct_wins
    remaining_t = score_t - t_wins
    for r in range(13, 13 + remaining_ct + remaining_t):
        if remaining_t > 0:
            timeline.append(_make_round(r, "T"))
            remaining_t -= 1
        else:
            timeline.append(_make_round(r, "CT"))
            remaining_ct -= 1

    return _make_demo(map_name, score_ct, score_t, timeline)


@pytest.fixture
def diverse_demos():
    """Your team's demos across multiple maps with varying results."""
    return [
        # Mirage: 2 wins, 1 loss
        _make_demo("de_mirage", 13, 8),
        _make_demo("de_mirage", 13, 11),
        _make_demo("de_mirage", 10, 13),
        # Inferno: 3 wins (strong map)
        _make_demo("de_inferno", 13, 5),
        _make_demo("de_inferno", 13, 9),
        _make_demo("de_inferno", 16, 14),
        # Nuke: 0 wins, 2 losses (weak map)
        _make_demo("de_nuke", 8, 13),
        _make_demo("de_nuke", 5, 13),
        # Ancient: 1 win, 1 loss
        _make_demo("de_ancient", 13, 10),
        _make_demo("de_ancient", 9, 13),
    ]


@pytest.fixture
def opponent_demos():
    """Opponent's demos."""
    return [
        # Nuke: 3 wins (strong map)
        _make_demo("de_nuke", 13, 4),
        _make_demo("de_nuke", 13, 7),
        _make_demo("de_nuke", 13, 10),
        # Mirage: 1 win, 2 losses
        _make_demo("de_mirage", 13, 11),
        _make_demo("de_mirage", 9, 13),
        _make_demo("de_mirage", 7, 13),
        # Dust2: 2 wins
        _make_demo("de_dust2", 13, 8),
        _make_demo("de_dust2", 13, 10),
        # Vertigo: 1 loss
        _make_demo("de_vertigo", 6, 13),
    ]


@pytest.fixture
def scouting_data():
    """Opponent scouting data in TeamScoutReport.to_dict() format."""
    return {
        "team_name": "OpponentTeam",
        "demos_analyzed": 4,
        "total_rounds": 100,
        "confidence_level": "medium",
        "maps_analyzed": ["de_nuke", "de_inferno"],
        "players": [],
        "map_tendencies": [
            {
                "map_name": "de_nuke",
                "t_side": {"aggression": 70, "default_setup": "outside control"},
                "ct_side": {"aggression": 40, "default_setup": "2-1-2"},
                "timing": {"avg_first_contact": 20.0, "avg_execute_time": 30.0},
            },
            {
                "map_name": "de_inferno",
                "t_side": {"aggression": 55, "default_setup": "banana control"},
                "ct_side": {"aggression": 35, "default_setup": "2-1-2"},
                "timing": {"avg_first_contact": 25.0, "avg_execute_time": 35.0},
            },
        ],
        "economy": {"tendency": "balanced", "force_buy_rate": 22.0, "eco_round_rate": 18.0},
    }


# =============================================================================
# TestMapStrength
# =============================================================================


class TestMapStrength:
    def test_construction(self):
        from opensight.ai.veto_optimizer import MapStrength

        ms = MapStrength(
            map_name="de_mirage",
            matches_played=5,
            win_rate=0.6,
            ct_win_rate=0.55,
            t_win_rate=0.65,
            avg_rounds_won=12.4,
            recency_weighted_win_rate=0.65,
            confidence=0.81,
        )
        assert ms.map_name == "de_mirage"
        assert ms.win_rate == 0.6

    def test_defaults(self):
        from opensight.ai.veto_optimizer import MapStrength

        ms = MapStrength(map_name="de_dust2")
        assert ms.matches_played == 0
        assert ms.win_rate == 0.5
        assert ms.confidence == 0.0

    def test_to_dict(self):
        from opensight.ai.veto_optimizer import MapStrength

        ms = MapStrength(map_name="de_nuke", matches_played=3, win_rate=0.667)
        d = ms.to_dict()
        assert d["map_name"] == "de_nuke"
        assert d["matches_played"] == 3
        assert d["win_rate"] == 0.667


# =============================================================================
# TestVetoRecommendation
# =============================================================================


class TestVetoRecommendation:
    def test_construction(self):
        from opensight.ai.veto_optimizer import VetoRecommendation

        vr = VetoRecommendation(
            action="ban",
            map_name="de_nuke",
            reason="Opponent 100% WR",
            your_win_rate=0.3,
            opponent_win_rate=1.0,
            net_advantage=-0.7,
        )
        assert vr.action == "ban"
        assert vr.net_advantage == -0.7

    def test_to_dict(self):
        from opensight.ai.veto_optimizer import VetoRecommendation

        vr = VetoRecommendation(action="pick", map_name="de_inferno", reason="Strong map")
        d = vr.to_dict()
        assert d["action"] == "pick"
        assert d["map_name"] == "de_inferno"


# =============================================================================
# TestVetoAnalysis
# =============================================================================


class TestVetoAnalysis:
    def test_construction(self):
        from opensight.ai.veto_optimizer import VetoAnalysis

        va = VetoAnalysis(
            best_map_for_you="de_inferno",
            worst_map_for_you="de_nuke",
            predicted_decider_map="de_ancient",
            confidence="medium",
        )
        assert va.best_map_for_you == "de_inferno"
        assert va.confidence == "medium"

    def test_to_dict(self):
        from opensight.ai.veto_optimizer import MapStrength, VetoAnalysis, VetoRecommendation

        va = VetoAnalysis(
            your_map_pool=[MapStrength(map_name="de_dust2")],
            opponent_map_pool=[MapStrength(map_name="de_dust2")],
            recommended_veto_sequence=[
                VetoRecommendation(action="ban", map_name="de_nuke", reason="test")
            ],
            best_map_for_you="de_inferno",
            worst_map_for_you="de_nuke",
        )
        d = va.to_dict()
        assert len(d["your_map_pool"]) == 1
        assert len(d["recommended_veto_sequence"]) == 1
        assert d["best_map_for_you"] == "de_inferno"


# =============================================================================
# TestComputeMapStrengths
# =============================================================================


class TestComputeMapStrengths:
    def test_basic(self, diverse_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        strengths = opt._compute_map_strengths(diverse_demos)

        map_dict = {ms.map_name: ms for ms in strengths}

        # Inferno: 3 wins out of 3
        assert map_dict["de_inferno"].win_rate == 1.0
        assert map_dict["de_inferno"].matches_played == 3

        # Nuke: 0 wins out of 2
        assert map_dict["de_nuke"].win_rate == 0.0
        assert map_dict["de_nuke"].matches_played == 2

        # Mirage: 2 wins out of 3
        assert abs(map_dict["de_mirage"].win_rate - 2 / 3) < 0.01

    def test_empty_demos(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        strengths = opt._compute_map_strengths([])
        assert strengths == []

    def test_single_demo(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        strengths = opt._compute_map_strengths([_make_demo("de_dust2", 13, 8)])
        assert len(strengths) == 1
        assert strengths[0].map_name == "de_dust2"
        assert strengths[0].win_rate == 1.0
        assert strengths[0].matches_played == 1

    def test_all_losses(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        demos = [
            _make_demo("de_vertigo", 5, 13),
            _make_demo("de_vertigo", 8, 13),
        ]
        strengths = opt._compute_map_strengths(demos)
        assert strengths[0].win_rate == 0.0

    def test_confidence_increases_with_matches(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        one = opt._compute_map_strengths([_make_demo("de_dust2", 13, 8)])
        five = opt._compute_map_strengths([_make_demo("de_dust2", 13, 8) for _ in range(5)])
        assert five[0].confidence > one[0].confidence

    def test_avg_rounds_won(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        demos = [
            _make_demo("de_mirage", 13, 8),  # 13 rounds won
            _make_demo("de_mirage", 10, 13),  # 10 rounds won
        ]
        strengths = opt._compute_map_strengths(demos)
        assert strengths[0].avg_rounds_won == 11.5  # (13+10)/2

    def test_unknown_map_excluded(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        # A map not in active pool should still be computed
        demos = [_make_demo("de_train", 13, 8)]
        strengths = opt._compute_map_strengths(demos)
        assert len(strengths) == 1
        assert strengths[0].map_name == "de_train"

    def test_ct_t_split_from_timeline(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        # Build a demo where CT wins 10 of 12 first-half rounds, T wins 3 of 9 second-half
        timeline = []
        for r in range(1, 13):
            timeline.append(_make_round(r, "CT" if r <= 10 else "T"))
        for r in range(13, 22):
            timeline.append(_make_round(r, "T" if r <= 15 else "CT"))

        demo = _make_demo("de_mirage", 13, 8, timeline)
        strengths = opt._compute_map_strengths([demo])
        ms = strengths[0]

        # CT first half: 10/12 rounds won
        assert abs(ms.ct_win_rate - 10 / 12) < 0.01
        # T second half: 3/9 rounds won (rounds 13-15 are T wins = our team wins)
        assert abs(ms.t_win_rate - 3 / 9) < 0.01


# =============================================================================
# TestRecencyWeight
# =============================================================================


class TestRecencyWeight:
    def test_all_wins(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        demos = [_make_demo("de_mirage", 13, 8) for _ in range(5)]
        assert opt._recency_weight(demos) == 1.0

    def test_all_losses(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        demos = [_make_demo("de_mirage", 5, 13) for _ in range(5)]
        assert opt._recency_weight(demos) == 0.0

    def test_empty(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        assert opt._recency_weight([]) == 0.5

    def test_recent_wins_weighted_higher(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()

        # Old losses, recent wins → should be > 0.5
        recent_good = [
            _make_demo("de_mirage", 5, 13),  # oldest: loss
            _make_demo("de_mirage", 5, 13),  # old: loss
            _make_demo("de_mirage", 13, 8),  # recent: win
            _make_demo("de_mirage", 13, 8),  # most recent: win
        ]

        # Old wins, recent losses → should be < 0.5
        recent_bad = [
            _make_demo("de_mirage", 13, 8),  # oldest: win
            _make_demo("de_mirage", 13, 8),  # old: win
            _make_demo("de_mirage", 5, 13),  # recent: loss
            _make_demo("de_mirage", 5, 13),  # most recent: loss
        ]

        good_wr = opt._recency_weight(recent_good)
        bad_wr = opt._recency_weight(recent_bad)

        assert good_wr > 0.5
        assert bad_wr < 0.5
        assert good_wr > bad_wr

    def test_single_demo(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        assert opt._recency_weight([_make_demo("de_dust2", 13, 8)]) == 1.0
        assert opt._recency_weight([_make_demo("de_dust2", 5, 13)]) == 0.0


# =============================================================================
# TestMapStrengthsFromScouting
# =============================================================================


class TestMapStrengthsFromScouting:
    def test_basic(self, scouting_data):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        strengths = opt._map_strengths_from_scouting(scouting_data)

        assert len(strengths) == 2
        map_names = {ms.map_name for ms in strengths}
        assert "de_nuke" in map_names
        assert "de_inferno" in map_names

    def test_estimated_win_rate(self, scouting_data):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        strengths = opt._map_strengths_from_scouting(scouting_data)
        # Maps they play on should be estimated at ~55%
        for ms in strengths:
            assert ms.win_rate >= 0.5

    def test_lower_confidence(self, scouting_data):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        strengths = opt._map_strengths_from_scouting(scouting_data)
        # Scouting confidence should be lower than direct computation
        for ms in strengths:
            assert ms.confidence < 1.0

    def test_empty_scouting(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        strengths = opt._map_strengths_from_scouting({})
        assert strengths == []

    def test_no_map_tendencies(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        strengths = opt._map_strengths_from_scouting({"map_tendencies": []})
        assert strengths == []


# =============================================================================
# TestFillPool
# =============================================================================


class TestFillPool:
    def test_fills_missing_maps(self):
        from opensight.ai.veto_optimizer import ACTIVE_MAP_POOL, MapStrength, VetoOptimizer

        opt = VetoOptimizer()
        existing = [MapStrength(map_name="de_mirage", win_rate=0.7)]
        filled = opt._fill_pool(existing)

        map_names = {ms.map_name for ms in filled}
        for pool_map in ACTIVE_MAP_POOL:
            assert pool_map in map_names

    def test_doesnt_duplicate(self):
        from opensight.ai.veto_optimizer import ACTIVE_MAP_POOL, MapStrength, VetoOptimizer

        opt = VetoOptimizer()
        existing = [MapStrength(map_name=m) for m in ACTIVE_MAP_POOL]
        filled = opt._fill_pool(existing)
        assert len(filled) == len(ACTIVE_MAP_POOL)

    def test_default_values(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        filled = opt._fill_pool([])
        for ms in filled:
            assert ms.win_rate == 0.5
            assert ms.confidence == 0.0
            assert ms.matches_played == 0


# =============================================================================
# TestBO1Veto
# =============================================================================


class TestBO1Veto:
    def test_sequence_length(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        # BO1: 6 bans + 1 remaining = 7 steps
        assert len(analysis.recommended_veto_sequence) == 7

    def test_all_bans_plus_one_pick(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        actions = [v.action for v in analysis.recommended_veto_sequence]
        assert actions.count("ban") == 6
        assert actions.count("pick") == 1
        assert actions[-1] == "pick"  # last step is the remaining map

    def test_no_duplicate_maps(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        maps = [v.map_name for v in analysis.recommended_veto_sequence]
        assert len(maps) == len(set(maps))

    def test_all_maps_from_pool(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import ACTIVE_MAP_POOL, VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        maps = {v.map_name for v in analysis.recommended_veto_sequence}
        assert maps == set(ACTIVE_MAP_POOL)

    def test_opponent_best_map_banned(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        # Opponent is strong on de_nuke (3-0) — we should ban it
        our_bans = [
            v.map_name
            for i, v in enumerate(analysis.recommended_veto_sequence)
            if i % 2 == 0 and v.action == "ban"
        ]
        assert "de_nuke" in our_bans

    def test_our_best_map_banned_by_opponent(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        # We're strong on de_inferno (3-0) — opponent should ban it
        opp_bans = [
            v.map_name
            for i, v in enumerate(analysis.recommended_veto_sequence)
            if i % 2 == 1 and v.action == "ban"
        ]
        assert "de_inferno" in opp_bans


# =============================================================================
# TestBO3Veto
# =============================================================================


class TestBO3Veto:
    def test_sequence_length(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo3")

        # BO3: ban-ban-pick-pick-ban-ban-decider = 7 steps
        assert len(analysis.recommended_veto_sequence) == 7

    def test_correct_action_sequence(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo3")

        actions = [v.action for v in analysis.recommended_veto_sequence]
        assert actions == ["ban", "ban", "pick", "pick", "ban", "ban", "pick"]

    def test_no_duplicate_maps(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo3")

        maps = [v.map_name for v in analysis.recommended_veto_sequence]
        assert len(maps) == len(set(maps))

    def test_our_pick_is_strong_map(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo3")

        # Our pick (step 3) should be a strong map for us
        our_pick = analysis.recommended_veto_sequence[2]
        assert our_pick.action == "pick"
        assert our_pick.net_advantage >= 0  # positive advantage


# =============================================================================
# TestFullAnalysis
# =============================================================================


class TestFullAnalysis:
    def test_best_worst_map(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        assert analysis.best_map_for_you != ""
        assert analysis.worst_map_for_you != ""
        assert analysis.best_map_for_you != analysis.worst_map_for_you

    def test_best_map_is_strongest(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        # de_inferno should be best (3-0)
        assert analysis.best_map_for_you == "de_inferno"

    def test_worst_map_is_weakest(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        # de_nuke should be worst (0-2)
        assert analysis.worst_map_for_you == "de_nuke"

    def test_decider_map_set(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")
        assert analysis.predicted_decider_map != ""

    def test_confidence_with_many_demos(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")
        assert analysis.confidence in ("high", "medium")

    def test_confidence_low_with_few_demos(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(
            [_make_demo("de_mirage", 13, 8)],
            [_make_demo("de_nuke", 13, 5)],
            format="bo1",
        )
        assert analysis.confidence == "low"

    def test_map_pool_complete(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import ACTIVE_MAP_POOL, VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")

        your_maps = {ms.map_name for ms in analysis.your_map_pool}
        opp_maps = {ms.map_name for ms in analysis.opponent_map_pool}

        for m in ACTIVE_MAP_POOL:
            assert m in your_maps
            assert m in opp_maps

    def test_format_stored(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        bo1 = opt.analyze(diverse_demos, opponent_demos, format="bo1")
        bo3 = opt.analyze(diverse_demos, opponent_demos, format="bo3")
        assert bo1.format == "bo1"
        assert bo3.format == "bo3"

    def test_to_dict(self, diverse_demos, opponent_demos):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, opponent_demos, format="bo1")
        d = analysis.to_dict()

        assert "your_map_pool" in d
        assert "opponent_map_pool" in d
        assert "recommended_veto_sequence" in d
        assert "best_map_for_you" in d
        assert "predicted_decider_map" in d
        assert d["format"] == "bo1"


# =============================================================================
# TestWithScoutingData
# =============================================================================


class TestWithScoutingData:
    def test_opponent_scouting_dict(self, diverse_demos, scouting_data):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, scouting_data, format="bo1")

        # Should still produce valid analysis
        assert len(analysis.recommended_veto_sequence) == 7
        assert analysis.best_map_for_you != ""

    def test_opponent_maps_from_scouting(self, diverse_demos, scouting_data):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze(diverse_demos, scouting_data, format="bo1")

        opp_dict = {ms.map_name: ms for ms in analysis.opponent_map_pool}
        # Scouted maps should have higher win rate than default
        assert opp_dict["de_nuke"].win_rate >= 0.5
        assert opp_dict["de_inferno"].win_rate >= 0.5

        # Unplayed maps should have default 0.5
        assert opp_dict["de_dust2"].win_rate == 0.5


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    def test_empty_your_demos(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze([], [_make_demo("de_nuke", 13, 5)], format="bo1")

        # All your maps should be at default 0.5
        for ms in analysis.your_map_pool:
            assert ms.win_rate == 0.5
        assert len(analysis.recommended_veto_sequence) == 7

    def test_empty_both(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        analysis = opt.analyze([], [], format="bo1")

        assert len(analysis.recommended_veto_sequence) == 7
        assert analysis.confidence == "low"

    def test_same_map_all_demos(self):
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        demos = [_make_demo("de_dust2", 13, 8) for _ in range(10)]
        analysis = opt.analyze(demos, demos, format="bo1")

        # Should still work — other maps at default
        assert len(analysis.recommended_veto_sequence) == 7

    def test_tied_score_is_loss(self):
        """Score 12-12 means score_ct == score_t, which is not a win."""
        from opensight.ai.veto_optimizer import VetoOptimizer

        opt = VetoOptimizer()
        demos = [_make_demo("de_mirage", 12, 12)]
        strengths = opt._compute_map_strengths(demos)
        assert strengths[0].win_rate == 0.0


# =============================================================================
# TestSingleton
# =============================================================================


class TestSingleton:
    def test_get_veto_optimizer(self):
        from opensight.ai.veto_optimizer import get_veto_optimizer

        opt1 = get_veto_optimizer()
        opt2 = get_veto_optimizer()
        assert opt1 is opt2


# =============================================================================
# TestActiveMapPool
# =============================================================================


class TestActiveMapPool:
    def test_seven_maps(self):
        from opensight.ai.veto_optimizer import ACTIVE_MAP_POOL

        assert len(ACTIVE_MAP_POOL) == 7

    def test_all_prefixed(self):
        from opensight.ai.veto_optimizer import ACTIVE_MAP_POOL

        for m in ACTIVE_MAP_POOL:
            assert m.startswith("de_")


# =============================================================================
# TestVetoAPI
# =============================================================================


class TestVetoAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from opensight.api import app

        return TestClient(app)

    def test_missing_job_ids(self, client):
        resp = client.post(
            "/api/veto/analyze",
            json={
                "your_job_ids": [],
                "opponent_job_ids": ["some-id"],
                "format": "bo1",
            },
        )
        assert resp.status_code == 400
        assert "At least one job ID" in resp.json()["detail"]

    def test_invalid_format(self, client):
        resp = client.post(
            "/api/veto/analyze",
            json={
                "your_job_ids": ["some-id"],
                "opponent_job_ids": ["some-id"],
                "format": "bo5",
            },
        )
        assert resp.status_code == 400
        assert "bo1" in resp.json()["detail"]

    def test_no_opponent_data(self, client):
        resp = client.post(
            "/api/veto/analyze",
            json={
                "your_job_ids": ["some-id"],
                "format": "bo1",
            },
        )
        assert resp.status_code == 400
        assert "opponent" in resp.json()["detail"].lower()

    def test_job_not_found(self, client):
        resp = client.post(
            "/api/veto/analyze",
            json={
                "your_job_ids": ["nonexistent-id"],
                "opponent_scouting": {"map_tendencies": []},
                "format": "bo1",
            },
        )
        assert resp.status_code == 404

    @patch("opensight.api.routes_match._get_job_store")
    def test_success_with_scouting(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.result = _make_demo("de_mirage", 13, 8)
        mock_store.get_job.return_value = mock_job
        mock_get_store.return_value = mock_store

        resp = client.post(
            "/api/veto/analyze",
            json={
                "your_job_ids": ["job-1"],
                "opponent_scouting": {
                    "map_tendencies": [
                        {
                            "map_name": "de_nuke",
                            "t_side": {"aggression": 60},
                            "ct_side": {"aggression": 40},
                        }
                    ],
                    "demos_analyzed": 3,
                },
                "format": "bo1",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "your_map_pool" in data
        assert "recommended_veto_sequence" in data
        assert len(data["recommended_veto_sequence"]) == 7

    @patch("opensight.api.routes_match._get_job_store")
    def test_success_bo3(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "completed"
        mock_job.result = _make_demo("de_inferno", 13, 10)
        mock_store.get_job.return_value = mock_job
        mock_get_store.return_value = mock_store

        resp = client.post(
            "/api/veto/analyze",
            json={
                "your_job_ids": ["job-1"],
                "opponent_job_ids": ["job-2"],
                "format": "bo3",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "bo3"
        actions = [v["action"] for v in data["recommended_veto_sequence"]]
        assert actions == ["ban", "ban", "pick", "pick", "ban", "ban", "pick"]

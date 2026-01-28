"""Tests for the timeline graph functionality."""


class MockKillEvent:
    """Mock kill event for testing."""

    def __init__(
        self,
        tick=1000,
        round_num=1,
        attacker_steamid=12345,
        victim_steamid=67890,
        attacker_side="CT",
        victim_side="T",
        weapon="ak47",
    ):
        self.tick = tick
        self.round_num = round_num
        self.attacker_steamid = attacker_steamid
        self.victim_steamid = victim_steamid
        self.attacker_side = attacker_side
        self.victim_side = victim_side
        self.weapon = weapon


class MockDamageEvent:
    """Mock damage event for testing."""

    def __init__(
        self,
        tick=1000,
        round_num=1,
        attacker_steamid=12345,
        damage=50,
    ):
        self.tick = tick
        self.round_num = round_num
        self.attacker_steamid = attacker_steamid
        self.damage = damage


class MockBlindEvent:
    """Mock blind event for testing."""

    def __init__(
        self,
        tick=1000,
        round_num=1,
        attacker_steamid=12345,
        blind_duration=2.0,
        is_teammate=False,
    ):
        self.tick = tick
        self.round_num = round_num
        self.attacker_steamid = attacker_steamid
        self.blind_duration = blind_duration
        self.is_teammate = is_teammate


class MockRoundInfo:
    """Mock round info for testing."""

    def __init__(self, round_num=1, start_tick=0, end_tick=5000):
        self.round_num = round_num
        self.start_tick = start_tick
        self.end_tick = end_tick


class MockDemoData:
    """Mock demo data for testing."""

    def __init__(self):
        self.kills = []
        self.damages = []
        self.blinds = []
        self.rounds = []
        self.player_names = {}
        self.player_teams = {}
        self.num_rounds = 0


class TestTimelineGraphData:
    """Tests for timeline graph data building."""

    def test_empty_demo_data(self):
        """Empty demo data returns empty timeline."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()

        result = analyzer._build_timeline_graph_data(demo_data)

        assert result["max_rounds"] == 1
        assert result["players"] == []

    def test_single_kill_tracks_correctly(self):
        """Single kill is tracked for attacker and death for victim."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.kills = [
            MockKillEvent(round_num=1, attacker_steamid=111, victim_steamid=222, weapon="ak47")
        ]
        demo_data.player_names = {111: "Player1", 222: "Player2"}
        demo_data.player_teams = {111: "CT", 222: "T"}
        demo_data.num_rounds = 1

        result = analyzer._build_timeline_graph_data(demo_data)

        assert result["max_rounds"] == 1
        assert len(result["players"]) == 2

        # Find attacker's timeline
        attacker = next(p for p in result["players"] if p["steam_id"] == 111)
        assert attacker["name"] == "Player1"
        assert attacker["team"] == "CT"
        assert attacker["rounds"][0]["kills"] == 1
        assert attacker["rounds"][0]["deaths"] == 0

        # Find victim's timeline
        victim = next(p for p in result["players"] if p["steam_id"] == 222)
        assert victim["name"] == "Player2"
        assert victim["team"] == "T"
        assert victim["rounds"][0]["kills"] == 0
        assert victim["rounds"][0]["deaths"] == 1

    def test_awp_kills_tracked_separately(self):
        """AWP kills are tracked in awp_kills field."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.kills = [
            MockKillEvent(round_num=1, attacker_steamid=111, victim_steamid=222, weapon="awp"),
            MockKillEvent(round_num=1, attacker_steamid=111, victim_steamid=333, weapon="ak47"),
        ]
        demo_data.player_names = {111: "AWPer", 222: "Victim1", 333: "Victim2"}
        demo_data.num_rounds = 1

        result = analyzer._build_timeline_graph_data(demo_data)

        attacker = next(p for p in result["players"] if p["steam_id"] == 111)
        assert attacker["rounds"][0]["kills"] == 2
        assert attacker["rounds"][0]["awp_kills"] == 1

    def test_cumulative_stats_over_rounds(self):
        """Stats accumulate correctly over multiple rounds."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.kills = [
            MockKillEvent(round_num=1, attacker_steamid=111, victim_steamid=222, weapon="ak47"),
            MockKillEvent(round_num=2, attacker_steamid=111, victim_steamid=222, weapon="ak47"),
            MockKillEvent(round_num=2, attacker_steamid=111, victim_steamid=333, weapon="ak47"),
            MockKillEvent(round_num=3, attacker_steamid=111, victim_steamid=222, weapon="ak47"),
        ]
        demo_data.player_names = {111: "Fragger"}
        demo_data.num_rounds = 3

        result = analyzer._build_timeline_graph_data(demo_data)

        fragger = next(p for p in result["players"] if p["steam_id"] == 111)

        # Check cumulative kills per round
        assert fragger["rounds"][0]["kills"] == 1  # Round 1: 1 kill total
        assert fragger["rounds"][1]["kills"] == 3  # Round 2: 3 kills total
        assert fragger["rounds"][2]["kills"] == 4  # Round 3: 4 kills total

    def test_damage_tracking(self):
        """Damage is tracked and accumulated."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.damages = [
            MockDamageEvent(round_num=1, attacker_steamid=111, damage=50),
            MockDamageEvent(round_num=1, attacker_steamid=111, damage=30),
            MockDamageEvent(round_num=2, attacker_steamid=111, damage=100),
        ]
        demo_data.player_names = {111: "Damager"}
        demo_data.num_rounds = 2

        result = analyzer._build_timeline_graph_data(demo_data)

        damager = next(p for p in result["players"] if p["steam_id"] == 111)
        assert damager["rounds"][0]["damage"] == 80  # 50 + 30
        assert damager["rounds"][1]["damage"] == 180  # 80 + 100

    def test_enemy_flashes_tracked(self):
        """Enemy flashes (duration > 0.5, not teammate) are tracked."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.blinds = [
            # Valid enemy flash
            MockBlindEvent(
                round_num=1, attacker_steamid=111, blind_duration=2.0, is_teammate=False
            ),
            # Too short
            MockBlindEvent(
                round_num=1, attacker_steamid=111, blind_duration=0.3, is_teammate=False
            ),
            # Teammate flash (should not count)
            MockBlindEvent(round_num=1, attacker_steamid=111, blind_duration=2.0, is_teammate=True),
            # Another valid enemy flash
            MockBlindEvent(
                round_num=2, attacker_steamid=111, blind_duration=1.5, is_teammate=False
            ),
        ]
        demo_data.player_names = {111: "Flasher"}
        demo_data.num_rounds = 2

        result = analyzer._build_timeline_graph_data(demo_data)

        flasher = next(p for p in result["players"] if p["steam_id"] == 111)
        assert flasher["rounds"][0]["enemies_flashed"] == 1  # Only 1 valid flash
        assert flasher["rounds"][1]["enemies_flashed"] == 2  # Cumulative: 1 + 1

    def test_players_sorted_by_team(self):
        """Players are sorted CT first, then T, then Unknown."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.kills = [
            MockKillEvent(
                round_num=1,
                attacker_steamid=111,
                victim_steamid=999,
                attacker_side="T",
            ),
            MockKillEvent(
                round_num=1,
                attacker_steamid=222,
                victim_steamid=999,
                attacker_side="CT",
            ),
            MockKillEvent(
                round_num=1,
                attacker_steamid=333,
                victim_steamid=999,
                attacker_side="Unknown",
            ),
        ]
        demo_data.player_names = {111: "T_Player", 222: "CT_Player", 333: "Unknown"}
        demo_data.player_teams = {111: "T", 222: "CT", 333: "Unknown"}
        demo_data.num_rounds = 1

        result = analyzer._build_timeline_graph_data(demo_data)

        # CT should come first
        teams = [p["team"] for p in result["players"]]
        ct_idx = next(i for i, t in enumerate(teams) if t == "CT")
        t_idx = next(i for i, t in enumerate(teams) if t == "T")
        assert ct_idx < t_idx

    def test_round_level_stats_included(self):
        """Per-round stats are included for tooltips."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.kills = [
            MockKillEvent(round_num=1, attacker_steamid=111, victim_steamid=222, weapon="ak47"),
            MockKillEvent(round_num=2, attacker_steamid=111, victim_steamid=222, weapon="ak47"),
            MockKillEvent(round_num=2, attacker_steamid=111, victim_steamid=333, weapon="ak47"),
        ]
        demo_data.player_names = {111: "Player1"}
        demo_data.num_rounds = 2

        result = analyzer._build_timeline_graph_data(demo_data)

        player = next(p for p in result["players"] if p["steam_id"] == 111)
        assert player["rounds"][0]["round_kills"] == 1  # Per-round kills
        assert player["rounds"][1]["round_kills"] == 2  # Per-round kills


class TestTimelineGraphDataEdgeCases:
    """Edge case tests for timeline graph data."""

    def test_missing_round_num_uses_fallback(self):
        """When round_num is 0, use fallback logic."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.kills = [MockKillEvent(round_num=0, attacker_steamid=111, victim_steamid=222)]
        demo_data.player_names = {111: "Player1"}

        # No round boundaries, so should fallback to round 1
        result = analyzer._build_timeline_graph_data(demo_data)

        player = next(p for p in result["players"] if p["steam_id"] == 111)
        assert len(player["rounds"]) >= 1

    def test_missing_attacker_id_skipped(self):
        """Kills without attacker_id are skipped."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.kills = [MockKillEvent(round_num=1, attacker_steamid=0, victim_steamid=222)]

        result = analyzer._build_timeline_graph_data(demo_data)

        # Only victim should be tracked (for death)
        assert len(result["players"]) == 1
        assert result["players"][0]["steam_id"] == 222

    def test_team_inferred_from_kills_if_missing(self):
        """Team is inferred from kill events if not in player_teams."""
        from opensight.infra.cache import CachedAnalyzer

        analyzer = CachedAnalyzer()
        demo_data = MockDemoData()
        demo_data.kills = [
            MockKillEvent(
                round_num=1,
                attacker_steamid=111,
                victim_steamid=222,
                attacker_side="CT",
                victim_side="T",
            )
        ]
        demo_data.player_names = {111: "CTPlayer", 222: "TPlayer"}
        demo_data.player_teams = {}  # Empty - should infer

        result = analyzer._build_timeline_graph_data(demo_data)

        attacker = next(p for p in result["players"] if p["steam_id"] == 111)
        victim = next(p for p in result["players"] if p["steam_id"] == 222)
        assert attacker["team"] == "CT"
        assert victim["team"] == "T"

"""Tests for the 2D Replay Viewer functionality."""

import io

from fastapi.testclient import TestClient

from opensight.api import app
from opensight.visualization.radar import (
    MAP_DATA,
    CoordinateTransformer,
    RadarPosition,
    get_map_metadata,
)
from opensight.visualization.replay import (
    BombFrame,
    GrenadeFrame,
    KillEvent,
    MatchReplay,
    PlayerFrame,
    ReplayExporter,
    ReplayFrame,
    RoundReplay,
)

client = TestClient(app)


# =============================================================================
# Data Structure Tests
# =============================================================================


class TestPlayerFrame:
    """Tests for PlayerFrame dataclass."""

    def test_player_frame_creation(self):
        """PlayerFrame can be created with all fields."""
        frame = PlayerFrame(
            steam_id=76561198012345678,
            name="TestPlayer",
            team="CT",
            x=100.0,
            y=200.0,
            z=50.0,
            yaw=90.0,
            pitch=0.0,
            health=100,
            armor=100,
            is_alive=True,
            is_scoped=False,
            is_crouching=False,
            active_weapon="weapon_ak47",
            money=4750,
            equipment_value=4100,
        )
        assert frame.name == "TestPlayer"
        assert frame.team == "CT"
        assert frame.health == 100
        assert frame.is_alive is True

    def test_player_frame_to_dict(self):
        """PlayerFrame converts to dict correctly."""
        frame = PlayerFrame(
            steam_id=76561198012345678,
            name="TestPlayer",
            team="T",
            x=123.456,
            y=789.012,
            z=100.0,
            yaw=45.678,
            pitch=10.0,
            health=75,
            armor=50,
            is_alive=True,
            is_scoped=True,
            is_crouching=True,
        )
        d = frame.to_dict()

        assert d["sid"] == 76561198012345678
        assert d["n"] == "TestPlayer"
        assert d["t"] == "T"
        assert d["x"] == 123.5  # Rounded to 1 decimal
        assert d["y"] == 789.0
        assert d["hp"] == 75
        assert d["alive"] is True
        assert d["scoped"] is True
        assert d["crouch"] is True


class TestGrenadeFrame:
    """Tests for GrenadeFrame dataclass."""

    def test_grenade_frame_creation(self):
        """GrenadeFrame can be created."""
        frame = GrenadeFrame(
            grenade_id=1,
            grenade_type="flashbang",
            x=500.0,
            y=600.0,
            z=100.0,
            thrower_steam_id=76561198012345678,
            is_active=True,
        )
        assert frame.grenade_type == "flashbang"
        assert frame.is_active is True

    def test_grenade_frame_to_dict(self):
        """GrenadeFrame converts to dict correctly."""
        frame = GrenadeFrame(
            grenade_id=42,
            grenade_type="smoke",
            x=100.5,
            y=200.5,
            z=50.5,
            thrower_steam_id=12345,
            is_active=True,
        )
        d = frame.to_dict()

        assert d["id"] == 42
        assert d["type"] == "smoke"
        assert d["x"] == 100.5
        assert d["thrower"] == 12345
        assert d["active"] is True


class TestBombFrame:
    """Tests for BombFrame dataclass."""

    def test_bomb_frame_creation(self):
        """BombFrame can be created."""
        frame = BombFrame(
            x=100.0,
            y=200.0,
            z=50.0,
            state="planted",
            time_remaining=35.5,
        )
        assert frame.state == "planted"
        assert frame.time_remaining == 35.5

    def test_bomb_frame_to_dict(self):
        """BombFrame converts to dict correctly."""
        frame = BombFrame(
            x=100.0,
            y=200.0,
            z=50.0,
            state="planted",
            carrier_steam_id=None,
            plant_progress=100.0,
            defuse_progress=25.0,
            time_remaining=30.5,
        )
        d = frame.to_dict()

        assert d["state"] == "planted"
        assert d["plant_pct"] == 100.0
        assert d["defuse_pct"] == 25.0
        assert d["timer"] == 30.5


class TestKillEvent:
    """Tests for KillEvent dataclass."""

    def test_kill_event_creation(self):
        """KillEvent can be created."""
        event = KillEvent(
            tick=12345,
            round_num=5,
            attacker_steam_id=111,
            attacker_name="Player1",
            victim_steam_id=222,
            victim_name="Player2",
            weapon="ak47",
            headshot=True,
            x=500.0,
            y=600.0,
        )
        assert event.attacker_name == "Player1"
        assert event.headshot is True

    def test_kill_event_to_dict(self):
        """KillEvent converts to dict correctly."""
        event = KillEvent(
            tick=12345,
            round_num=5,
            attacker_steam_id=111,
            attacker_name="Attacker",
            victim_steam_id=222,
            victim_name="Victim",
            weapon="awp",
            headshot=True,
            x=100.5,
            y=200.5,
        )
        d = event.to_dict()

        assert d["tick"] == 12345
        assert d["attacker"] == "Attacker"
        assert d["victim"] == "Victim"
        assert d["weapon"] == "awp"
        assert d["hs"] is True


class TestReplayFrame:
    """Tests for ReplayFrame dataclass."""

    def test_replay_frame_creation(self):
        """ReplayFrame can be created with players."""
        players = [
            PlayerFrame(
                steam_id=1,
                name="P1",
                team="CT",
                x=0,
                y=0,
                z=0,
                yaw=0,
                pitch=0,
                health=100,
                armor=100,
                is_alive=True,
                is_scoped=False,
                is_crouching=False,
            ),
        ]
        frame = ReplayFrame(
            tick=1000,
            round_num=1,
            game_time=15.5,
            players=players,
        )
        assert frame.tick == 1000
        assert frame.round_num == 1
        assert len(frame.players) == 1

    def test_replay_frame_to_dict(self):
        """ReplayFrame converts to dict correctly."""
        players = [
            PlayerFrame(
                steam_id=1,
                name="P1",
                team="CT",
                x=0,
                y=0,
                z=0,
                yaw=0,
                pitch=0,
                health=100,
                armor=100,
                is_alive=True,
                is_scoped=False,
                is_crouching=False,
            ),
        ]
        frame = ReplayFrame(
            tick=1000,
            round_num=3,
            game_time=25.5,
            players=players,
        )
        d = frame.to_dict()

        assert d["tick"] == 1000
        assert d["round"] == 3
        assert d["time"] == 25.5
        assert len(d["players"]) == 1


class TestRoundReplay:
    """Tests for RoundReplay dataclass."""

    def test_round_replay_creation(self):
        """RoundReplay can be created."""
        replay = RoundReplay(
            round_num=1,
            start_tick=1000,
            end_tick=5000,
            winner="CT",
            win_reason="elimination",
            ct_score=1,
            t_score=0,
        )
        assert replay.round_num == 1
        assert replay.winner == "CT"

    def test_round_replay_duration(self):
        """RoundReplay calculates duration correctly."""
        replay = RoundReplay(
            round_num=1,
            start_tick=1000,
            end_tick=5000,
            winner="T",
            win_reason="bomb_exploded",
            ct_score=0,
            t_score=1,
        )
        assert replay.duration_ticks == 4000
        assert replay.duration_seconds == 62.5  # 4000 / 64


class TestMatchReplay:
    """Tests for MatchReplay dataclass."""

    def test_match_replay_creation(self):
        """MatchReplay can be created."""
        replay = MatchReplay(
            map_name="de_dust2",
            tick_rate=64,
            sample_rate=8,
            total_rounds=30,
            team1_score=16,
            team2_score=14,
            player_names={1: "Player1", 2: "Player2"},
            player_teams={1: "CT", 2: "T"},
        )
        assert replay.map_name == "de_dust2"
        assert replay.tick_rate == 64

    def test_match_replay_to_dict(self):
        """MatchReplay converts to dict correctly."""
        replay = MatchReplay(
            map_name="de_mirage",
            tick_rate=64,
            sample_rate=8,
            total_rounds=24,
            team1_score=13,
            team2_score=11,
            player_names={1: "P1"},
            player_teams={1: "CT"},
        )
        d = replay.to_dict()

        assert d["map_name"] == "de_mirage"
        assert d["tick_rate"] == 64
        assert d["fps"] == 8  # 64 / 8
        assert d["score"] == "13 - 11"

    def test_match_replay_get_round(self):
        """MatchReplay can retrieve specific round."""
        round1 = RoundReplay(
            round_num=1,
            start_tick=0,
            end_tick=1000,
            winner="CT",
            win_reason="elim",
            ct_score=1,
            t_score=0,
        )
        round2 = RoundReplay(
            round_num=2,
            start_tick=1000,
            end_tick=2000,
            winner="T",
            win_reason="bomb",
            ct_score=1,
            t_score=1,
        )
        replay = MatchReplay(
            map_name="de_dust2",
            tick_rate=64,
            sample_rate=8,
            total_rounds=2,
            team1_score=1,
            team2_score=1,
            player_names={},
            player_teams={},
            rounds=[round1, round2],
        )

        assert replay.get_round(1) == round1
        assert replay.get_round(2) == round2
        assert replay.get_round(3) is None


# =============================================================================
# Coordinate Transformation Tests
# =============================================================================


class TestCoordinateTransformer:
    """Tests for CoordinateTransformer."""

    def test_transformer_known_map(self):
        """CoordinateTransformer initializes for known maps."""
        transformer = CoordinateTransformer("de_dust2")
        assert transformer.map_name == "de_dust2"
        assert transformer.metadata.name == "Dust II"

    def test_transformer_unknown_map(self):
        """CoordinateTransformer handles unknown maps."""
        transformer = CoordinateTransformer("de_unknown_map")
        assert transformer.map_name == "de_unknown_map"
        # Should use default values
        assert transformer.metadata.scale == 5.0

    def test_game_to_radar_conversion(self):
        """CoordinateTransformer converts game to radar coords."""
        transformer = CoordinateTransformer("de_dust2")

        # Test conversion at a known point
        pos = transformer.game_to_radar(0.0, 0.0, 0.0)
        assert isinstance(pos, RadarPosition)
        assert pos.is_valid is True
        assert pos.x >= 0
        assert pos.y >= 0

    def test_radar_to_game_conversion(self):
        """CoordinateTransformer converts radar to game coords."""
        transformer = CoordinateTransformer("de_mirage")

        # Convert and convert back
        game_x, game_y = transformer.radar_to_game(512.0, 512.0)
        pos = transformer.game_to_radar(game_x, game_y, 0)

        # Should be close to original
        assert abs(pos.x - 512.0) < 1.0
        assert abs(pos.y - 512.0) < 1.0

    def test_is_upper_level(self):
        """CoordinateTransformer detects upper level correctly."""
        # Nuke has z_cutoff
        transformer = CoordinateTransformer("de_nuke")
        assert transformer.is_upper_level(0) is True  # Above cutoff
        assert transformer.is_upper_level(-600) is False  # Below cutoff

        # Dust2 has no z_cutoff
        transformer2 = CoordinateTransformer("de_dust2")
        assert transformer2.is_upper_level(0) is True
        assert transformer2.is_upper_level(-9999) is True


class TestMapMetadata:
    """Tests for map metadata functions."""

    def test_get_map_metadata_known_map(self):
        """get_map_metadata returns data for known maps."""
        metadata = get_map_metadata("de_dust2")
        assert metadata is not None
        assert metadata.name == "Dust II"
        assert metadata.internal_name == "de_dust2"
        assert metadata.radar_url is not None

    def test_get_map_metadata_unknown_map(self):
        """get_map_metadata returns None for unknown maps."""
        metadata = get_map_metadata("de_nonexistent")
        assert metadata is None

    def test_all_maps_have_required_data(self):
        """All maps in MAP_DATA have required fields."""
        required_fields = ["name", "pos_x", "pos_y", "scale", "radar_url"]
        for map_name, data in MAP_DATA.items():
            for field in required_fields:
                assert field in data, f"Map {map_name} missing {field}"


# =============================================================================
# Replay Export Tests
# =============================================================================


class TestReplayExporter:
    """Tests for ReplayExporter."""

    def test_to_json_basic(self):
        """ReplayExporter.to_json produces valid JSON."""
        replay = MatchReplay(
            map_name="de_dust2",
            tick_rate=64,
            sample_rate=8,
            total_rounds=1,
            team1_score=1,
            team2_score=0,
            player_names={1: "Test"},
            player_teams={1: "CT"},
        )

        json_str = ReplayExporter.to_json(replay)
        assert isinstance(json_str, str)
        assert "de_dust2" in json_str

    def test_to_json_pretty(self):
        """ReplayExporter.to_json can produce pretty JSON."""
        replay = MatchReplay(
            map_name="de_dust2",
            tick_rate=64,
            sample_rate=8,
            total_rounds=1,
            team1_score=0,
            team2_score=0,
            player_names={},
            player_teams={},
        )

        json_str = ReplayExporter.to_json(replay, pretty=True)
        assert "\n" in json_str  # Pretty print has newlines


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestReplayGenerateEndpoint:
    """Tests for the /replay/generate endpoint."""

    def test_replay_generate_rejects_non_dem(self):
        """Replay generate rejects non-.dem files."""
        files = {"file": ("test.txt", io.BytesIO(b"not a demo"), "text/plain")}
        response = client.post("/replay/generate", files=files)
        assert response.status_code == 400
        assert ".dem" in response.json()["detail"]

    def test_replay_generate_rejects_empty(self):
        """Replay generate rejects empty files."""
        files = {"file": ("test.dem", io.BytesIO(b""), "application/octet-stream")}
        response = client.post("/replay/generate", files=files)
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    def test_replay_generate_sample_rate_validation(self):
        """Replay generate validates sample_rate parameter."""
        files = {"file": ("test.dem", io.BytesIO(b"DEMO"), "application/octet-stream")}

        # Too low
        response = client.post("/replay/generate?sample_rate=0", files=files)
        assert response.status_code == 422

        # Too high
        response = client.post("/replay/generate?sample_rate=200", files=files)
        assert response.status_code == 422


class TestMapsEndpoint:
    """Tests for map-related endpoints."""

    def test_list_maps(self):
        """Maps endpoint returns list of maps."""
        response = client.get("/maps")
        assert response.status_code == 200
        data = response.json()

        assert "maps" in data
        assert len(data["maps"]) > 0

        # Each map should have required fields
        for map_info in data["maps"]:
            assert "internal_name" in map_info
            assert "display_name" in map_info
            assert "has_radar" in map_info

    def test_get_map_info_known_map(self):
        """Map info endpoint returns data for known maps."""
        response = client.get("/maps/de_dust2")
        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "Dust II"
        assert data["internal_name"] == "de_dust2"
        assert "radar_url" in data
        assert "pos_x" in data
        assert "pos_y" in data
        assert "scale" in data

    def test_get_map_info_unknown_map(self):
        """Map info endpoint returns 404 for unknown maps."""
        response = client.get("/maps/de_nonexistent")
        assert response.status_code == 404


class TestRadarTransformEndpoint:
    """Tests for radar coordinate transformation endpoint."""

    def test_transform_coordinates(self):
        """Transform endpoint converts coordinates correctly."""
        response = client.post(
            "/radar/transform",
            json={
                "map_name": "de_dust2",
                "positions": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 1000.0, "y": 1000.0, "z": 0.0},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["map_name"] == "de_dust2"
        assert len(data["positions"]) == 2

        for pos in data["positions"]:
            assert "game" in pos
            assert "radar" in pos
            assert "is_upper_level" in pos

    def test_transform_missing_map_name(self):
        """Transform endpoint requires map_name."""
        response = client.post(
            "/radar/transform",
            json={"positions": [{"x": 0, "y": 0, "z": 0}]},
        )
        assert response.status_code == 422


# =============================================================================
# Integration Tests
# =============================================================================


class TestReplayDataFlow:
    """Integration tests for replay data flow."""

    def test_replay_frame_serialization_roundtrip(self):
        """ReplayFrame data survives serialization roundtrip."""
        import json

        players = [
            PlayerFrame(
                steam_id=76561198012345678,
                name="TestPlayer",
                team="CT",
                x=100.5,
                y=200.5,
                z=50.0,
                yaw=90.0,
                pitch=0.0,
                health=85,
                armor=100,
                is_alive=True,
                is_scoped=False,
                is_crouching=False,
                active_weapon="weapon_ak47",
            ),
        ]
        frame = ReplayFrame(
            tick=5000,
            round_num=3,
            game_time=45.5,
            players=players,
        )

        # Serialize and deserialize
        json_str = json.dumps(frame.to_dict())
        data = json.loads(json_str)

        assert data["tick"] == 5000
        assert data["round"] == 3
        assert len(data["players"]) == 1
        assert data["players"][0]["n"] == "TestPlayer"
        assert data["players"][0]["hp"] == 85

    def test_coordinate_transform_preserves_precision(self):
        """Coordinate transformation maintains reasonable precision."""
        transformer = CoordinateTransformer("de_mirage")

        # Test point
        game_x, game_y = 1500.0, 500.0

        # Convert to radar
        radar_pos = transformer.game_to_radar(game_x, game_y, 0)

        # Convert back
        back_x, back_y = transformer.radar_to_game(radar_pos.x, radar_pos.y)

        # Should be within small tolerance
        assert abs(back_x - game_x) < 1.0
        assert abs(back_y - game_y) < 1.0

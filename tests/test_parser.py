"""Comprehensive tests for the parser module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from opensight.parser import (
    DemoParser,
    DemoData,
    PlayerState,
    GameEvent,
    parse_demo,
)


class TestPlayerState:
    """Tests for PlayerState dataclass."""

    def test_creation(self):
        """Test creating a PlayerState."""
        state = PlayerState(
            steam_id=12345,
            name="TestPlayer",
            team="CT",
            position=(100.0, 200.0, 50.0),
            view_angles=(5.0, 90.0),
            health=100,
            armor=100,
            is_alive=True,
            weapon="ak47",
            tick=1000,
        )
        assert state.steam_id == 12345
        assert state.name == "TestPlayer"
        assert state.team == "CT"
        assert state.position == (100.0, 200.0, 50.0)
        assert state.view_angles == (5.0, 90.0)
        assert state.health == 100
        assert state.armor == 100
        assert state.is_alive is True
        assert state.weapon == "ak47"
        assert state.tick == 1000

    def test_dead_player_state(self):
        """Test player state when dead."""
        state = PlayerState(
            steam_id=12345,
            name="DeadPlayer",
            team="T",
            position=(0.0, 0.0, 0.0),
            view_angles=(0.0, 0.0),
            health=0,
            armor=0,
            is_alive=False,
            weapon="knife",
            tick=2000,
        )
        assert state.is_alive is False
        assert state.health == 0


class TestGameEvent:
    """Tests for GameEvent dataclass."""

    def test_creation(self):
        """Test creating a GameEvent."""
        event = GameEvent(
            event_type="player_death",
            tick=5000,
            data={"attacker": 123, "victim": 456, "weapon": "awp"},
        )
        assert event.event_type == "player_death"
        assert event.tick == 5000
        assert event.data["attacker"] == 123
        assert event.data["victim"] == 456
        assert event.data["weapon"] == "awp"

    def test_empty_data(self):
        """Test event with empty data."""
        event = GameEvent(
            event_type="round_start",
            tick=100,
            data={},
        )
        assert event.data == {}


class TestDemoData:
    """Tests for DemoData dataclass."""

    @pytest.fixture
    def sample_demo_data(self):
        """Create sample DemoData for testing."""
        return DemoData(
            file_path=Path("/test/demo.dem"),
            map_name="de_dust2",
            tick_rate=64,
            duration_ticks=128000,
            duration_seconds=2000.0,
            player_names={1: "Player1", 2: "Player2"},
            teams={1: "CT", 2: "T"},
            player_positions=pd.DataFrame({
                "tick": [100, 100, 200, 200],
                "steam_id": [1, 2, 1, 2],
                "x": [100.0, -100.0, 105.0, -105.0],
                "y": [200.0, -200.0, 205.0, -205.0],
                "z": [50.0, 50.0, 50.0, 50.0],
                "pitch": [0.0, 0.0, 5.0, -5.0],
                "yaw": [90.0, 270.0, 92.0, 268.0],
            }),
            player_health=pd.DataFrame({
                "tick": [100, 100, 200, 200],
                "steam_id": [1, 2, 1, 2],
                "health": [100, 100, 80, 100],
                "armor": [100, 50, 100, 50],
            }),
            shots_fired=pd.DataFrame({
                "tick": [150],
                "steam_id": [1],
                "weapon": ["ak47"],
            }),
            damage_events=pd.DataFrame({
                "tick": [160],
                "attacker_id": [1],
                "victim_id": [2],
                "damage": [27],
                "weapon": ["ak47"],
                "hitgroup": [1],
            }),
            kill_events=pd.DataFrame({
                "tick": [180],
                "attacker_id": [1],
                "victim_id": [2],
                "weapon": ["ak47"],
                "headshot": [True],
            }),
            round_starts=[0, 64000],
            round_ends=[60000, 120000],
        )

    def test_tick_interval(self, sample_demo_data):
        """Test tick interval calculation."""
        assert sample_demo_data.tick_interval == pytest.approx(1/64, abs=0.0001)

    def test_tick_interval_zero_tickrate(self):
        """Test tick interval with zero tick rate."""
        data = DemoData(
            file_path=Path("/test.dem"),
            map_name="test",
            tick_rate=0,
            duration_ticks=0,
            duration_seconds=0.0,
            player_names={},
            teams={},
            player_positions=pd.DataFrame(),
            player_health=pd.DataFrame(),
            shots_fired=pd.DataFrame(),
            damage_events=pd.DataFrame(),
            kill_events=pd.DataFrame(),
            round_starts=[],
            round_ends=[],
        )
        assert data.tick_interval == 0.0

    def test_get_player_state_at_tick(self, sample_demo_data):
        """Test retrieving player state at a specific tick."""
        state = sample_demo_data.get_player_state_at_tick(1, 100)
        assert state is not None
        assert state.steam_id == 1
        assert state.position == (100.0, 200.0, 50.0)
        assert state.health == 100

    def test_get_player_state_at_invalid_tick(self, sample_demo_data):
        """Test retrieving state at non-existent tick."""
        state = sample_demo_data.get_player_state_at_tick(1, 9999)
        assert state is None

    def test_get_player_state_for_invalid_player(self, sample_demo_data):
        """Test retrieving state for non-existent player."""
        state = sample_demo_data.get_player_state_at_tick(999, 100)
        assert state is None


class TestDemoParser:
    """Tests for DemoParser class."""

    def test_init_with_valid_path(self, tmp_path):
        """Test initializing parser with valid path."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        parser = DemoParser(demo_file)
        assert parser.demo_path == demo_file

    def test_init_with_nonexistent_file(self):
        """Test initializing parser with non-existent file."""
        with pytest.raises(FileNotFoundError):
            DemoParser("/nonexistent/path/demo.dem")

    def test_init_with_wrong_extension(self, tmp_path):
        """Test initializing parser with wrong file extension."""
        wrong_file = tmp_path / "test.txt"
        wrong_file.write_text("not a demo")

        with pytest.raises(ValueError, match="Expected .dem file"):
            DemoParser(wrong_file)

    def test_init_with_string_path(self, tmp_path):
        """Test initializing parser with string path."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        parser = DemoParser(str(demo_file))
        assert parser.demo_path == demo_file

    @patch('opensight.parser.Demoparser2', None)
    def test_parse_without_demoparser2(self, tmp_path):
        """Test parsing when demoparser2 is not installed."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        parser = DemoParser(demo_file)

        with pytest.raises(ImportError, match="demoparser2 is required"):
            parser.parse()

    @patch('opensight.parser.Demoparser2')
    def test_parse_caches_result(self, mock_demoparser2, tmp_path):
        """Test that parsing caches the result."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        # Set up mock
        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "de_dust2", "tickrate": 64}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1], "steamid": [1], "X": [0], "Y": [0], "Z": [0]
        })
        mock_parser.parse_events.return_value = {}
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)

        # Parse twice
        result1 = parser.parse()
        result2 = parser.parse()

        # Should be the same object (cached)
        assert result1 is result2
        # Demoparser2 should only be called once
        assert mock_demoparser2.call_count == 1

    @patch('opensight.parser.Demoparser2')
    def test_parse_extracts_map_name(self, mock_demoparser2, tmp_path):
        """Test that parse extracts map name from header."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "de_mirage", "tickrate": 128}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1], "steamid": [1], "X": [0], "Y": [0], "Z": [0]
        })
        mock_parser.parse_events.return_value = {}
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        result = parser.parse()

        assert result.map_name == "de_mirage"
        assert result.tick_rate == 128

    @patch('opensight.parser.Demoparser2')
    def test_parse_extracts_player_names(self, mock_demoparser2, tmp_path):
        """Test that parse extracts player names."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = [
            {"steamid": 123, "name": "Alice"},
            {"steamid": 456, "name": "Bob"},
        ]
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1, 1], "steamid": [123, 456], "X": [0, 0], "Y": [0, 0], "Z": [0, 0]
        })
        mock_parser.parse_events.return_value = {}
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        result = parser.parse()

        assert result.player_names[123] == "Alice"
        assert result.player_names[456] == "Bob"

    @patch('opensight.parser.Demoparser2')
    def test_parse_handles_team_mapping(self, mock_demoparser2, tmp_path):
        """Test that parse correctly maps team numbers to names."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = [
            {"steamid": 1, "name": "CT_Player"},
            {"steamid": 2, "name": "T_Player"},
        ]
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1, 1],
            "steamid": [1, 2],
            "X": [0, 0], "Y": [0, 0], "Z": [0, 0],
            "team_num": [3, 2],  # 3 = CT, 2 = T
        })
        mock_parser.parse_events.return_value = {}
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        result = parser.parse()

        assert result.teams[1] == "CT"
        assert result.teams[2] == "T"

    @patch('opensight.parser.Demoparser2')
    def test_parse_handles_missing_events(self, mock_demoparser2, tmp_path):
        """Test that parse handles missing event data gracefully."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1], "steamid": [1], "X": [0], "Y": [0], "Z": [0]
        })
        mock_parser.parse_events.side_effect = Exception("No events")
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        result = parser.parse()

        # Should still work, just with empty DataFrames
        assert result.damage_events.empty
        assert result.kill_events.empty
        assert result.shots_fired.empty

    @patch('opensight.parser.Demoparser2')
    def test_get_player_positions_between(self, mock_demoparser2, tmp_path):
        """Test get_player_positions_between method."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [100, 150, 200, 250, 300],
            "steamid": [1, 1, 1, 1, 1],
            "X": [0, 10, 20, 30, 40],
            "Y": [0, 10, 20, 30, 40],
            "Z": [0, 0, 0, 0, 0],
        })
        mock_parser.parse_events.return_value = {}
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        positions = parser.get_player_positions_between(150, 250)

        assert len(positions) == 3
        assert positions["tick"].min() == 150
        assert positions["tick"].max() == 250

    @patch('opensight.parser.Demoparser2')
    def test_get_player_positions_between_with_steam_id(self, mock_demoparser2, tmp_path):
        """Test get_player_positions_between with steam_id filter."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [100, 100, 200, 200],
            "steamid": [1, 2, 1, 2],
            "X": [0, 100, 10, 110],
            "Y": [0, 100, 10, 110],
            "Z": [0, 0, 0, 0],
        })
        mock_parser.parse_events.return_value = {}
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        positions = parser.get_player_positions_between(100, 200, steam_id=1)

        assert len(positions) == 2
        assert all(positions["steam_id"] == 1)


class TestParseDemoFunction:
    """Tests for the parse_demo convenience function."""

    @patch('opensight.parser.Demoparser2')
    def test_parse_demo_creates_parser_and_parses(self, mock_demoparser2, tmp_path):
        """Test parse_demo convenience function."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1], "steamid": [1], "X": [0], "Y": [0], "Z": [0]
        })
        mock_parser.parse_events.return_value = {}
        mock_demoparser2.return_value = mock_parser

        result = parse_demo(demo_file)

        assert isinstance(result, DemoData)
        assert result.file_path == demo_file


class TestProcessingMethods:
    """Tests for internal processing methods."""

    @patch('opensight.parser.Demoparser2')
    def test_process_damage_events(self, mock_demoparser2, tmp_path):
        """Test damage event processing."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1], "steamid": [1], "X": [0], "Y": [0], "Z": [0]
        })
        mock_parser.parse_events.return_value = {
            "player_hurt": [
                {"tick": 100, "attacker_steamid": 1, "userid_steamid": 2,
                 "dmg_health": 27, "weapon": "ak47", "hitgroup": 1}
            ]
        }
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        result = parser.parse()

        assert len(result.damage_events) == 1
        assert result.damage_events.iloc[0]["attacker_id"] == 1
        assert result.damage_events.iloc[0]["damage"] == 27

    @patch('opensight.parser.Demoparser2')
    def test_process_kill_events(self, mock_demoparser2, tmp_path):
        """Test kill event processing."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1], "steamid": [1], "X": [0], "Y": [0], "Z": [0]
        })
        mock_parser.parse_events.return_value = {
            "player_death": [
                {"tick": 200, "attacker_steamid": 1, "userid_steamid": 2,
                 "weapon": "awp", "headshot": True}
            ]
        }
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        result = parser.parse()

        assert len(result.kill_events) == 1
        assert result.kill_events.iloc[0]["headshot"] is True
        assert result.kill_events.iloc[0]["weapon"] == "awp"

    @patch('opensight.parser.Demoparser2')
    def test_process_shots(self, mock_demoparser2, tmp_path):
        """Test shots fired processing."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1], "steamid": [1], "X": [0], "Y": [0], "Z": [0]
        })
        mock_parser.parse_events.return_value = {
            "weapon_fire": [
                {"tick": 150, "userid_steamid": 1, "weapon": "m4a1_silencer"}
            ]
        }
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        result = parser.parse()

        assert len(result.shots_fired) == 1
        assert result.shots_fired.iloc[0]["weapon"] == "m4a1_silencer"

    @patch('opensight.parser.Demoparser2')
    def test_process_round_events(self, mock_demoparser2, tmp_path):
        """Test round event processing."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"demo content")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "test", "tickrate": 64}
        mock_parser.parse_player_info.return_value = []
        mock_parser.parse_ticks.return_value = pd.DataFrame({
            "tick": [1], "steamid": [1], "X": [0], "Y": [0], "Z": [0]
        })

        def parse_events_side_effect(events):
            if "round_start" in events:
                return {
                    "round_start": [{"tick": 0}, {"tick": 3200}],
                    "round_end": [{"tick": 3000}, {"tick": 6000}]
                }
            return {}

        mock_parser.parse_events.side_effect = parse_events_side_effect
        mock_demoparser2.return_value = mock_parser

        parser = DemoParser(demo_file)
        result = parser.parse()

        assert result.round_starts == [0, 3200]
        assert result.round_ends == [3000, 6000]

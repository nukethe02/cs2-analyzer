"""Tests for the parser module."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from opensight.parser import (
    DemoData,
    DemoParser,
    KillEvent,
    DamageEvent,
    PlayerState,
    safe_int,
    safe_str,
    safe_bool,
    parse_demo,
)


class TestSafeConversions:
    """Tests for safe type conversion utilities."""

    def test_safe_int_with_valid_int(self):
        """Return int when given valid int."""
        assert safe_int(42) == 42

    def test_safe_int_with_valid_float(self):
        """Return int when given valid float."""
        assert safe_int(42.7) == 42

    def test_safe_int_with_valid_string(self):
        """Return int when given valid string."""
        assert safe_int("42") == 42

    def test_safe_int_with_none(self):
        """Return default when given None."""
        assert safe_int(None) == 0
        assert safe_int(None, 99) == 99

    def test_safe_int_with_nan(self):
        """Return default when given NaN."""
        assert safe_int(float("nan")) == 0

    def test_safe_int_with_invalid_string(self):
        """Return default when given invalid string."""
        assert safe_int("not a number") == 0
        assert safe_int("not a number", 99) == 99

    def test_safe_str_with_valid_string(self):
        """Return string when given valid string."""
        assert safe_str("test") == "test"

    def test_safe_str_with_int(self):
        """Return string when given int."""
        assert safe_str(42) == "42"

    def test_safe_str_with_none(self):
        """Return default when given None."""
        assert safe_str(None) == ""
        assert safe_str(None, "default") == "default"

    def test_safe_str_with_nan(self):
        """Return default when given NaN."""
        assert safe_str(float("nan")) == ""

    def test_safe_bool_with_valid_bool(self):
        """Return bool when given valid bool."""
        assert safe_bool(True) is True
        assert safe_bool(False) is False

    def test_safe_bool_with_truthy_value(self):
        """Return True when given truthy value."""
        assert safe_bool(1) is True
        assert safe_bool("yes") is True

    def test_safe_bool_with_falsy_value(self):
        """Return False when given falsy value."""
        assert safe_bool(0) is False
        assert safe_bool("") is False

    def test_safe_bool_with_none(self):
        """Return default when given None."""
        assert safe_bool(None) is False
        assert safe_bool(None, True) is True

    def test_safe_bool_with_nan(self):
        """Return default when given NaN."""
        assert safe_bool(float("nan")) is False


class TestDataClasses:
    """Tests for data class structures."""

    def test_kill_event_creation(self):
        """KillEvent can be created with required fields."""
        event = KillEvent(
            tick=1000,
            attacker_steamid=12345,
            attacker_name="Player1",
            victim_steamid=67890,
            victim_name="Player2",
            weapon="ak47",
            headshot=True,
        )
        assert event.tick == 1000
        assert event.attacker_steamid == 12345
        assert event.headshot is True
        assert event.attacker_position is None  # Optional field

    def test_kill_event_with_positions(self):
        """KillEvent can include optional position data."""
        pos = np.array([100.0, 200.0, 300.0])
        angles = np.array([0.0, 90.0])
        event = KillEvent(
            tick=1000,
            attacker_steamid=12345,
            attacker_name="Player1",
            victim_steamid=67890,
            victim_name="Player2",
            weapon="ak47",
            headshot=False,
            attacker_position=pos,
            attacker_angles=angles,
        )
        np.testing.assert_array_equal(event.attacker_position, pos)
        np.testing.assert_array_equal(event.attacker_angles, angles)

    def test_damage_event_creation(self):
        """DamageEvent can be created with required fields."""
        event = DamageEvent(
            tick=1000,
            attacker_steamid=12345,
            victim_steamid=67890,
            damage=27,
            weapon="ak47",
            hitgroup="head",
        )
        assert event.tick == 1000
        assert event.damage == 27
        assert event.hitgroup == "head"

    def test_player_state_creation(self):
        """PlayerState can be created with required fields."""
        state = PlayerState(
            tick=1000,
            steam_id=12345,
            name="Player1",
            team=2,
            position=np.array([100.0, 200.0, 300.0]),
            eye_angles=np.array([0.0, 90.0]),
            health=100,
            is_alive=True,
        )
        assert state.tick == 1000
        assert state.team == 2
        assert state.health == 100
        assert state.is_alive is True


class TestDemoData:
    """Tests for DemoData structure."""

    def test_demo_data_creation(self):
        """DemoData can be created with required fields."""
        data = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=24,
            player_stats={},
            player_names={12345: "Player1"},
            player_teams={12345: 2},
            kills=[],
            damages=[],
            kills_df=pd.DataFrame(),
            damages_df=pd.DataFrame(),
        )
        assert data.map_name == "de_dust2"
        assert data.tick_rate == 64
        assert data.num_rounds == 24
        assert data.ticks_df is None  # Optional

    def test_demo_data_with_ticks(self):
        """DemoData can include optional tick data."""
        ticks_df = pd.DataFrame({
            "tick": [1, 2, 3],
            "steamid": [12345, 12345, 12345],
            "X": [100.0, 101.0, 102.0],
            "Y": [200.0, 200.0, 200.0],
            "Z": [0.0, 0.0, 0.0],
        })
        data = DemoData(
            file_path=Path("/tmp/test.dem"),
            map_name="de_dust2",
            duration_seconds=1800.0,
            tick_rate=64,
            num_rounds=24,
            player_stats={},
            player_names={},
            player_teams={},
            kills=[],
            damages=[],
            kills_df=pd.DataFrame(),
            damages_df=pd.DataFrame(),
            ticks_df=ticks_df,
        )
        assert data.ticks_df is not None
        assert len(data.ticks_df) == 3


class TestDemoParser:
    """Tests for DemoParser class."""

    def test_parser_raises_on_missing_file(self):
        """Parser raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            DemoParser("/nonexistent/path/demo.dem")

    def test_parser_raises_on_missing_demoparser2(self, tmp_path):
        """Parser raises ImportError when demoparser2 not available."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        with patch("opensight.parser.Demoparser2", None):
            parser = DemoParser(demo_file)
            with pytest.raises(ImportError, match="demoparser2 is required"):
                parser.parse()

    def test_parse_caches_result(self, tmp_path):
        """Parser caches the result after first parse."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        # Create mock demoparser2
        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "de_dust2"}
        mock_parser.parse_event.return_value = pd.DataFrame()

        with patch("opensight.parser.Demoparser2", return_value=mock_parser):
            parser = DemoParser(demo_file)
            result1 = parser.parse()
            result2 = parser.parse()

            # Should be same object (cached)
            assert result1 is result2
            # parse_header should only be called once
            assert mock_parser.parse_header.call_count == 1


class TestParseDemoFunction:
    """Tests for parse_demo convenience function."""

    def test_parse_demo_creates_parser_and_parses(self, tmp_path):
        """parse_demo creates parser and calls parse."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        mock_parser = MagicMock()
        mock_parser.parse_header.return_value = {"map_name": "de_mirage"}
        mock_parser.parse_event.return_value = pd.DataFrame()

        with patch("opensight.parser.Demoparser2", return_value=mock_parser):
            result = parse_demo(demo_file)
            assert result.map_name == "de_mirage"

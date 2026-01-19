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
            round_num=1,
            attacker_steamid=12345,
            attacker_name="Player1",
            attacker_side="CT",
            victim_steamid=67890,
            victim_name="Player2",
            victim_side="T",
            weapon="ak47",
            headshot=True,
        )
        assert event.tick == 1000
        assert event.attacker_steamid == 12345
        assert event.headshot is True
        assert event.attacker_x is None  # Optional field

    def test_kill_event_with_positions(self):
        """KillEvent can include optional position data."""
        event = KillEvent(
            tick=1000,
            round_num=1,
            attacker_steamid=12345,
            attacker_name="Player1",
            attacker_side="CT",
            victim_steamid=67890,
            victim_name="Player2",
            victim_side="T",
            weapon="ak47",
            headshot=False,
            attacker_x=100.0,
            attacker_y=200.0,
            attacker_z=300.0,
            attacker_pitch=0.0,
            attacker_yaw=90.0,
        )
        assert event.attacker_x == 100.0
        assert event.attacker_yaw == 90.0

    def test_damage_event_creation(self):
        """DamageEvent can be created with required fields."""
        event = DamageEvent(
            tick=1000,
            round_num=1,
            attacker_steamid=12345,
            attacker_name="Player1",
            attacker_side="CT",
            victim_steamid=67890,
            victim_name="Player2",
            victim_side="T",
            damage=27,
            damage_armor=5,
            health_remaining=73,
            armor_remaining=95,
            weapon="ak47",
            hitgroup="head",
        )
        assert event.tick == 1000
        assert event.damage == 27
        assert event.hitgroup == "head"

    def test_player_state_creation(self):
        """PlayerState (PlayerRoundSnapshot) can be created with required fields."""
        state = PlayerState(
            tick=1000,
            round_num=1,
            steamid=12345,
            name="Player1",
            side="T",
            x=100.0,
            y=200.0,
            z=300.0,
            pitch=0.0,
            yaw=90.0,
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_z=0.0,
            health=100,
            armor=100,
            is_alive=True,
            is_scoped=False,
            is_walking=False,
            is_crouching=False,
            money=4000,
            equipment_value=5500,
            place_name="TSpawn",
        )
        assert state.tick == 1000
        assert state.side == "T"
        assert state.health == 100


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

        # Patch the lazy availability check functions
        with patch("opensight.parser._check_demoparser2_available", return_value=False):
            with patch("opensight.parser._check_awpy_available", return_value=False):
                parser = DemoParser(demo_file)
                with pytest.raises(ImportError, match="No parser available"):
                    parser.parse()

    def test_parse_caches_result(self, tmp_path):
        """Parser caches the result after first parse."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        # Create mock demoparser2
        mock_demoparser2 = MagicMock()
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_header.return_value = {"map_name": "de_dust2"}
        mock_parser_instance.parse_event.return_value = pd.DataFrame()
        mock_demoparser2.return_value = mock_parser_instance

        # Patch the lazy import inside _parse_with_demoparser2
        with patch("opensight.parser._check_demoparser2_available", return_value=True):
            with patch.dict("sys.modules", {"demoparser2": MagicMock(DemoParser=mock_demoparser2)}):
                parser = DemoParser(demo_file)
                result1 = parser.parse()
                result2 = parser.parse()

                # Should be same object (cached)
                assert result1 is result2
                # parse_header should only be called once
                assert mock_parser_instance.parse_header.call_count == 1


class TestParseDemoFunction:
    """Tests for parse_demo convenience function."""

    def test_parse_demo_creates_parser_and_parses(self, tmp_path):
        """parse_demo creates parser and calls parse."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        mock_demoparser2 = MagicMock()
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_header.return_value = {"map_name": "de_mirage"}
        mock_parser_instance.parse_event.return_value = pd.DataFrame()
        mock_demoparser2.return_value = mock_parser_instance

        # Patch the lazy import inside _parse_with_demoparser2
        with patch("opensight.parser._check_demoparser2_available", return_value=True):
            with patch.dict("sys.modules", {"demoparser2": MagicMock(DemoParser=mock_demoparser2)}):
                result = parse_demo(demo_file)
                assert result.map_name == "de_mirage"


class TestOptimizations:
    """Tests for new optimization features."""

    def test_parse_mode_enum(self):
        """Test ParseMode enum values."""
        from opensight.parser import ParseMode

        assert ParseMode.MINIMAL.value == "minimal"
        assert ParseMode.STANDARD.value == "standard"
        assert ParseMode.COMPREHENSIVE.value == "comprehensive"

    def test_parser_backend_enum(self):
        """Test ParserBackend enum values."""
        from opensight.parser import ParserBackend

        assert ParserBackend.DEMOPARSER2.value == "demoparser2"
        assert ParserBackend.AWPY.value == "awpy"
        assert ParserBackend.AUTO.value == "auto"

    def test_optimize_dataframe_dtypes_empty(self):
        """Test dtype optimization on empty DataFrame."""
        from opensight.parser import optimize_dataframe_dtypes

        df = pd.DataFrame()
        result = optimize_dataframe_dtypes(df)
        assert result.empty

    def test_optimize_dataframe_dtypes_int64_to_int32(self):
        """Test int64 to int32 conversion."""
        from opensight.parser import optimize_dataframe_dtypes

        df = pd.DataFrame({
            "tick": np.array([100, 200, 300], dtype=np.int64),
            "damage": np.array([50, 75, 100], dtype=np.int64),
        })
        result = optimize_dataframe_dtypes(df.copy(), inplace=False)

        # Should convert to Int32 (nullable)
        assert result["tick"].dtype in [np.int32, "Int32"]
        assert result["damage"].dtype in [np.int32, "Int32"]

    def test_optimize_dataframe_dtypes_float64_to_float32(self):
        """Test float64 to float32 conversion for position data."""
        from opensight.parser import optimize_dataframe_dtypes

        df = pd.DataFrame({
            "X": np.array([100.5, 200.5, 300.5], dtype=np.float64),
            "Y": np.array([50.5, 75.5, 100.5], dtype=np.float64),
            "pitch": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        })
        result = optimize_dataframe_dtypes(df.copy(), inplace=False)

        assert result["X"].dtype == np.float32
        assert result["Y"].dtype == np.float32
        assert result["pitch"].dtype == np.float32

    def test_optimize_dataframe_dtypes_categorical(self):
        """Test string to categorical conversion."""
        from opensight.parser import optimize_dataframe_dtypes

        df = pd.DataFrame({
            "weapon": ["ak47", "m4a1", "ak47", "awp", "m4a1"],
            "player_name": ["player1", "player2", "player1", "player3", "player2"],
        })
        result = optimize_dataframe_dtypes(df.copy(), inplace=False)

        assert result["weapon"].dtype.name == "category"
        assert result["player_name"].dtype.name == "category"

    def test_demo_parser_accepts_string_backend(self, tmp_path):
        """Test DemoParser accepts string backend."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        parser = DemoParser(demo_file, backend="auto")
        assert parser.backend == ParserBackend.AUTO

        parser2 = DemoParser(demo_file, backend="demoparser2")
        assert parser2.backend == ParserBackend.DEMOPARSER2

    def test_demo_parser_optimize_dtypes_flag(self, tmp_path):
        """Test DemoParser optimize_dtypes flag."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        parser = DemoParser(demo_file, optimize_dtypes=True)
        assert parser.optimize_dtypes is True

        parser2 = DemoParser(demo_file, optimize_dtypes=False)
        assert parser2.optimize_dtypes is False


# Import ParserBackend for tests
from opensight.parser import ParserBackend

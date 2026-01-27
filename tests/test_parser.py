"""Tests for the parser module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from opensight.core.parser import (
    DamageEvent,
    DemoData,
    DemoParser,
    KillEvent,
    RoundInfo,
    parse_demo,
    safe_bool,
    safe_int,
    safe_str,
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

    def test_round_info_creation(self):
        """RoundInfo can be created with required fields."""
        round_info = RoundInfo(
            round_num=1,
            start_tick=1000,
            end_tick=5000,
            freeze_end_tick=1200,
            winner="CT",
            reason="bomb_defused",
            ct_score=1,
            t_score=0,
            bomb_plant_tick=3000,
            bomb_site="A",
        )
        assert round_info.round_num == 1
        assert round_info.winner == "CT"
        assert round_info.bomb_site == "A"


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
        ticks_df = pd.DataFrame(
            {
                "tick": [1, 2, 3],
                "steamid": [12345, 12345, 12345],
                "X": [100.0, 101.0, 102.0],
                "Y": [200.0, 200.0, 200.0],
                "Z": [0.0, 0.0, 0.0],
            }
        )
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

    def test_parser_raises_on_missing_awpy(self, tmp_path):
        """Parser raises ImportError when awpy not available."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        # Patch the lazy availability check function
        with patch("opensight.core.parser._check_awpy_available", return_value=False):
            parser = DemoParser(demo_file)
            with pytest.raises(ImportError, match="awpy is required"):
                parser.parse()

    def test_parse_caches_result(self, tmp_path):
        """Parser caches the result after first parse."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        # Create mock awpy Demo class
        mock_demo_instance = MagicMock()
        mock_demo_instance.header = {"map_name": "de_dust2"}
        mock_demo_instance.kills = MagicMock()
        mock_demo_instance.kills.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.damages = MagicMock()
        mock_demo_instance.damages.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.rounds = MagicMock()
        mock_demo_instance.rounds.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.grenades = MagicMock()
        mock_demo_instance.grenades.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.bomb = MagicMock()
        mock_demo_instance.bomb.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.shots = MagicMock()
        mock_demo_instance.shots.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.smokes = None
        mock_demo_instance.infernos = None

        mock_demo_class = MagicMock(return_value=mock_demo_instance)
        mock_awpy = MagicMock(Demo=mock_demo_class)

        # Patch awpy import and disable demoparser2 so awpy is used
        with patch("opensight.core.parser.DEMOPARSER2_AVAILABLE", False):
            with patch("opensight.core.parser.AWPY_AVAILABLE", True):
                with patch.dict("sys.modules", {"awpy": mock_awpy}):
                    parser = DemoParser(demo_file)
                    result1 = parser.parse()
                    result2 = parser.parse()

                    # Should be same object (cached)
                    assert result1 is result2
                    # Demo constructor should only be called once
                    assert mock_demo_class.call_count == 1


class TestParseDemoFunction:
    """Tests for parse_demo convenience function."""

    def test_parse_demo_creates_parser_and_parses(self, tmp_path):
        """parse_demo creates parser and calls parse."""
        demo_file = tmp_path / "test.dem"
        demo_file.write_bytes(b"FAKE_DEMO_DATA")

        # Create mock awpy Demo class
        mock_demo_instance = MagicMock()
        mock_demo_instance.header = {"map_name": "de_mirage"}
        mock_demo_instance.kills = MagicMock()
        mock_demo_instance.kills.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.damages = MagicMock()
        mock_demo_instance.damages.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.rounds = MagicMock()
        mock_demo_instance.rounds.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.grenades = MagicMock()
        mock_demo_instance.grenades.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.bomb = MagicMock()
        mock_demo_instance.bomb.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.shots = MagicMock()
        mock_demo_instance.shots.to_pandas.return_value = pd.DataFrame()
        mock_demo_instance.smokes = None
        mock_demo_instance.infernos = None

        mock_demo_class = MagicMock(return_value=mock_demo_instance)
        mock_awpy = MagicMock(Demo=mock_demo_class)

        # Patch awpy import and disable demoparser2 so awpy is used
        with patch("opensight.core.parser.DEMOPARSER2_AVAILABLE", False):
            with patch("opensight.core.parser.AWPY_AVAILABLE", True):
                with patch.dict("sys.modules", {"awpy": mock_awpy}):
                    result = parse_demo(demo_file)
                    assert result.map_name == "de_mirage"


class TestOptimizations:
    """Tests for legacy optimization features (now no-ops with awpy)."""

    def test_parse_mode_enum(self):
        """Test ParseMode enum values (legacy)."""
        from opensight.core.parser import ParseMode

        assert ParseMode.MINIMAL.value == "minimal"
        assert ParseMode.STANDARD.value == "standard"
        assert ParseMode.COMPREHENSIVE.value == "comprehensive"

    def test_parser_backend_enum(self):
        """Test ParserBackend enum values (legacy - only AWPY and AUTO)."""
        from opensight.core.parser import ParserBackend

        assert ParserBackend.AWPY.value == "awpy"
        assert ParserBackend.AUTO.value == "auto"

    def test_optimize_dataframe_dtypes_empty(self):
        """Test dtype optimization on empty DataFrame (legacy no-op)."""
        from opensight.core.parser import optimize_dataframe_dtypes

        df = pd.DataFrame()
        result = optimize_dataframe_dtypes(df)
        assert result.empty

    def test_optimize_dataframe_dtypes_passthrough(self):
        """Test optimize_dataframe_dtypes returns DataFrame unchanged (awpy handles this)."""
        from opensight.core.parser import optimize_dataframe_dtypes

        df = pd.DataFrame(
            {
                "tick": np.array([100, 200, 300], dtype=np.int64),
                "damage": np.array([50, 75, 100], dtype=np.int64),
            }
        )
        original_dtypes = df.dtypes.copy()
        result = optimize_dataframe_dtypes(df.copy(), inplace=False)

        # Should pass through unchanged (awpy handles optimization)
        assert result["tick"].dtype == original_dtypes["tick"]
        assert result["damage"].dtype == original_dtypes["damage"]

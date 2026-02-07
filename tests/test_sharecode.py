"""Tests for the sharecode module."""

import pytest

from opensight.integrations.sharecode import (
    SHARECODE_ALPHABET,
    ShareCodeInfo,
    decode_sharecode,
    encode_sharecode,
    validate_sharecode,
)


class TestSharecodeAlphabet:
    """Tests for the sharecode alphabet."""

    def test_alphabet_length(self):
        """Verify alphabet is base57."""
        assert len(SHARECODE_ALPHABET) == 57

    def test_alphabet_excludes_ambiguous(self):
        """Verify ambiguous characters are excluded."""
        assert "0" not in SHARECODE_ALPHABET
        assert "1" not in SHARECODE_ALPHABET
        assert "I" not in SHARECODE_ALPHABET
        assert "O" not in SHARECODE_ALPHABET
        assert "l" not in SHARECODE_ALPHABET

    def test_alphabet_unique(self):
        """Verify all characters are unique."""
        assert len(set(SHARECODE_ALPHABET)) == len(SHARECODE_ALPHABET)


class TestDecodeSharecode:
    """Tests for share code decoding."""

    def test_decode_returns_sharecode_info(self):
        """Verify decode returns ShareCodeInfo dataclass with correct fields."""
        # Use a valid round-tripped code instead of a placeholder
        encoded = encode_sharecode(12345, 67890, 100)
        result = decode_sharecode(encoded)
        assert isinstance(result, ShareCodeInfo)
        assert hasattr(result, "match_id")
        assert hasattr(result, "outcome_id")
        assert hasattr(result, "token")
        assert hasattr(result, "raw_code")

    def test_decode_empty_code_raises(self):
        """Verify empty code raises ValueError."""
        with pytest.raises(ValueError):
            decode_sharecode("")

    def test_decode_short_code_raises(self):
        """Verify short code raises ValueError."""
        with pytest.raises(ValueError):
            decode_sharecode("CSGO-ABC")

    def test_decode_preserves_raw_code(self):
        """Verify raw code is preserved in result."""
        # Use a valid round-tripped code so decode actually runs
        code = encode_sharecode(99999, 88888, 500)
        result = decode_sharecode(code)
        assert result.raw_code == code


class TestEncodeSharecode:
    """Tests for share code encoding."""

    def test_encode_format(self):
        """Verify encoded code has correct format."""
        result = encode_sharecode(12345, 67890, 100)
        assert result.startswith("CSGO-")
        parts = result.split("-")
        assert len(parts) == 6  # CSGO + 5 segments
        for part in parts[1:]:
            assert len(part) == 5

    def test_encode_uses_valid_alphabet(self):
        """Verify encoded code only uses valid characters."""
        result = encode_sharecode(12345, 67890, 100)
        code_part = result.replace("CSGO-", "").replace("-", "")
        for char in code_part:
            assert char in SHARECODE_ALPHABET


class TestValidateSharecode:
    """Tests for share code validation."""

    def test_validate_empty_returns_false(self):
        """Verify empty code returns False."""
        assert validate_sharecode("") is False

    def test_validate_short_returns_false(self):
        """Verify short code returns False."""
        assert validate_sharecode("CSGO-ABC") is False


class TestRoundTrip:
    """Tests for encode/decode round-trip."""

    def test_roundtrip_preserves_values(self):
        """Verify encoding then decoding preserves original values."""
        match_id = 3524696969420133337
        outcome_id = 3524696969420133338
        token = 32767

        encoded = encode_sharecode(match_id, outcome_id, token)
        decoded = decode_sharecode(encoded)

        assert decoded.match_id == match_id
        assert decoded.outcome_id == outcome_id
        assert decoded.token == token

"""
Share Code Decoder for CS2 Match Replays

Decodes CS2 share codes (e.g., CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx) into
match metadata including match ID, outcome ID, and token.

The share code uses a custom base57 alphabet and encodes 144 bits of data
using big-endian bitwise operations.
"""

import struct
from dataclasses import dataclass

# CS2 share code alphabet (excludes ambiguous characters: 0, 1, I, O, l)
SHARECODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
SHARECODE_BASE = len(SHARECODE_ALPHABET)  # 57

# Reverse lookup table for decoding
ALPHABET_MAP = {char: idx for idx, char in enumerate(SHARECODE_ALPHABET)}


@dataclass
class ShareCodeInfo:
    """Decoded share code metadata."""

    match_id: int
    outcome_id: int
    token: int
    raw_code: str

    def __repr__(self) -> str:
        return (
            f"ShareCodeInfo(match_id={self.match_id}, "
            f"outcome_id={self.outcome_id}, token={self.token})"
        )


def _strip_prefix(code: str) -> str:
    """Remove CSGO- prefix and dashes from share code."""
    code = code.strip().upper()
    if code.startswith("CSGO-"):
        code = code[5:]
    # Handle both upper and original case for the actual code
    return code.replace("-", "")


def _decode_base57(code: str) -> bytes:
    """
    Decode a base57 string into raw bytes.

    The share code encodes 144 bits (18 bytes) of match data.
    """
    # Normalize the code - share codes are case-sensitive in the original alphabet
    normalized = code.replace("-", "")
    if normalized.startswith("CSGO"):
        normalized = normalized[4:]

    # Need to work with original case for proper decoding
    _strip_prefix(code).replace("-", "")

    # For proper decoding, we need the original case
    # Re-extract from original code
    original = code.strip()
    if original.upper().startswith("CSGO-"):
        original = original[5:]
    original = original.replace("-", "")

    # Decode base57 to big integer
    value = 0
    for char in original:
        if char not in ALPHABET_MAP:
            raise ValueError(f"Invalid character in share code: {char}")
        value = value * SHARECODE_BASE + ALPHABET_MAP[char]

    # Convert to 18 bytes (144 bits), big-endian
    result = []
    for _ in range(18):
        result.append(value & 0xFF)
        value >>= 8

    return bytes(reversed(result))


def decode_sharecode(code: str) -> ShareCodeInfo:
    """
    Decode a CS2 share code into match metadata.

    Args:
        code: Share code in format CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx

    Returns:
        ShareCodeInfo containing match_id, outcome_id, and token

    Raises:
        ValueError: If the share code format is invalid

    Example:
        >>> info = decode_sharecode("CSGO-Ab1Cd-EfGhI-jKlMn-OpQrS-tUvWx")
        >>> print(info.match_id)
    """
    if not code or len(code.replace("-", "").replace("CSGO", "")) < 25:
        raise ValueError("Share code appears to be too short")

    try:
        raw_bytes = _decode_base57(code)
    except Exception as e:
        raise ValueError(f"Failed to decode share code: {e}")

    # Extract fields from the decoded bytes
    # Layout: match_id (8 bytes) | outcome_id (8 bytes) | token (2 bytes)
    # Note: Actual layout may vary; this is a common interpretation

    # Using little-endian unpacking based on Valve's typical conventions
    if len(raw_bytes) >= 18:
        match_id = struct.unpack("<Q", raw_bytes[0:8])[0]
        outcome_id = struct.unpack("<Q", raw_bytes[8:16])[0]
        token = struct.unpack("<H", raw_bytes[16:18])[0]
    else:
        raise ValueError("Decoded share code has insufficient bytes")

    return ShareCodeInfo(match_id=match_id, outcome_id=outcome_id, token=token, raw_code=code)


def encode_sharecode(match_id: int, outcome_id: int, token: int) -> str:
    """
    Encode match metadata into a CS2 share code.

    Args:
        match_id: The match ID
        outcome_id: The outcome/reservation ID
        token: The token value

    Returns:
        Share code in format CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx
    """
    # Pack the values into bytes
    raw_bytes = (
        struct.pack("<Q", match_id) + struct.pack("<Q", outcome_id) + struct.pack("<H", token)
    )

    # Convert bytes to big integer
    value = 0
    for byte in raw_bytes:
        value = (value << 8) | byte

    # Encode as base57
    chars = []
    for _ in range(25):  # Share codes are 25 characters
        chars.append(SHARECODE_ALPHABET[value % SHARECODE_BASE])
        value //= SHARECODE_BASE

    # Reverse and format with dashes
    code = "".join(reversed(chars))
    formatted = f"CSGO-{code[0:5]}-{code[5:10]}-{code[10:15]}-{code[15:20]}-{code[20:25]}"

    return formatted


def validate_sharecode(code: str) -> bool:
    """
    Check if a share code appears to be valid.

    Args:
        code: The share code to validate

    Returns:
        True if the code appears valid, False otherwise
    """
    try:
        decode_sharecode(code)
        return True
    except ValueError:
        return False

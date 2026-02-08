"""Tests for the auth module — passwords, JWT, and tier limits."""

from __future__ import annotations

import pytest


class TestPasswordHashing:
    """Test password hashing and verification with bcrypt."""

    @pytest.fixture(autouse=True)
    def _require_bcrypt(self):
        pytest.importorskip("bcrypt", reason="bcrypt not installed")

    def test_hash_password_returns_string(self):
        """hash_password returns a string hash."""
        from opensight.auth.passwords import hash_password

        hashed = hash_password("my_secure_password")
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_hash_password_differs_from_plain(self):
        """Hash must not equal the plaintext."""
        from opensight.auth.passwords import hash_password

        plain = "test_password_123"
        hashed = hash_password(plain)
        assert hashed != plain

    def test_verify_password_correct(self):
        """verify_password returns True for matching password."""
        from opensight.auth.passwords import hash_password, verify_password

        plain = "correct_horse_battery_staple"
        hashed = hash_password(plain)
        assert verify_password(plain, hashed) is True

    def test_verify_password_wrong(self):
        """verify_password returns False for wrong password."""
        from opensight.auth.passwords import hash_password, verify_password

        hashed = hash_password("right_password")
        assert verify_password("wrong_password", hashed) is False

    def test_hash_uniqueness(self):
        """Two hashes of the same password should differ (due to salt)."""
        from opensight.auth.passwords import hash_password

        plain = "same_password"
        hash1 = hash_password(plain)
        hash2 = hash_password(plain)
        assert hash1 != hash2  # different salts


class TestJWT:
    """Test JWT token creation and decoding."""

    @pytest.fixture(autouse=True)
    def _set_jwt_secret(self, monkeypatch):
        """Set JWT_SECRET for the duration of each test."""
        pytest.importorskip("jose", reason="python-jose not installed")
        monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-unit-tests")
        # Force module to re-read the env var by reloading
        import importlib

        import opensight.auth.jwt as jwt_mod

        importlib.reload(jwt_mod)

    def test_create_access_token_returns_string(self):
        """create_access_token returns a JWT string."""
        from opensight.auth.jwt import create_access_token

        token = create_access_token(user_id=42, email="user@example.com")
        assert isinstance(token, str)
        assert len(token) > 0
        # JWT has 3 parts separated by dots
        assert token.count(".") == 2

    def test_decode_token_returns_claims(self):
        """decode_token returns a dict with the encoded claims."""
        from opensight.auth.jwt import create_access_token, decode_token

        token = create_access_token(user_id=42, email="user@example.com")
        claims = decode_token(token)
        assert claims["user_id"] == 42
        assert claims["email"] == "user@example.com"
        assert "exp" in claims  # expiry claim must be present

    def test_decode_invalid_token_raises(self):
        """decode_token raises ValueError for invalid tokens."""
        from opensight.auth.jwt import decode_token

        with pytest.raises(ValueError, match="Invalid token"):
            decode_token("not.a.valid-token")

    def test_decode_tampered_token_raises(self):
        """decode_token raises ValueError for a tampered JWT."""
        from opensight.auth.jwt import create_access_token, decode_token

        token = create_access_token(user_id=1, email="a@b.com")
        # Tamper with the payload section
        parts = token.split(".")
        parts[1] = parts[1] + "TAMPERED"
        tampered = ".".join(parts)
        with pytest.raises(ValueError, match="Invalid token"):
            decode_token(tampered)

    def test_missing_secret_raises_runtime_error(self, monkeypatch):
        """create_access_token raises RuntimeError when JWT_SECRET is empty."""
        monkeypatch.setenv("JWT_SECRET", "")
        import importlib

        import opensight.auth.jwt as jwt_mod

        importlib.reload(jwt_mod)

        from opensight.auth.jwt import create_access_token

        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            create_access_token(user_id=1, email="a@b.com")


class TestTierLimits:
    """Test user tier definitions and access control."""

    def test_tier_enum_values(self):
        """Tier enum has the expected members and ordering."""
        from opensight.auth.tiers import Tier

        assert Tier.FREE == 0
        assert Tier.PRO == 1
        assert Tier.TEAM == 2
        assert Tier.ADMIN == 3
        assert Tier.FREE < Tier.PRO < Tier.TEAM < Tier.ADMIN

    def test_tier_limits_all_defined(self):
        """TIER_LIMITS has entries for free, pro, team, admin."""
        from opensight.auth.tiers import TIER_LIMITS

        assert "free" in TIER_LIMITS
        assert "pro" in TIER_LIMITS
        assert "team" in TIER_LIMITS
        assert "admin" in TIER_LIMITS

    def test_free_tier_has_restrictions(self):
        """Free tier should have demo limits and disabled features."""
        from opensight.auth.tiers import TIER_LIMITS

        free = TIER_LIMITS["free"]
        assert free["demos_per_day"] > 0  # positive limit, not unlimited
        assert free["ai_enabled"] is False
        assert free["scouting_enabled"] is False

    def test_admin_tier_unlimited(self):
        """Admin tier should have unlimited demos and all features."""
        from opensight.auth.tiers import TIER_LIMITS

        admin = TIER_LIMITS["admin"]
        assert admin["demos_per_day"] == -1  # unlimited
        assert admin["ai_enabled"] is True
        assert admin["scouting_enabled"] is True
        assert admin["export_enabled"] is True

    def test_check_tier_same_level(self):
        """check_tier returns True when user tier equals minimum."""
        from opensight.auth.tiers import check_tier

        assert check_tier("free", "free") is True
        assert check_tier("pro", "pro") is True
        assert check_tier("admin", "admin") is True

    def test_check_tier_higher_level(self):
        """check_tier returns True when user tier exceeds minimum."""
        from opensight.auth.tiers import check_tier

        assert check_tier("admin", "free") is True
        assert check_tier("team", "pro") is True
        assert check_tier("pro", "free") is True

    def test_check_tier_lower_level(self):
        """check_tier returns False when user tier is below minimum."""
        from opensight.auth.tiers import check_tier

        assert check_tier("free", "pro") is False
        assert check_tier("free", "admin") is False
        assert check_tier("pro", "team") is False

    def test_check_tier_unknown_user(self):
        """check_tier treats unknown user tier as free (0)."""
        from opensight.auth.tiers import check_tier

        assert check_tier("unknown", "free") is True  # 0 >= 0
        assert check_tier("unknown", "pro") is False  # 0 < 1

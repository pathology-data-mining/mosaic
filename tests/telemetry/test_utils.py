"""Tests for telemetry utility functions."""

import base64
import json
import time

import pytest

from mosaic.telemetry.utils import (
    StageTimer,
    sanitize_error_message,
    hash_session_id,
    UserInfo,
    extract_user_info,
)


class TestStageTimer:
    """Tests for StageTimer context manager."""

    def test_basic_timing(self):
        """Test basic timing functionality."""
        timings = {}
        with StageTimer("test_stage", timings):
            time.sleep(0.1)

        assert "test_stage_duration_sec" in timings
        assert timings["test_stage_duration_sec"] >= 0.1

    def test_multiple_stages(self):
        """Test timing multiple stages."""
        timings = {}

        with StageTimer("stage_a", timings):
            time.sleep(0.05)

        with StageTimer("stage_b", timings):
            time.sleep(0.05)

        assert "stage_a_duration_sec" in timings
        assert "stage_b_duration_sec" in timings
        assert timings["stage_a_duration_sec"] >= 0.05
        assert timings["stage_b_duration_sec"] >= 0.05

    def test_timing_with_exception(self):
        """Test that timing is recorded even when exception occurs."""
        timings = {}

        with pytest.raises(ValueError):
            with StageTimer("failing_stage", timings):
                time.sleep(0.05)
                raise ValueError("Test error")

        # Timing should still be recorded
        assert "failing_stage_duration_sec" in timings
        assert timings["failing_stage_duration_sec"] >= 0.05


class TestSanitizeErrorMessage:
    """Tests for error message sanitization."""

    def test_sanitize_unix_paths(self):
        """Test sanitization of Unix-style paths."""
        message = "File not found: /home/user/data/slide.svs"
        sanitized = sanitize_error_message(message)
        assert "/home/user" not in sanitized
        assert "[PATH]" in sanitized

    def test_sanitize_windows_paths(self):
        """Test sanitization of Windows-style paths."""
        message = "File not found: C:\\Users\\John\\Documents\\slide.svs"
        sanitized = sanitize_error_message(message)
        assert "C:\\Users" not in sanitized
        assert "[PATH]" in sanitized

    def test_sanitize_ip_addresses(self):
        """Test sanitization of IP addresses."""
        message = "Connection refused to 192.168.1.100:8080"
        sanitized = sanitize_error_message(message)
        assert "192.168.1.100" not in sanitized
        assert "[IP]" in sanitized

    def test_sanitize_email_addresses(self):
        """Test sanitization of email addresses."""
        message = "Invalid user: john.doe@example.com"
        sanitized = sanitize_error_message(message)
        assert "john.doe@example.com" not in sanitized
        assert "[EMAIL]" in sanitized

    def test_sanitize_urls(self):
        """Test sanitization of URLs."""
        message = "Failed to fetch https://api.example.com/data"
        sanitized = sanitize_error_message(message)
        assert "https://api.example.com" not in sanitized
        assert "[URL]" in sanitized

    def test_sanitize_multiple_patterns(self):
        """Test sanitization of multiple patterns in one message."""
        message = (
            "Error at /home/user/app: "
            "Could not connect to 10.0.0.1 for user@domain.com"
        )
        sanitized = sanitize_error_message(message)
        assert "/home/user" not in sanitized
        assert "10.0.0.1" not in sanitized
        assert "user@domain.com" not in sanitized

    def test_sanitize_empty_message(self):
        """Test handling of empty message."""
        assert sanitize_error_message("") == ""
        assert sanitize_error_message(None) is None

    def test_sanitize_preserves_error_context(self):
        """Test that error context is preserved."""
        message = "ValueError: Invalid configuration"
        sanitized = sanitize_error_message(message)
        assert "ValueError" in sanitized
        assert "Invalid configuration" in sanitized


class TestHashSessionId:
    """Tests for session ID hashing."""

    def test_hash_session_id(self):
        """Test basic session ID hashing."""
        hashed = hash_session_id("test-session-123")
        assert hashed is not None
        assert hashed != "test-session-123"
        assert len(hashed) == 16  # Truncated to 16 chars

    def test_hash_none_returns_none(self):
        """Test that None input returns None."""
        assert hash_session_id(None) is None

    def test_hash_is_deterministic(self):
        """Test that same input produces same hash."""
        hash1 = hash_session_id("session-abc")
        hash2 = hash_session_id("session-abc")
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        """Test that different inputs produce different hashes."""
        hash1 = hash_session_id("session-1")
        hash2 = hash_session_id("session-2")
        assert hash1 != hash2

    def test_hash_is_consistent_across_calls(self):
        """Test hash consistency for privacy linking."""
        # Same session should always produce same hash
        session_id = "user-session-12345"
        hashes = [hash_session_id(session_id) for _ in range(10)]
        assert len(set(hashes)) == 1  # All hashes should be identical


class TestUserInfo:
    """Tests for UserInfo dataclass."""

    def test_default_values(self):
        """Test default UserInfo values."""
        user_info = UserInfo()
        assert user_info.is_logged_in is False
        assert user_info.username is None

    def test_custom_values(self):
        """Test UserInfo with custom values."""
        user_info = UserInfo(is_logged_in=True, username="testuser")
        assert user_info.is_logged_in is True
        assert user_info.username == "testuser"


class TestExtractUserInfo:
    """Tests for extract_user_info function."""

    def _create_jwt_token(self, username: str) -> str:
        """Helper to create a valid JWT token for testing."""
        # JWT format: header.payload.signature
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {"onBehalfOf": {"user": username}}

        # Encode header and payload
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode()
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()

        # Remove padding (JWT standard)
        header_b64 = header_b64.rstrip("=")
        payload_b64 = payload_b64.rstrip("=")

        # Create fake signature
        signature = "fake_signature"

        return f"{header_b64}.{payload_b64}.{signature}"

    def _create_mock_request(self, referer: str = None):
        """Helper to create a mock request object."""

        class MockHeaders:
            def __init__(self, referer):
                self._referer = referer

            def get(self, key, default=""):
                if key == "referer":
                    return self._referer if self._referer else default
                return default

        class MockRequest:
            def __init__(self, referer):
                self.headers = MockHeaders(referer)

        return MockRequest(referer)

    def test_extract_user_info_with_valid_jwt(self):
        """Test extraction with valid JWT token."""
        # Create JWT token
        token = self._create_jwt_token("testuser123")
        referer = f"https://huggingface.co/spaces/test/app?__sign={token}&other=param"

        request = self._create_mock_request(referer)
        user_info = extract_user_info(request, is_hf_spaces=True)

        assert user_info.is_logged_in is True
        assert user_info.username == "testuser123"

    def test_extract_user_info_anonymous_no_token(self):
        """Test extraction when no JWT token is present (anonymous)."""
        referer = "https://huggingface.co/spaces/test/app"
        request = self._create_mock_request(referer)

        user_info = extract_user_info(request, is_hf_spaces=True)

        assert user_info.is_logged_in is False
        assert user_info.username is None

    def test_extract_user_info_malformed_jwt(self):
        """Test extraction with malformed JWT token."""
        referer = "https://huggingface.co/spaces/test/app?__sign=invalid.jwt.token"
        request = self._create_mock_request(referer)

        user_info = extract_user_info(request, is_hf_spaces=True)

        # Should return default UserInfo (no exception)
        assert user_info.is_logged_in is False
        assert user_info.username is None

    def test_extract_user_info_not_hf_spaces(self):
        """Test extraction when not on HF Spaces."""
        token = self._create_jwt_token("testuser")
        referer = f"http://localhost:7860?__sign={token}"
        request = self._create_mock_request(referer)

        # Even with valid token, should return default if is_hf_spaces=False
        user_info = extract_user_info(request, is_hf_spaces=False)

        assert user_info.is_logged_in is False
        assert user_info.username is None

    def test_extract_user_info_none_request(self):
        """Test extraction with None request."""
        user_info = extract_user_info(None, is_hf_spaces=True)

        assert user_info.is_logged_in is False
        assert user_info.username is None

    def test_extract_user_info_no_headers(self):
        """Test extraction when request has no headers attribute."""

        class RequestWithoutHeaders:
            pass

        request = RequestWithoutHeaders()
        user_info = extract_user_info(request, is_hf_spaces=True)

        assert user_info.is_logged_in is False
        assert user_info.username is None

    def test_extract_user_info_jwt_missing_user(self):
        """Test extraction when JWT is valid but missing user field."""
        # Create JWT with different structure
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {"someOtherField": "value"}  # No onBehalfOf

        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode()
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        header_b64 = header_b64.rstrip("=")
        payload_b64 = payload_b64.rstrip("=")

        token = f"{header_b64}.{payload_b64}.fake_sig"
        referer = f"https://huggingface.co/spaces/test/app?__sign={token}"

        request = self._create_mock_request(referer)
        user_info = extract_user_info(request, is_hf_spaces=True)

        assert user_info.is_logged_in is False
        assert user_info.username is None

    def test_extract_user_info_jwt_with_special_characters(self):
        """Test extraction with username containing special characters."""
        token = self._create_jwt_token("user-name_123")
        referer = f"https://huggingface.co/spaces/test/app?__sign={token}"

        request = self._create_mock_request(referer)
        user_info = extract_user_info(request, is_hf_spaces=True)

        assert user_info.is_logged_in is True
        assert user_info.username == "user-name_123"

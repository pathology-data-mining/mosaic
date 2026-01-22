"""Tests for TelemetryConfig."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mosaic.telemetry.config import (
    TelemetryConfig,
    _detect_hf_spaces,
    _get_telemetry_directory,
)


class TestHFSpacesDetection:
    """Tests for HuggingFace Spaces detection."""

    def test_detect_hf_spaces_with_space_id(self):
        """Test detection when SPACE_ID is set."""
        with patch.dict(os.environ, {"SPACE_ID": "test-space"}):
            assert _detect_hf_spaces() is True

    def test_detect_hf_spaces_without_space_id(self):
        """Test detection when SPACE_ID is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert _detect_hf_spaces() is False


class TestTelemetryDirectory:
    """Tests for telemetry directory resolution."""

    def test_env_override(self):
        """Test MOSAIC_TELEMETRY_DIR environment override."""
        with patch.dict(os.environ, {"MOSAIC_TELEMETRY_DIR": "/custom/path"}):
            result = _get_telemetry_directory()
            assert result == Path("/custom/path")

    def test_fallback_to_temp(self):
        """Test fallback to temp directory."""
        with patch.dict(os.environ, {}, clear=True):
            result = _get_telemetry_directory()
            assert "mosaic_telemetry" in str(result)
            assert tempfile.gettempdir() in str(result)


class TestTelemetryConfig:
    """Tests for TelemetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = TelemetryConfig()
            assert config.enabled is True
            assert config.hf_repo is None
            assert config.idle_timeout_min == 30
            assert config.heartbeat_interval_sec == 300

    def test_enabled_from_env(self):
        """Test enabling/disabling via environment."""
        with patch.dict(os.environ, {"MOSAIC_TELEMETRY_ENABLED": "false"}):
            config = TelemetryConfig()
            assert config.enabled is False

        with patch.dict(os.environ, {"MOSAIC_TELEMETRY_ENABLED": "true"}):
            config = TelemetryConfig()
            assert config.enabled is True

    def test_hourly_rate_from_env(self):
        """Test hourly rate from environment."""
        with patch.dict(os.environ, {"MOSAIC_HOURLY_RATE": "1.50"}):
            config = TelemetryConfig()
            assert config.hourly_rate == 1.50

    def test_hourly_rate_default_hf_spaces(self):
        """Test default hourly rate on HF Spaces."""
        with patch.dict(os.environ, {"SPACE_ID": "test-space"}, clear=True):
            config = TelemetryConfig()
            assert config.hourly_rate == 0.40
            assert config.is_hf_spaces is True

    def test_hourly_rate_default_non_hf(self):
        """Test default hourly rate outside HF Spaces."""
        with patch.dict(os.environ, {}, clear=True):
            config = TelemetryConfig()
            assert config.hourly_rate == 0.0
            assert config.is_hf_spaces is False

    def test_hf_repo_from_env(self):
        """Test HF repo from environment."""
        with patch.dict(os.environ, {"MOSAIC_TELEMETRY_HF_REPO": "org/repo"}):
            config = TelemetryConfig()
            assert config.hf_repo == "org/repo"

    def test_idle_timeout_from_env(self):
        """Test idle timeout from environment."""
        with patch.dict(os.environ, {"MOSAIC_IDLE_TIMEOUT_MIN": "60"}):
            config = TelemetryConfig()
            assert config.idle_timeout_min == 60

    def test_custom_base_dir(self):
        """Test custom base directory."""
        custom_path = Path("/custom/telemetry")
        config = TelemetryConfig(base_dir=custom_path)
        assert config.base_dir == custom_path

    def test_base_dir_string_conversion(self):
        """Test that string base_dir is converted to Path."""
        config = TelemetryConfig(base_dir="/custom/path")
        assert isinstance(config.base_dir, Path)
        assert config.base_dir == Path("/custom/path")

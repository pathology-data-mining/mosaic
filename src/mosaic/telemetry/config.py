"""Configuration for Mosaic telemetry.

This module provides configuration dataclass for telemetry settings,
including storage location detection and environment variable handling.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _detect_hf_spaces() -> bool:
    """Detect if running on HuggingFace Spaces."""
    return bool(os.environ.get("SPACE_ID"))


def _get_default_hourly_rate() -> float:
    """Get default hourly rate based on environment.

    Returns $0.40/hr for HF Spaces (T4 small pricing), $0 otherwise.
    """
    if env_rate := os.environ.get("MOSAIC_HOURLY_RATE"):
        try:
            return float(env_rate)
        except ValueError:
            pass
    return 0.40 if _detect_hf_spaces() else 0.0


def _get_telemetry_directory() -> Path:
    """Determine telemetry storage directory.

    Priority:
    1. MOSAIC_TELEMETRY_DIR environment variable
    2. HF Spaces persistent storage (/data/mosaic_telemetry)
    3. System temp directory fallback
    """
    # 1. Environment variable override
    if env_dir := os.environ.get("MOSAIC_TELEMETRY_DIR"):
        return Path(env_dir)

    # 2. HF Spaces persistent storage
    if os.environ.get("SPACE_ID"):
        persistent = Path("/data")
        if persistent.exists() and os.access(persistent, os.W_OK):
            return persistent / "mosaic_telemetry"

    # 3. Fallback to temp directory
    return Path(tempfile.gettempdir()) / "mosaic_telemetry"


@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection and storage.

    Attributes:
        enabled: Whether telemetry is enabled (default: True, from MOSAIC_TELEMETRY_ENABLED)
        base_dir: Base directory for telemetry storage
        hf_repo: HuggingFace Dataset repo for uploads (from MOSAIC_TELEMETRY_HF_REPO)
        hourly_rate: Cost rate in USD/hour (from MOSAIC_HOURLY_RATE)
        idle_timeout_min: HF Spaces idle timeout in minutes (from MOSAIC_IDLE_TIMEOUT_MIN)
        is_hf_spaces: Whether running on HuggingFace Spaces
        heartbeat_interval_sec: Seconds between heartbeat events
    """

    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "MOSAIC_TELEMETRY_ENABLED", "true"
        ).lower()
        != "false"
    )
    base_dir: Path = field(default_factory=_get_telemetry_directory)
    hf_repo: Optional[str] = field(
        default_factory=lambda: os.environ.get("MOSAIC_TELEMETRY_HF_REPO")
    )
    hourly_rate: float = field(default_factory=_get_default_hourly_rate)
    idle_timeout_min: int = field(
        default_factory=lambda: int(os.environ.get("MOSAIC_IDLE_TIMEOUT_MIN", "30"))
    )
    is_hf_spaces: bool = field(default_factory=_detect_hf_spaces)
    heartbeat_interval_sec: int = 300  # 5 minutes

    def __post_init__(self):
        """Ensure base_dir is a Path object."""
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)

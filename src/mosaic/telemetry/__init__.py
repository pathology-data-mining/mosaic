"""Mosaic Telemetry Module.

This module provides lightweight telemetry for the Mosaic HuggingFace app to track:
- Usage: Analysis requests, slide counts, settings used
- Cost: App awake time and active analysis time (cost only on HF Spaces)
- Failures: Errors with sanitized messages and context

Key features:
- Gradio-only: No telemetry for CLI batch processing
- No external dependencies: Uses stdlib only (json, dataclasses)
- Privacy-first: Session IDs hashed, no file paths stored. HF usernames recorded for usage tracking.
- File-based storage: JSONL format with daily rotation
- HF Spaces compatible: Uses /data persistent storage when available

Usage:
    from mosaic.telemetry import TelemetryTracker

    tracker = TelemetryTracker.get_instance()
    tracker.log_app_start()

    # On analysis start
    tracker.log_usage_event(
        event_type="analysis_start",
        analysis_id=analysis_id,
        slide_count=len(slides),
        ...
    )

Environment Variables:
    MOSAIC_TELEMETRY_ENABLED: Enable/disable telemetry (default: true)
    MOSAIC_TELEMETRY_DIR: Override storage directory
    MOSAIC_TELEMETRY_HF_REPO: HF Dataset repo for uploads
    MOSAIC_HOURLY_RATE: Hourly cost rate (default: $0.40 on HF Spaces, $0 otherwise)
    MOSAIC_IDLE_TIMEOUT_MIN: HF Spaces idle timeout in minutes (default: 30)
"""

from mosaic.telemetry.config import TelemetryConfig
from mosaic.telemetry.events import (
    AppSessionEvent,
    UsageEvent,
    ResourceEvent,
    FailureEvent,
)
from mosaic.telemetry.storage import TelemetryStorage
from mosaic.telemetry.tracker import TelemetryTracker
from mosaic.telemetry.utils import (
    StageTimer,
    sanitize_error_message,
    hash_session_id,
    hash_username,
    UserInfo,
    extract_user_info,
)

__all__ = [
    # Main interface
    "TelemetryTracker",
    # Configuration
    "TelemetryConfig",
    # Events
    "AppSessionEvent",
    "UsageEvent",
    "ResourceEvent",
    "FailureEvent",
    # Storage
    "TelemetryStorage",
    # Utilities
    "StageTimer",
    "sanitize_error_message",
    "hash_session_id",
    "hash_username",
    "UserInfo",
    "extract_user_info",
]

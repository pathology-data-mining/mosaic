"""Tests for telemetry event dataclasses."""

import json
from datetime import datetime

from mosaic.telemetry.events import (
    AppSessionEvent,
    UsageEvent,
    ResourceEvent,
    FailureEvent,
)


class TestAppSessionEvent:
    """Tests for AppSessionEvent."""

    def test_app_start_event(self):
        """Test creating app_start event."""
        event = AppSessionEvent(
            event_type="app_start",
            app_start_time=datetime.utcnow().isoformat() + "Z",
            uptime_sec=0.0,
            analysis_count=0,
            analysis_time_sec=0.0,
            idle_time_sec=0.0,
            hourly_rate=0.40,
            idle_timeout_min=30,
            is_hf_spaces=True,
            estimated_cost_usd=0.0,
        )

        assert event.event_type == "app_start"
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = AppSessionEvent(
            event_type="heartbeat",
            app_start_time="2026-01-21T10:00:00Z",
            uptime_sec=3600.0,
            analysis_count=5,
            analysis_time_sec=1800.0,
            idle_time_sec=1800.0,
            hourly_rate=0.40,
            idle_timeout_min=30,
            is_hf_spaces=True,
            estimated_cost_usd=0.40,
        )

        data = event.to_dict()
        assert data["event_type"] == "heartbeat"
        assert data["uptime_sec"] == 3600.0
        assert data["estimated_cost_usd"] == 0.40

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert "heartbeat" in json_str


class TestUsageEvent:
    """Tests for UsageEvent."""

    def test_analysis_start_event(self):
        """Test creating analysis_start event."""
        event = UsageEvent(
            event_type="analysis_start",
            analysis_id="test-123",
            session_hash="abc123",
            slide_count=5,
            site_type="Primary",
            cancer_subtype="LUAD",
            seg_config="Biopsy",
            gpu_type="T4",
        )

        assert event.event_type == "analysis_start"
        assert event.slide_count == 5
        assert event.duration_sec is None  # Not set for start events

    def test_analysis_complete_event(self):
        """Test creating analysis_complete event."""
        event = UsageEvent(
            event_type="analysis_complete",
            analysis_id="test-123",
            session_hash="abc123",
            slide_count=5,
            duration_sec=120.5,
            success=True,
            gpu_type="T4",
        )

        assert event.event_type == "analysis_complete"
        assert event.duration_sec == 120.5
        assert event.success is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = UsageEvent(
            event_type="analysis_start",
            analysis_id="test-123",
            session_hash=None,
            slide_count=3,
        )

        data = event.to_dict()
        assert data["event_type"] == "analysis_start"
        assert data["slide_count"] == 3
        assert data["session_hash"] is None

    def test_user_info_fields(self):
        """Test is_logged_in and hf_username fields in UsageEvent."""
        event = UsageEvent(
            event_type="analysis_start",
            analysis_id="test-123",
            session_hash=None,
            slide_count=1,
            is_logged_in=True,
            hf_username="testuser",
        )

        assert event.is_logged_in is True
        assert event.hf_username == "testuser"

        data = event.to_dict()
        assert data["is_logged_in"] is True
        assert data["hf_username"] == "testuser"

    def test_user_info_fields_default_none(self):
        """Test that user info fields default to None in to_dict()."""
        event = UsageEvent(
            event_type="analysis_start",
            analysis_id="test-123",
            session_hash=None,
            slide_count=1,
        )

        data = event.to_dict()
        assert data["is_logged_in"] is None
        assert data["hf_username"] is None


class TestResourceEvent:
    """Tests for ResourceEvent."""

    def test_resource_event(self):
        """Test creating resource event."""
        event = ResourceEvent(
            analysis_id="test-123",
            session_hash="abc123",
            total_duration_sec=180.5,
            segmentation_duration_sec=10.0,
            ctranspath_duration_sec=30.0,
            optimus_duration_sec=50.0,
            aeon_duration_sec=20.0,
            paladin_duration_sec=40.0,
            tile_count=1000,
            filtered_tile_count=800,
            batch_size=64,
            gpu_type="T4",
            peak_gpu_memory_gb=12.5,
        )

        assert event.total_duration_sec == 180.5
        assert event.tile_count == 1000
        assert event.peak_gpu_memory_gb == 12.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = ResourceEvent(
            analysis_id="test-123",
            session_hash=None,
            total_duration_sec=100.0,
            tile_count=500,
        )

        data = event.to_dict()
        assert data["total_duration_sec"] == 100.0
        assert data["tile_count"] == 500

    def test_user_info_fields(self):
        """Test is_logged_in and hf_username fields in ResourceEvent."""
        event = ResourceEvent(
            analysis_id="test-123",
            session_hash=None,
            total_duration_sec=100.0,
            is_logged_in=True,
            hf_username="testuser",
        )

        data = event.to_dict()
        assert data["is_logged_in"] is True
        assert data["hf_username"] == "testuser"


class TestFailureEvent:
    """Tests for FailureEvent."""

    def test_failure_event(self):
        """Test creating failure event."""
        event = FailureEvent(
            error_type="ValueError",
            error_message="Invalid slide format",
            error_stage="upload",
            analysis_id="test-123",
            slide_count=1,
            gpu_type="T4",
        )

        assert event.error_type == "ValueError"
        assert event.error_stage == "upload"
        assert event.event_id is not None

    def test_failure_event_minimal(self):
        """Test creating failure event with minimal fields."""
        event = FailureEvent(
            error_type="RuntimeError",
            error_message="Unknown error",
            error_stage="inference",
        )

        assert event.error_type == "RuntimeError"
        assert event.analysis_id is None
        assert event.slide_count is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = FailureEvent(
            error_type="MemoryError",
            error_message="Out of GPU memory",
            error_stage="ctranspath",
        )

        data = event.to_dict()
        assert data["error_type"] == "MemoryError"
        assert data["error_stage"] == "ctranspath"

    def test_user_info_fields(self):
        """Test is_logged_in and hf_username fields in FailureEvent."""
        event = FailureEvent(
            error_type="ValueError",
            error_message="test error",
            error_stage="upload",
            is_logged_in=False,
            hf_username=None,
        )

        data = event.to_dict()
        assert data["is_logged_in"] is False
        assert data["hf_username"] is None


class TestEventIdGeneration:
    """Tests for event ID generation."""

    def test_unique_event_ids(self):
        """Test that event IDs are unique."""
        events = [
            UsageEvent(
                event_type="analysis_start",
                analysis_id="test",
                session_hash=None,
                slide_count=1,
            )
            for _ in range(100)
        ]

        event_ids = [e.event_id for e in events]
        assert len(set(event_ids)) == 100  # All unique


class TestTimestampGeneration:
    """Tests for timestamp generation."""

    def test_timestamp_format(self):
        """Test timestamp format is ISO 8601 with Z suffix."""
        event = UsageEvent(
            event_type="analysis_start",
            analysis_id="test",
            session_hash=None,
            slide_count=1,
        )

        assert event.timestamp.endswith("Z")
        # Should be parseable
        datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))

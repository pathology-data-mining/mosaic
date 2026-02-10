"""Tests for TelemetryTracker."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from mosaic.telemetry import TelemetryTracker, TelemetryConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for telemetry storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tracker(temp_dir):
    """Create a fresh tracker instance for each test."""
    TelemetryTracker.reset_instance()
    config = TelemetryConfig(
        enabled=True,
        base_dir=temp_dir,
        hourly_rate=0.40,
        idle_timeout_min=30,
        is_hf_spaces=False,
    )
    tracker = TelemetryTracker.get_instance(config)
    yield tracker
    TelemetryTracker.reset_instance()


class TestTelemetryTrackerSingleton:
    """Tests for singleton behavior."""

    def test_get_instance_returns_same_instance(self, temp_dir):
        """Test that get_instance always returns the same instance."""
        TelemetryTracker.reset_instance()
        config = TelemetryConfig(base_dir=temp_dir)
        tracker1 = TelemetryTracker.get_instance(config)
        tracker2 = TelemetryTracker.get_instance()

        assert tracker1 is tracker2
        TelemetryTracker.reset_instance()

    def test_reset_instance_creates_new_instance(self, temp_dir):
        """Test that reset_instance allows creating a new instance."""
        TelemetryTracker.reset_instance()
        config = TelemetryConfig(base_dir=temp_dir)
        tracker1 = TelemetryTracker.get_instance(config)

        TelemetryTracker.reset_instance()
        tracker2 = TelemetryTracker.get_instance(config)

        assert tracker1 is not tracker2
        TelemetryTracker.reset_instance()


class TestAppSessionEvents:
    """Tests for app session event logging."""

    def test_log_app_start(self, tracker, temp_dir):
        """Test logging app start event."""
        tracker.log_app_start()

        session_files = list((temp_dir / "daily").glob("session_*.jsonl"))
        assert len(session_files) == 1

        with open(session_files[0]) as f:
            event = json.loads(f.read().strip())

        assert event["event_type"] == "app_start"
        assert event["uptime_sec"] == 0.0
        assert event["analysis_count"] == 0
        assert event["estimated_cost_usd"] == 0.0

    def test_log_heartbeat(self, tracker, temp_dir):
        """Test logging heartbeat event."""
        tracker.log_app_start()
        time.sleep(0.1)  # Small delay to accumulate uptime
        tracker.log_heartbeat()

        session_files = list((temp_dir / "daily").glob("session_*.jsonl"))
        assert len(session_files) == 1

        with open(session_files[0]) as f:
            lines = f.readlines()

        assert len(lines) == 2
        heartbeat = json.loads(lines[1])
        assert heartbeat["event_type"] == "heartbeat"
        assert heartbeat["uptime_sec"] > 0

    def test_log_app_shutdown(self, tracker, temp_dir):
        """Test logging app shutdown event."""
        tracker.log_app_start()
        tracker.log_app_shutdown()

        session_files = list((temp_dir / "daily").glob("session_*.jsonl"))
        with open(session_files[0]) as f:
            lines = f.readlines()

        shutdown = json.loads(lines[-1])
        assert shutdown["event_type"] == "app_shutdown"


class TestUsageEvents:
    """Tests for usage event logging."""

    def test_log_analysis_start(self, tracker, temp_dir):
        """Test logging analysis start event."""
        tracker.log_usage_event(
            event_type="analysis_start",
            analysis_id="test-123",
            slide_count=5,
            session_hash="abc123",
            site_type="Primary",
            cancer_subtype="LUAD",
            seg_config="Biopsy",
            gpu_type="T4",
        )

        usage_files = list((temp_dir / "daily").glob("usage_*.jsonl"))
        assert len(usage_files) == 1

        with open(usage_files[0]) as f:
            event = json.loads(f.read().strip())

        assert event["event_type"] == "analysis_start"
        assert event["analysis_id"] == "test-123"
        assert event["slide_count"] == 5
        assert event["site_type"] == "Primary"
        # Session hash should be hashed
        assert event["session_hash"] is not None
        assert event["session_hash"] != "abc123"

    def test_log_usage_event_with_user_info(self, tracker, temp_dir):
        """Test that is_logged_in and hf_username are persisted in usage events."""
        tracker.log_usage_event(
            event_type="analysis_start",
            analysis_id="test-user-info",
            slide_count=1,
            is_logged_in=True,
            hf_username="testuser",
        )

        usage_files = list((temp_dir / "daily").glob("usage_*.jsonl"))
        with open(usage_files[0]) as f:
            event = json.loads(f.read().strip())

        assert event["is_logged_in"] is True
        assert event["hf_username"] == "testuser"

    def test_log_analysis_complete(self, tracker, temp_dir):
        """Test logging analysis complete event."""
        tracker.log_app_start()
        tracker.log_usage_event(
            event_type="analysis_complete",
            analysis_id="test-123",
            slide_count=5,
            duration_sec=120.5,
            success=True,
            gpu_type="T4",
        )

        usage_files = list((temp_dir / "daily").glob("usage_*.jsonl"))
        with open(usage_files[0]) as f:
            event = json.loads(f.read().strip())

        assert event["event_type"] == "analysis_complete"
        assert event["duration_sec"] == 120.5
        assert event["success"] is True

    def test_analysis_complete_updates_session_metrics(self, tracker):
        """Test that analysis complete updates session metrics."""
        tracker.log_app_start()
        tracker.log_usage_event(
            event_type="analysis_complete",
            analysis_id="test-1",
            slide_count=3,
            duration_sec=60.0,
            success=True,
        )
        tracker.log_usage_event(
            event_type="analysis_complete",
            analysis_id="test-2",
            slide_count=2,
            duration_sec=45.0,
            success=True,
        )

        metrics = tracker._get_session_metrics()
        assert metrics["analysis_count"] == 2
        assert metrics["analysis_time_sec"] == 105.0

    def test_log_usage_event_with_cached_slides(self, tracker, temp_dir):
        """Test that cached_slide_count is persisted in usage events."""
        tracker.log_usage_event(
            event_type="analysis_complete",
            analysis_id="test-cache",
            slide_count=10,
            duration_sec=50.0,
            success=True,
            cached_slide_count=3,
        )

        usage_files = list((temp_dir / "daily").glob("usage_*.jsonl"))
        with open(usage_files[0]) as f:
            event = json.loads(f.read().strip())

        assert event["cached_slide_count"] == 3

    def test_fully_cached_analysis_excludes_duration_from_metrics(self, tracker):
        """Test that fully cached analyses don't count toward analysis_time_sec."""
        tracker.log_app_start()
        tracker.log_usage_event(
            event_type="analysis_complete",
            analysis_id="test-cache",
            slide_count=5,
            duration_sec=2.0,
            success=True,
            cached_slide_count=5,  # All slides cached
        )

        metrics = tracker._get_session_metrics()
        assert metrics["analysis_count"] == 1
        assert metrics["analysis_time_sec"] == 0.0

    def test_mixed_cache_analysis_includes_duration_in_metrics(self, tracker):
        """Test that mixed analyses (some cached) count toward analysis_time_sec."""
        tracker.log_app_start()
        tracker.log_usage_event(
            event_type="analysis_complete",
            analysis_id="test-mixed",
            slide_count=10,
            duration_sec=60.0,
            success=True,
            cached_slide_count=3,  # Only 3/10 cached
        )

        metrics = tracker._get_session_metrics()
        assert metrics["analysis_count"] == 1
        assert metrics["analysis_time_sec"] == 60.0


class TestResourceEvents:
    """Tests for resource event logging."""

    def test_log_resource_event(self, tracker, temp_dir):
        """Test logging resource event."""
        tracker.log_resource_event(
            analysis_id="test-123",
            session_hash="abc123",
            total_duration_sec=180.5,
            tile_count=1000,
            filtered_tile_count=800,
            gpu_type="T4",
            peak_gpu_memory_gb=12.5,
        )

        resource_files = list((temp_dir / "daily").glob("resource_*.jsonl"))
        assert len(resource_files) == 1

        with open(resource_files[0]) as f:
            event = json.loads(f.read().strip())

        assert event["analysis_id"] == "test-123"
        assert event["total_duration_sec"] == 180.5
        assert event["tile_count"] == 1000
        assert event["peak_gpu_memory_gb"] == 12.5

    def test_log_resource_event_with_user_info(self, tracker, temp_dir):
        """Test that is_logged_in and hf_username are persisted in resource events."""
        tracker.log_resource_event(
            analysis_id="test-user-info",
            total_duration_sec=60.0,
            is_logged_in=True,
            hf_username="testuser",
        )

        resource_files = list((temp_dir / "daily").glob("resource_*.jsonl"))
        with open(resource_files[0]) as f:
            event = json.loads(f.read().strip())

        assert event["is_logged_in"] is True
        assert event["hf_username"] == "testuser"


class TestFailureEvents:
    """Tests for failure event logging."""

    def test_log_failure_event(self, tracker, temp_dir):
        """Test logging failure event."""
        tracker.log_failure_event(
            error_type="ValueError",
            error_message="Invalid slide format",
            error_stage="upload",
            analysis_id="test-123",
            slide_count=1,
            gpu_type="T4",
        )

        failure_files = list((temp_dir / "daily").glob("failure_*.jsonl"))
        assert len(failure_files) == 1

        with open(failure_files[0]) as f:
            event = json.loads(f.read().strip())

        assert event["error_type"] == "ValueError"
        assert event["error_stage"] == "upload"
        assert event["analysis_id"] == "test-123"

    def test_log_failure_event_with_user_info(self, tracker, temp_dir):
        """Test that is_logged_in and hf_username are persisted in failure events."""
        tracker.log_failure_event(
            error_type="ValueError",
            error_message="test error",
            error_stage="upload",
            is_logged_in=False,
            hf_username=None,
        )

        failure_files = list((temp_dir / "daily").glob("failure_*.jsonl"))
        with open(failure_files[0]) as f:
            event = json.loads(f.read().strip())

        assert event["is_logged_in"] is False
        assert event["hf_username"] is None

    def test_error_message_sanitized(self, tracker, temp_dir):
        """Test that error messages are sanitized."""
        tracker.log_failure_event(
            error_type="FileNotFoundError",
            error_message="File not found: /home/user/secret/data.svs",
            error_stage="upload",
        )

        failure_files = list((temp_dir / "daily").glob("failure_*.jsonl"))
        with open(failure_files[0]) as f:
            event = json.loads(f.read().strip())

        # Path should be sanitized
        assert "/home/user" not in event["error_message"]
        assert "[PATH]" in event["error_message"]


class TestDisabledTelemetry:
    """Tests for disabled telemetry."""

    def test_disabled_telemetry_no_events(self, temp_dir):
        """Test that disabled telemetry doesn't write events."""
        TelemetryTracker.reset_instance()
        config = TelemetryConfig(
            enabled=False,
            base_dir=temp_dir,
        )
        tracker = TelemetryTracker.get_instance(config)

        tracker.log_app_start()
        tracker.log_usage_event(
            event_type="analysis_start",
            analysis_id="test",
            slide_count=1,
        )

        # No files should be created
        all_files = list((temp_dir / "daily").glob("*.jsonl"))
        assert len(all_files) == 0
        TelemetryTracker.reset_instance()


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_cost_calculation(self, tracker):
        """Test cost calculation formula."""
        # Test with 1 hour of uptime at $0.40/hr
        cost = tracker._calculate_cost(3600)  # 1 hour in seconds
        assert cost == 0.40

    def test_cost_in_session_events(self, tracker, temp_dir):
        """Test that cost is included in session events."""
        tracker.log_app_start()
        time.sleep(0.1)
        tracker.log_app_shutdown()

        session_files = list((temp_dir / "daily").glob("session_*.jsonl"))
        with open(session_files[0]) as f:
            lines = f.readlines()

        shutdown = json.loads(lines[-1])
        assert "estimated_cost_usd" in shutdown
        assert shutdown["estimated_cost_usd"] >= 0

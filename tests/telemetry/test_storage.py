"""Tests for TelemetryStorage."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from mosaic.telemetry import TelemetryStorage
from mosaic.telemetry.events import UsageEvent, FailureEvent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for telemetry storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def storage(temp_dir):
    """Create a storage instance for testing."""
    return TelemetryStorage(temp_dir)


class TestStorageInitialization:
    """Tests for storage initialization."""

    def test_creates_directories(self, temp_dir):
        """Test that storage creates necessary directories."""
        storage = TelemetryStorage(temp_dir)
        assert (temp_dir / "daily").exists()

    def test_handles_existing_directories(self, temp_dir):
        """Test that storage handles existing directories."""
        (temp_dir / "daily").mkdir(parents=True)
        storage = TelemetryStorage(temp_dir)
        assert (temp_dir / "daily").exists()


class TestEventWriting:
    """Tests for writing events."""

    def test_write_event(self, storage, temp_dir):
        """Test basic event writing."""
        event_data = {
            "event_id": "test-123",
            "event_type": "test",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        result = storage.write_event("test", event_data)

        assert result is True

        files = list((temp_dir / "daily").glob("test_*.jsonl"))
        assert len(files) == 1

        with open(files[0]) as f:
            written = json.loads(f.read().strip())
        assert written["event_id"] == "test-123"

    def test_write_multiple_events(self, storage, temp_dir):
        """Test writing multiple events to same file."""
        for i in range(3):
            storage.write_event("test", {"event_id": f"test-{i}"})

        files = list((temp_dir / "daily").glob("test_*.jsonl"))
        assert len(files) == 1

        with open(files[0]) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_write_usage_event(self, storage, temp_dir):
        """Test writing usage event."""
        event = UsageEvent(
            event_type="analysis_start",
            analysis_id="test-123",
            session_hash="abc",
            slide_count=5,
        )
        result = storage.write_usage_event(event)
        assert result is True

        files = list((temp_dir / "daily").glob("usage_*.jsonl"))
        assert len(files) == 1

    def test_write_failure_event(self, storage, temp_dir):
        """Test writing failure event."""
        event = FailureEvent(
            error_type="ValueError",
            error_message="Test error",
            error_stage="test",
        )
        result = storage.write_failure_event(event)
        assert result is True

        files = list((temp_dir / "daily").glob("failure_*.jsonl"))
        assert len(files) == 1


class TestDailyRotation:
    """Tests for daily file rotation."""

    def test_daily_file_naming(self, storage, temp_dir):
        """Test that files are named with today's date."""
        storage.write_event("test", {"event_id": "test-1"})

        files = list((temp_dir / "daily").glob("test_*.jsonl"))
        assert len(files) == 1

        today = datetime.utcnow().strftime("%Y-%m-%d")
        assert f"test_{today}.jsonl" in files[0].name


class TestGetAllFiles:
    """Tests for getting all telemetry files."""

    def test_get_all_files_empty(self, storage):
        """Test getting files when none exist."""
        files = storage.get_all_files()
        assert len(files) == 0

    def test_get_all_files(self, storage):
        """Test getting all files after writing events."""
        storage.write_event("usage", {"event_id": "1"})
        storage.write_event("failure", {"event_id": "2"})

        files = storage.get_all_files()
        assert len(files) == 2


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_writes(self, storage, temp_dir):
        """Test that concurrent writes don't corrupt data."""
        import threading

        def write_events(event_type, count):
            for i in range(count):
                storage.write_event(event_type, {"event_id": f"{event_type}-{i}"})

        threads = [
            threading.Thread(target=write_events, args=("usage", 10)),
            threading.Thread(target=write_events, args=("usage", 10)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        files = list((temp_dir / "daily").glob("usage_*.jsonl"))
        assert len(files) == 1

        with open(files[0]) as f:
            lines = f.readlines()

        # All 20 events should be written
        assert len(lines) == 20

        # Each line should be valid JSON
        for line in lines:
            json.loads(line)

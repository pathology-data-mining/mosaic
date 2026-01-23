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


class TestHuggingFaceDownload:
    """Tests for downloading from HuggingFace Dataset repositories."""

    def test_download_returns_false_when_hf_not_installed(self, storage, monkeypatch):
        """Test that download returns False when huggingface_hub is not available."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("No module named 'huggingface_hub'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        result = storage.download_from_hf_dataset("test-org/test-repo")
        assert result is False

    def test_download_returns_false_when_repo_list_fails(self, storage, monkeypatch):
        """Test that download returns False when listing repo files fails."""
        mock_api = type(
            "MockHfApi",
            (),
            {
                "list_repo_files": lambda self, repo_id, repo_type: (
                    _ for _ in ()
                ).throw(Exception("API Error"))
            },
        )()

        def mock_hfapi():
            return mock_api

        monkeypatch.setattr("huggingface_hub.HfApi", mock_hfapi)

        result = storage.download_from_hf_dataset("test-org/test-repo")
        assert result is False

    def test_download_returns_false_when_no_files(self, storage, monkeypatch):
        """Test that download returns False when no JSONL files exist."""
        mock_api = type(
            "MockHfApi",
            (),
            {
                "list_repo_files": lambda self, repo_id, repo_type: [
                    "README.md",
                    "other_file.txt",
                ]
            },
        )()

        def mock_hfapi():
            return mock_api

        monkeypatch.setattr("huggingface_hub.HfApi", mock_hfapi)

        result = storage.download_from_hf_dataset("test-org/test-repo")
        assert result is False

    def test_download_creates_new_file(self, storage, temp_dir, monkeypatch):
        """Test that download creates new local file when it doesn't exist."""
        # Create a temp file to simulate the downloaded content
        downloaded_content = '{"event_id": "remote-1", "event_type": "test"}\n{"event_id": "remote-2", "event_type": "test"}\n'
        downloaded_file = temp_dir / "downloaded_test.jsonl"
        downloaded_file.write_text(downloaded_content)

        mock_api = type(
            "MockHfApi",
            (),
            {
                "list_repo_files": lambda self, repo_id, repo_type: [
                    "daily/test_2026-01-20.jsonl"
                ]
            },
        )()

        def mock_hfapi():
            return mock_api

        def mock_download(repo_id, filename, repo_type):
            return str(downloaded_file)

        monkeypatch.setattr("huggingface_hub.HfApi", mock_hfapi)
        monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_download)

        result = storage.download_from_hf_dataset("test-org/test-repo")

        assert result is True
        target_file = temp_dir / "daily" / "test_2026-01-20.jsonl"
        assert target_file.exists()

        with open(target_file) as f:
            content = f.read()
        assert "remote-1" in content
        assert "remote-2" in content

    def test_download_merges_with_existing_file(self, storage, temp_dir, monkeypatch):
        """Test that download merges new content with existing local file."""
        # Create existing local file
        existing_content = '{"event_id": "local-1", "event_type": "test"}\n'
        daily_dir = temp_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        local_file = daily_dir / "test_2026-01-20.jsonl"
        local_file.write_text(existing_content)

        # Create remote content with one duplicate and one new
        remote_content = '{"event_id": "local-1", "event_type": "test"}\n{"event_id": "remote-1", "event_type": "test"}\n'
        downloaded_file = temp_dir / "downloaded_test.jsonl"
        downloaded_file.write_text(remote_content)

        mock_api = type(
            "MockHfApi",
            (),
            {
                "list_repo_files": lambda self, repo_id, repo_type: [
                    "daily/test_2026-01-20.jsonl"
                ]
            },
        )()

        def mock_hfapi():
            return mock_api

        def mock_download(repo_id, filename, repo_type):
            return str(downloaded_file)

        monkeypatch.setattr("huggingface_hub.HfApi", mock_hfapi)
        monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_download)

        result = storage.download_from_hf_dataset("test-org/test-repo")

        assert result is True

        with open(local_file) as f:
            lines = f.readlines()

        # Should have 2 lines: original local-1 and new remote-1 (no duplicate)
        assert len(lines) == 2
        event_ids = [json.loads(line)["event_id"] for line in lines]
        assert "local-1" in event_ids
        assert "remote-1" in event_ids

    def test_download_handles_multiple_files(self, storage, temp_dir, monkeypatch):
        """Test that download handles multiple remote files."""
        # Create remote content files
        usage_content = '{"event_id": "usage-1"}\n'
        failure_content = '{"event_id": "failure-1"}\n'

        usage_file = temp_dir / "usage_download.jsonl"
        failure_file = temp_dir / "failure_download.jsonl"
        usage_file.write_text(usage_content)
        failure_file.write_text(failure_content)

        mock_api = type(
            "MockHfApi",
            (),
            {
                "list_repo_files": lambda self, repo_id, repo_type: [
                    "daily/usage_2026-01-20.jsonl",
                    "daily/failure_2026-01-20.jsonl",
                ]
            },
        )()

        def mock_hfapi():
            return mock_api

        def mock_download(repo_id, filename, repo_type):
            if "usage" in filename:
                return str(usage_file)
            return str(failure_file)

        monkeypatch.setattr("huggingface_hub.HfApi", mock_hfapi)
        monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_download)

        result = storage.download_from_hf_dataset("test-org/test-repo")

        assert result is True

        daily_dir = temp_dir / "daily"
        usage_target = daily_dir / "usage_2026-01-20.jsonl"
        failure_target = daily_dir / "failure_2026-01-20.jsonl"

        assert usage_target.exists()
        assert failure_target.exists()

"""Unit tests for user_storage module."""

import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from mosaic.user_storage import (
    DEFAULT_QUOTA_BYTES,
    SlideMetadata,
    StorageInfo,
    cleanup_old_files,
    delete_user_slide,
    get_slide_path,
    get_storage_usage,
    get_user_storage_dir,
    list_user_slides,
    save_uploaded_slide,
)


@pytest.fixture
def temp_storage_base(tmp_path, monkeypatch):
    """Temporarily override storage base to tmp_path."""
    monkeypatch.setattr("mosaic.user_storage.USER_STORAGE_BASE", tmp_path)
    return tmp_path


@pytest.fixture
def test_user():
    """Test username."""
    return "test_user"


@pytest.fixture
def test_slide_file(tmp_path):
    """Create a temporary test slide file."""
    slide_path = tmp_path / "test_slide.svs"
    # Create a 1KB test file
    slide_path.write_bytes(b"x" * 1024)
    return slide_path


def test_get_user_storage_dir(temp_storage_base, test_user):
    """Test creating user storage directory."""
    user_dir = get_user_storage_dir(test_user)

    assert user_dir.exists()
    assert user_dir == temp_storage_base / test_user
    assert (user_dir / "slides").exists()
    assert (user_dir / "results").exists()


def test_save_uploaded_slide(temp_storage_base, test_user, test_slide_file):
    """Test saving an uploaded slide."""
    slide_id, saved_path = save_uploaded_slide(test_user, test_slide_file)

    # Check slide ID is a valid UUID
    assert uuid.UUID(slide_id)

    # Check file was copied
    saved_path = Path(saved_path)
    assert saved_path.exists()
    assert saved_path.stat().st_size == 1024

    # Check metadata was created
    user_dir = get_user_storage_dir(test_user)
    metadata_path = user_dir / "slides" / "metadata.json"
    assert metadata_path.exists()

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert slide_id in metadata
    assert metadata[slide_id]["original_filename"] == "test_slide.svs"
    assert metadata[slide_id]["size_bytes"] == 1024


def test_save_uploaded_slide_quota_exceeded(
    temp_storage_base, test_user, test_slide_file
):
    """Test quota enforcement when saving slides."""
    # Set quota to 512 bytes (smaller than test file)
    with pytest.raises(ValueError, match="Storage quota exceeded"):
        save_uploaded_slide(test_user, test_slide_file, quota_bytes=512)


def test_list_user_slides(temp_storage_base, test_user, test_slide_file):
    """Test listing user slides."""
    # Save two slides
    slide_id1, _ = save_uploaded_slide(test_user, test_slide_file)
    slide_id2, _ = save_uploaded_slide(test_user, test_slide_file)

    slides = list_user_slides(test_user)

    assert len(slides) == 2
    # Should be sorted by upload time, newest first
    assert slides[0].slide_id == slide_id2
    assert slides[1].slide_id == slide_id1
    assert all(isinstance(s, SlideMetadata) for s in slides)


def test_get_slide_path(temp_storage_base, test_user, test_slide_file):
    """Test getting slide file path."""
    slide_id, saved_path = save_uploaded_slide(test_user, test_slide_file)

    retrieved_path = get_slide_path(test_user, slide_id)
    assert retrieved_path == Path(saved_path)
    assert retrieved_path.exists()

    # Test non-existent slide
    assert get_slide_path(test_user, "nonexistent") is None


def test_delete_user_slide(temp_storage_base, test_user, test_slide_file):
    """Test deleting a slide."""
    slide_id, saved_path = save_uploaded_slide(test_user, test_slide_file)

    # Verify file exists
    assert Path(saved_path).exists()

    # Delete slide
    assert delete_user_slide(test_user, slide_id) is True

    # Verify file is gone
    assert not Path(saved_path).exists()

    # Verify metadata is updated
    slides = list_user_slides(test_user)
    assert len(slides) == 0

    # Test deleting non-existent slide
    assert delete_user_slide(test_user, "nonexistent") is False


def test_get_storage_usage(temp_storage_base, test_user, test_slide_file):
    """Test calculating storage usage."""
    # Initially empty
    usage = get_storage_usage(test_user)
    assert usage.total_bytes == 0
    assert usage.file_count == 0
    assert usage.quota_bytes == DEFAULT_QUOTA_BYTES

    # Save a slide
    save_uploaded_slide(test_user, test_slide_file)

    # Check updated usage
    usage = get_storage_usage(test_user)
    assert usage.total_bytes == 1024
    assert usage.file_count == 1


def test_cleanup_old_files(temp_storage_base, test_user, tmp_path):
    """Test FIFO cleanup of old files."""
    # Create three test files of different sizes
    slide1 = tmp_path / "slide1.svs"
    slide2 = tmp_path / "slide2.svs"
    slide3 = tmp_path / "slide3.svs"

    slide1.write_bytes(b"x" * 1024)  # 1KB
    slide2.write_bytes(b"x" * 2048)  # 2KB
    slide3.write_bytes(b"x" * 3072)  # 3KB

    # Save slides (oldest to newest)
    id1, _ = save_uploaded_slide(test_user, slide1)
    id2, _ = save_uploaded_slide(test_user, slide2)
    id3, _ = save_uploaded_slide(test_user, slide3)

    # Total usage: 6KB (1KB + 2KB + 3KB)
    usage = get_storage_usage(test_user)
    assert usage.total_bytes == 6144

    # Cleanup to 4KB target
    # Should delete oldest files: slide1 (1KB) + slide2 (2KB) = 3KB deleted
    # Remaining: slide3 (3KB) which is ≤ 4KB target
    deleted = cleanup_old_files(test_user, target_bytes=4096)
    assert deleted == 2

    # Verify slide1 and slide2 are gone, slide3 remains
    assert get_slide_path(test_user, id1) is None
    assert get_slide_path(test_user, id2) is None
    assert get_slide_path(test_user, id3) is not None

    # Check final usage
    usage = get_storage_usage(test_user)
    assert usage.total_bytes == 3072  # 3KB only


def test_cleanup_old_files_no_action_needed(
    temp_storage_base, test_user, test_slide_file
):
    """Test cleanup when already under target."""
    save_uploaded_slide(test_user, test_slide_file)

    # Target is higher than current usage
    deleted = cleanup_old_files(test_user, target_bytes=10240)
    assert deleted == 0

    # Slide should still exist
    slides = list_user_slides(test_user)
    assert len(slides) == 1


def test_storage_info_persistence(temp_storage_base, test_user, test_slide_file):
    """Test that storage info persists across function calls."""
    save_uploaded_slide(test_user, test_slide_file)

    # Get usage first time
    usage1 = get_storage_usage(test_user)

    # Get usage again (should load from saved file)
    usage2 = get_storage_usage(test_user)

    assert usage1.total_bytes == usage2.total_bytes
    assert usage1.file_count == usage2.file_count


def test_slide_metadata_dataclass():
    """Test SlideMetadata dataclass."""
    metadata = SlideMetadata(
        slide_id="test-id",
        original_filename="test.svs",
        upload_time="2024-01-01T00:00:00",
        size_bytes=1024,
        file_extension=".svs",
    )

    assert metadata.slide_id == "test-id"
    assert metadata.original_filename == "test.svs"
    assert metadata.size_bytes == 1024


def test_storage_info_dataclass():
    """Test StorageInfo dataclass."""
    info = StorageInfo(
        total_bytes=1024, file_count=5, quota_bytes=5368709120, last_cleanup=None
    )

    assert info.total_bytes == 1024
    assert info.file_count == 5
    assert info.quota_bytes == 5368709120
    assert info.last_cleanup is None

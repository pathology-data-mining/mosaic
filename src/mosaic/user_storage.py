"""User storage management for HuggingFace Spaces.

This module provides ephemeral storage for uploaded slide files, enforcing
per-user quotas and providing cleanup utilities. Storage is cleared on
instance restart (no persistence).

Storage structure:
    /tmp/mosaic_user_data/{username}/
        slides/
            {slide_id}.{ext}    # Uploaded slide files
            metadata.json       # Slide metadata
        storage_info.json       # Usage tracking
"""

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

# Default per-user storage quota (5GB)
DEFAULT_QUOTA_GB = int(os.environ.get("MOSAIC_USER_STORAGE_QUOTA_GB", "5"))
DEFAULT_QUOTA_BYTES = DEFAULT_QUOTA_GB * 1024 * 1024 * 1024

# Base directory for user storage (ephemeral)
USER_STORAGE_BASE = Path("/tmp/mosaic_user_data")


@dataclass
class SlideMetadata:
    """Metadata for an uploaded slide.

    Attributes:
        slide_id: Unique identifier for the slide
        original_filename: Original filename when uploaded
        upload_time: ISO timestamp of upload
        size_bytes: File size in bytes
        file_extension: File extension (e.g., '.svs', '.tif')
    """

    slide_id: str
    original_filename: str
    upload_time: str
    size_bytes: int
    file_extension: str


@dataclass
class StorageInfo:
    """Storage usage information for a user.

    Attributes:
        total_bytes: Total bytes used by all files
        file_count: Number of slide files stored
        quota_bytes: Storage quota in bytes
        last_cleanup: ISO timestamp of last cleanup (or None)
    """

    total_bytes: int
    file_count: int
    quota_bytes: int
    last_cleanup: Optional[str] = None


def get_user_storage_dir(username: str) -> Path:
    """Get the storage directory for a user.

    Creates the directory structure if it doesn't exist:
        /tmp/mosaic_user_data/{username}/
            slides/
            results/

    Args:
        username: Username from HF Spaces OAuth

    Returns:
        Path to user's storage directory
    """
    user_dir = USER_STORAGE_BASE / username
    slides_dir = user_dir / "slides"
    results_dir = user_dir / "results"

    slides_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return user_dir


def _load_slide_metadata(username: str) -> dict[str, SlideMetadata]:
    """Load slide metadata from disk.

    Args:
        username: Username

    Returns:
        Dict mapping slide_id to SlideMetadata
    """
    user_dir = get_user_storage_dir(username)
    metadata_path = user_dir / "slides" / "metadata.json"

    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return {k: SlideMetadata(**v) for k, v in data.items()}
    except Exception as e:
        logger.error(f"Failed to load slide metadata for {username}: {e}")
        return {}


def _save_slide_metadata(username: str, metadata: dict[str, SlideMetadata]) -> None:
    """Save slide metadata to disk.

    Args:
        username: Username
        metadata: Dict mapping slide_id to SlideMetadata
    """
    user_dir = get_user_storage_dir(username)
    metadata_path = user_dir / "slides" / "metadata.json"

    try:
        data = {k: asdict(v) for k, v in metadata.items()}
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save slide metadata for {username}: {e}")


def _load_storage_info(username: str) -> StorageInfo:
    """Load storage info from disk.

    Args:
        username: Username

    Returns:
        StorageInfo object
    """
    user_dir = get_user_storage_dir(username)
    info_path = user_dir / "storage_info.json"

    if not info_path.exists():
        return StorageInfo(total_bytes=0, file_count=0, quota_bytes=DEFAULT_QUOTA_BYTES)

    try:
        with open(info_path, "r") as f:
            data = json.load(f)
        return StorageInfo(**data)
    except Exception as e:
        logger.error(f"Failed to load storage info for {username}: {e}")
        return StorageInfo(total_bytes=0, file_count=0, quota_bytes=DEFAULT_QUOTA_BYTES)


def _save_storage_info(username: str, info: StorageInfo) -> None:
    """Save storage info to disk.

    Args:
        username: Username
        info: StorageInfo object
    """
    user_dir = get_user_storage_dir(username)
    info_path = user_dir / "storage_info.json"

    try:
        with open(info_path, "w") as f:
            json.dump(asdict(info), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save storage info for {username}: {e}")


def get_storage_usage(username: str) -> StorageInfo:
    """Calculate current storage usage for a user.

    Recalculates from actual files to ensure accuracy.

    Args:
        username: Username

    Returns:
        StorageInfo with current usage
    """
    user_dir = get_user_storage_dir(username)
    slides_dir = user_dir / "slides"

    total_bytes = 0
    file_count = 0

    # Calculate from actual files
    for slide_file in slides_dir.glob("*"):
        if slide_file.is_file() and slide_file.name != "metadata.json":
            total_bytes += slide_file.stat().st_size
            file_count += 1

    info = _load_storage_info(username)
    info.total_bytes = total_bytes
    info.file_count = file_count

    _save_storage_info(username, info)
    return info


def save_uploaded_slide(
    username: str, slide_path: str | Path, quota_bytes: Optional[int] = None
) -> tuple[str, str]:
    """Save an uploaded slide to user storage.

    Enforces per-user quota. If quota would be exceeded, raises ValueError.

    Args:
        username: Username
        slide_path: Path to the uploaded slide file
        quota_bytes: Optional override for quota (uses default if None)

    Returns:
        Tuple of (slide_id, saved_path)

    Raises:
        ValueError: If quota would be exceeded
        OSError: If file operations fail
    """
    slide_path = Path(slide_path)
    file_size = slide_path.stat().st_size
    file_extension = slide_path.suffix

    # Check quota
    quota = quota_bytes or DEFAULT_QUOTA_BYTES
    current_usage = get_storage_usage(username)

    if current_usage.total_bytes + file_size > quota:
        quota_gb = quota / (1024**3)
        current_gb = current_usage.total_bytes / (1024**3)
        raise ValueError(
            f"Storage quota exceeded. Used: {current_gb:.2f} GB / {quota_gb:.2f} GB. "
            f"Delete old files before uploading."
        )

    # Generate unique slide ID
    slide_id = str(uuid.uuid4())
    user_dir = get_user_storage_dir(username)
    dest_path = user_dir / "slides" / f"{slide_id}{file_extension}"

    # Copy file
    shutil.copy2(slide_path, dest_path)
    logger.info(f"Saved slide {slide_id} for user {username} ({file_size} bytes)")

    # Update metadata
    metadata = _load_slide_metadata(username)
    metadata[slide_id] = SlideMetadata(
        slide_id=slide_id,
        original_filename=slide_path.name,
        upload_time=datetime.utcnow().isoformat(),
        size_bytes=file_size,
        file_extension=file_extension,
    )
    _save_slide_metadata(username, metadata)

    # Update storage info
    current_usage.total_bytes += file_size
    current_usage.file_count += 1
    _save_storage_info(username, current_usage)

    return slide_id, str(dest_path)


def list_user_slides(username: str) -> list[SlideMetadata]:
    """List all uploaded slides for a user.

    Returns slides sorted by upload time (newest first).

    Args:
        username: Username

    Returns:
        List of SlideMetadata objects
    """
    metadata = _load_slide_metadata(username)
    slides = list(metadata.values())
    # Sort by upload time, newest first
    slides.sort(key=lambda s: s.upload_time, reverse=True)
    return slides


def get_slide_path(username: str, slide_id: str) -> Optional[Path]:
    """Get the file path for a slide.

    Args:
        username: Username
        slide_id: Slide ID

    Returns:
        Path to slide file, or None if not found
    """
    metadata = _load_slide_metadata(username)
    if slide_id not in metadata:
        return None

    user_dir = get_user_storage_dir(username)
    slide_meta = metadata[slide_id]
    slide_path = user_dir / "slides" / f"{slide_id}{slide_meta.file_extension}"

    if slide_path.exists():
        return slide_path
    return None


def delete_user_slide(username: str, slide_id: str) -> bool:
    """Delete a slide and all associated results.

    Args:
        username: Username
        slide_id: Slide ID to delete

    Returns:
        True if deleted successfully, False if not found
    """
    metadata = _load_slide_metadata(username)
    if slide_id not in metadata:
        logger.warning(f"Slide {slide_id} not found for user {username}")
        return False

    user_dir = get_user_storage_dir(username)
    slide_meta = metadata[slide_id]
    slide_path = user_dir / "slides" / f"{slide_id}{slide_meta.file_extension}"

    # Delete the file
    try:
        if slide_path.exists():
            slide_path.unlink()
            logger.info(f"Deleted slide {slide_id} for user {username}")
    except Exception as e:
        logger.error(f"Failed to delete slide file {slide_id}: {e}")
        return False

    # Delete associated results (import here to avoid circular dependency)
    try:
        from mosaic.user_results import delete_results_for_slide

        delete_results_for_slide(username, slide_id)
    except Exception as e:
        logger.warning(f"Failed to delete results for slide {slide_id}: {e}")

    # Update metadata
    del metadata[slide_id]
    _save_slide_metadata(username, metadata)

    # Update storage info
    current_usage = get_storage_usage(username)
    _save_storage_info(username, current_usage)

    return True


def find_slide_by_name(username: str, name: str) -> Optional[str]:
    """Find a slide UUID by original filename or filename stem.

    Searches for an exact filename match first, then tries stem match,
    then falls back to treating the input as a UUID.

    Args:
        username: Username
        name: Original filename, filename stem, or slide UUID

    Returns:
        slide_id (UUID) if found, None otherwise
    """
    metadata = _load_slide_metadata(username)

    # Try exact filename match
    for slide_id, meta in metadata.items():
        if meta.original_filename == name:
            return slide_id

    # Try stem match
    for slide_id, meta in metadata.items():
        if Path(meta.original_filename).stem == name:
            return slide_id

    # Try as UUID (backward compat)
    if name in metadata:
        return name

    return None


def cleanup_old_files(username: str, target_bytes: int) -> int:
    """Remove oldest files until storage is under target.

    Uses FIFO strategy (oldest files deleted first).

    Args:
        username: Username
        target_bytes: Target storage size in bytes

    Returns:
        Number of files deleted
    """
    current_usage = get_storage_usage(username)
    if current_usage.total_bytes <= target_bytes:
        return 0

    slides = list_user_slides(username)
    # Sort by upload time, oldest first
    slides.sort(key=lambda s: s.upload_time)

    deleted_count = 0
    current_bytes = current_usage.total_bytes

    for slide in slides:
        if current_bytes <= target_bytes:
            break

        delete_user_slide(username, slide.slide_id)
        current_bytes -= slide.size_bytes
        deleted_count += 1

    # Update last cleanup time
    info = _load_storage_info(username)
    info.last_cleanup = datetime.utcnow().isoformat()
    _save_storage_info(username, info)

    logger.info(
        f"Cleanup for user {username}: deleted {deleted_count} files, "
        f"now using {current_bytes / (1024**3):.2f} GB"
    )

    return deleted_count


def check_disk_space() -> tuple[int, int, float]:
    """Check global disk space on the instance.

    Returns:
        Tuple of (used_bytes, total_bytes, percent_used)
    """
    try:
        stat = shutil.disk_usage("/tmp")
        percent_used = (stat.used / stat.total) * 100
        return stat.used, stat.total, percent_used
    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")
        return 0, 0, 0.0

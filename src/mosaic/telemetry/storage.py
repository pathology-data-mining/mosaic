"""Thread-safe JSONL storage for Mosaic telemetry.

This module provides storage backend for telemetry events:
- Daily JSONL file rotation
- Thread-safe writes
- HuggingFace Dataset upload support
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from mosaic.telemetry.events import (
        AppSessionEvent,
        UsageEvent,
        ResourceEvent,
        FailureEvent,
    )


class TelemetryStorage:
    """Thread-safe JSONL storage backend for telemetry events.

    Writes events to daily-rotated JSONL files organized by event type.

    Directory structure:
        base_dir/
            daily/
                session_2026-01-21.jsonl
                usage_2026-01-21.jsonl
                resource_2026-01-21.jsonl
                failure_2026-01-21.jsonl
    """

    def __init__(self, base_dir: Path):
        """Initialize storage.

        Args:
            base_dir: Base directory for telemetry storage
        """
        self.base_dir = Path(base_dir)
        self.daily_dir = self.base_dir / "daily"
        self._lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        try:
            self.daily_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Failed to create telemetry directory: {e}")

    def _get_daily_file(self, event_type: str) -> Path:
        """Get path to today's JSONL file for given event type.

        Args:
            event_type: Type of event ("session", "usage", "resource", "failure")

        Returns:
            Path to the daily JSONL file
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self.daily_dir / f"{event_type}_{today}.jsonl"

    def write_event(self, event_type: str, event_data: dict) -> bool:
        """Write an event to the appropriate daily file.

        Thread-safe write operation.

        Args:
            event_type: Type of event ("session", "usage", "resource", "failure")
            event_data: Event data dictionary

        Returns:
            True if write succeeded, False otherwise
        """
        try:
            file_path = self._get_daily_file(event_type)
            line = json.dumps(event_data, default=str) + "\n"

            with self._lock:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(line)
            return True
        except Exception as e:
            logger.warning(f"Failed to write telemetry event: {e}")
            return False

    def write_session_event(self, event: "AppSessionEvent") -> bool:
        """Write an app session event."""
        return self.write_event("session", event.to_dict())

    def write_usage_event(self, event: "UsageEvent") -> bool:
        """Write a usage event."""
        return self.write_event("usage", event.to_dict())

    def write_resource_event(self, event: "ResourceEvent") -> bool:
        """Write a resource event."""
        return self.write_event("resource", event.to_dict())

    def write_failure_event(self, event: "FailureEvent") -> bool:
        """Write a failure event."""
        return self.write_event("failure", event.to_dict())

    def upload_to_hf_dataset(self, repo_id: str) -> bool:
        """Upload accumulated telemetry files to HF Dataset.

        Args:
            repo_id: HuggingFace Dataset repository ID (e.g., "PDM-Group/mosaic-telemetry")

        Returns:
            True if at least one file was uploaded, False otherwise
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.warning("huggingface_hub not installed, skipping upload")
            return False

        api = HfApi()
        uploaded = 0

        for jsonl_file in self.daily_dir.glob("*.jsonl"):
            try:
                # Check if file is non-empty
                if jsonl_file.stat().st_size == 0:
                    continue

                api.upload_file(
                    path_or_fileobj=str(jsonl_file),
                    path_in_repo=f"daily/{jsonl_file.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                uploaded += 1
                logger.info(f"Uploaded telemetry file: {jsonl_file.name}")
            except Exception as e:
                logger.warning(f"Failed to upload {jsonl_file.name}: {e}")

        return uploaded > 0

    def get_all_files(self) -> list:
        """Get list of all telemetry files.

        Returns:
            List of Path objects for all JSONL files
        """
        if not self.daily_dir.exists():
            return []
        return list(self.daily_dir.glob("*.jsonl"))

    def download_from_hf_dataset(self, repo_id: str) -> bool:
        """Download telemetry files from HF Dataset repository.

        Downloads all JSONL files from the daily/ folder in the repository
        to the local daily directory, merging with any existing local files.

        Args:
            repo_id: HuggingFace Dataset repository ID (e.g., "PDM-Group/mosaic-telemetry")

        Returns:
            True if at least one file was downloaded, False otherwise
        """
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            logger.warning("huggingface_hub not installed, skipping download")
            return False

        api = HfApi()
        downloaded = 0

        try:
            # List files in the repository
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        except Exception as e:
            logger.error(f"Failed to list files in {repo_id}: {e}")
            return False

        # Filter for daily/*.jsonl files
        jsonl_files = [
            f for f in files if f.startswith("daily/") and f.endswith(".jsonl")
        ]

        if not jsonl_files:
            logger.info(f"No telemetry files found in {repo_id}")
            return False

        self._ensure_directories()

        for remote_path in jsonl_files:
            try:
                # Download file to a temp location
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                )

                # Get the filename (e.g., "session_2026-01-21.jsonl")
                filename = os.path.basename(remote_path)
                target_path = self.daily_dir / filename

                # Read downloaded content
                with open(local_path, "r", encoding="utf-8") as f:
                    remote_content = f.read()

                # If local file exists, merge (append new lines only)
                if target_path.exists():
                    with open(target_path, "r", encoding="utf-8") as f:
                        local_content = f.read()

                    # Parse existing lines to avoid duplicates
                    local_lines = (
                        set(local_content.strip().split("\n"))
                        if local_content.strip()
                        else set()
                    )
                    remote_lines = (
                        remote_content.strip().split("\n")
                        if remote_content.strip()
                        else []
                    )

                    # Append only new lines
                    new_lines = [
                        line
                        for line in remote_lines
                        if line and line not in local_lines
                    ]
                    if new_lines:
                        with self._lock:
                            with open(target_path, "a", encoding="utf-8") as f:
                                for line in new_lines:
                                    f.write(line + "\n")
                        logger.info(
                            f"Merged {len(new_lines)} new events into {filename}"
                        )
                else:
                    # No local file, just copy
                    with self._lock:
                        with open(target_path, "w", encoding="utf-8") as f:
                            f.write(remote_content)
                    logger.info(f"Downloaded telemetry file: {filename}")

                downloaded += 1
            except Exception as e:
                logger.warning(f"Failed to download {remote_path}: {e}")

        return downloaded > 0

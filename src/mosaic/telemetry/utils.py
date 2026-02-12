"""Utility functions for Mosaic telemetry.

This module provides helper utilities:
- StageTimer: Context manager for timing pipeline stages
- sanitize_error_message: Remove sensitive data from error messages
- hash_session_id: Hash session IDs for privacy
- UserInfo: Dataclass for user information from HF Spaces
- extract_user_info: Extract user info from Gradio request object
"""

import hashlib
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

from loguru import logger


@contextmanager
def StageTimer(stage_name: str, timings: Dict[str, float]):
    """Context manager for timing pipeline stages.

    Args:
        stage_name: Name of the stage (e.g., "segmentation", "ctranspath")
        timings: Dictionary to store timing results

    Usage:
        stage_timings = {}
        with StageTimer("segmentation", stage_timings):
            # ... segmentation code ...

        print(stage_timings["segmentation_duration_sec"])
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        timings[f"{stage_name}_duration_sec"] = duration


def sanitize_error_message(message: str) -> str:
    """Remove sensitive information from error messages.

    Removes:
    - URLs
    - File paths (Unix and Windows)
    - IP addresses
    - Email addresses

    Args:
        message: Original error message

    Returns:
        Sanitized error message with sensitive data removed
    """
    if not message:
        return message

    # Remove URLs first (before path matching can catch them)
    message = re.sub(r"https?://[^\s]+", "[URL]", message)

    # Remove Unix-style absolute paths
    message = re.sub(r"/[^\s:\"']+", "[PATH]", message)

    # Remove Windows-style paths
    message = re.sub(r"[A-Za-z]:\\[^\s:\"']+", "[PATH]", message)

    # Remove IP addresses
    message = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]", message)

    # Remove email addresses
    message = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", message
    )

    return message


def hash_session_id(session_id: Optional[str]) -> Optional[str]:
    """Hash a session ID for privacy.

    Uses SHA-256 with a prefix to create a one-way hash.

    Args:
        session_id: Original session ID (can be None)

    Returns:
        Hashed session ID or None if input is None
    """
    if session_id is None:
        return None

    # Use a consistent salt for reproducibility within the same deployment
    # This allows linking events from the same session without storing the original ID
    salted = f"mosaic_telemetry:{session_id}"
    return hashlib.sha256(salted.encode()).hexdigest()[:16]


def hash_username(username: Optional[str]) -> Optional[str]:
    """Hash a username for privacy in telemetry.

    Uses SHA-256 with a different salt than session IDs to create a one-way hash.
    This allows distinguishing users in telemetry without storing actual usernames.

    Args:
        username: HuggingFace username (can be None for anonymous users)

    Returns:
        Hashed username or None if input is None
    """
    if username is None:
        return None

    salted = f"mosaic_user:{username}"
    return hashlib.sha256(salted.encode()).hexdigest()[:16]


@dataclass
class UserInfo:
    """User information extracted from HF Spaces request.

    Attributes:
        is_logged_in: True if user has valid JWT token
        username: HuggingFace username (None if not logged in)
    """

    is_logged_in: bool = False
    username: Optional[str] = None


def extract_user_info(request, is_hf_spaces: bool = False, profile=None) -> UserInfo:
    """Extract user info from Gradio OAuth profile.

    With gr.LoginButton(), Gradio provides user info via gr.OAuthProfile
    (injected automatically into event handlers). The profile object has
    a `username` attribute with the HuggingFace username.

    Does NOT use request.username, which on HF Spaces returns the Space
    owner's username rather than the visitor's.

    Args:
        request: Gradio request object (unused, kept for API compatibility)
        is_hf_spaces: Whether running on HuggingFace Spaces (only extract on HF)
        profile: Gradio OAuthProfile object (injected by LoginButton OAuth flow)

    Returns:
        UserInfo with is_logged_in and username (or defaults if not available)

    Example:
        user_info = extract_user_info(request, IS_HF_SPACES, profile)
        if user_info.is_logged_in:
            print(f"User: {user_info.username}")
    """
    if not is_hf_spaces:
        return UserInfo()

    # On HF Spaces, only OAuthProfile reliably identifies the logged-in user.
    # request.username returns the Space owner's username (not the visitor's),
    # so we must NOT fall back to it.
    if profile is not None:
        try:
            username = getattr(profile, "username", None)
            if username:
                logger.info(f"Extracted user from OAuthProfile: {username}")
                return UserInfo(is_logged_in=True, username=username)
        except Exception as e:
            logger.debug(f"Could not extract username from OAuthProfile: {e}")

    logger.debug("User not logged in: no OAuthProfile available")
    return UserInfo()

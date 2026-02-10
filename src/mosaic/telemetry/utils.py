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


@dataclass
class UserInfo:
    """User information extracted from HF Spaces request.

    Attributes:
        is_logged_in: True if user has valid JWT token
        username: HuggingFace username (None if not logged in)
    """

    is_logged_in: bool = False
    username: Optional[str] = None


def extract_user_info(request, is_hf_spaces: bool = False) -> UserInfo:
    """Extract user info from Gradio request object.

    In Gradio 6.x+, the request object has a built-in `username` attribute
    that is automatically populated when a user is logged in on HuggingFace Spaces.

    This replaces the old JWT token parsing approach which stopped working when
    HuggingFace changed how they pass authentication information.

    Args:
        request: Gradio request object (can be None)
        is_hf_spaces: Whether running on HuggingFace Spaces (only extract on HF)

    Returns:
        UserInfo with is_logged_in and username (or defaults if not available)

    Example:
        user_info = extract_user_info(request, IS_HF_SPACES)
        if user_info.is_logged_in:
            print(f"User: {user_info.username}")
    """
    if not is_hf_spaces:
        return UserInfo()

    if request is None:
        logger.debug("Cannot extract user info: request is None")
        return UserInfo()

    try:
        # Gradio 6.x+ with OAuth provides username via request.username
        # However, this is a property that may require authentication middleware
        # Try to access it, handling potential middleware errors
        username = None
        
        try:
            username = request.username
        except Exception as auth_error:
            # AuthenticationMiddleware error or other property access issues
            logger.debug(f"Could not access request.username directly: {auth_error}")
            
            # Try alternative: check if request has kwargs with user info
            if hasattr(request, "kwargs"):
                username = request.kwargs.get("username") or request.kwargs.get("user")
                if username:
                    logger.debug(f"Found username in request.kwargs: {username}")
            
            # Try alternative: check request.request (underlying Starlette request)
            if not username and hasattr(request, "request"):
                try:
                    # Try to get user from state or session
                    if hasattr(request.request, "state") and hasattr(request.request.state, "user"):
                        user = request.request.state.user
                        if hasattr(user, "username"):
                            username = user.username
                        elif isinstance(user, dict):
                            username = user.get("username") or user.get("preferred_username")
                        logger.debug(f"Found user in request.request.state: {username}")
                except Exception as nested_error:
                    logger.debug(f"Could not access request.request.state.user: {nested_error}")

        if username:
            logger.info(f"Successfully extracted user info: {username}")
            return UserInfo(is_logged_in=True, username=username)
        else:
            logger.debug(
                "User info not available: username is None (user not logged in)"
            )
            return UserInfo()

    except Exception as e:
        logger.error(f"Unexpected error extracting user info: {e}")
        return UserInfo()

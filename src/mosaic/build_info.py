"""Runtime build information helpers for Mosaic."""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger


def _clean(value: str | None) -> str | None:
    """Normalize empty and placeholder build values to None."""
    if not value:
        return None
    value = value.strip()
    if not value or value.lower() in {"unknown", "unset"}:
        return None
    return value


def get_build_sha() -> str | None:
    """Return the build SHA injected into the runtime, if present."""
    return _clean(
        os.environ.get("MOSAIC_BUILD_SHA")
        or os.environ.get("GIT_SHA")
        or os.environ.get("SOURCE_COMMIT")
    )


def get_image_tag() -> str | None:
    """Return the image tag injected into the runtime, if present."""
    return _clean(os.environ.get("MOSAIC_IMAGE_TAG"))


def get_image_digest() -> str | None:
    """Return the image digest injected into the runtime, if present."""
    return _clean(os.environ.get("MOSAIC_IMAGE_DIGEST"))


def get_import_path() -> str:
    """Return the path of the imported Mosaic package."""
    return str(Path(__file__).resolve())


def format_version_string(version: str) -> str:
    """Format a human-usable version string for CLI output."""
    sha = get_build_sha()
    return f"mosaic {version} ({sha})" if sha else f"mosaic {version}"


def log_runtime_build_info(version: str) -> None:
    """Log build/runtime identity at process startup."""
    parts = [f"version={version}"]

    sha = get_build_sha()
    if sha:
        parts.append(f"build_sha={sha}")

    image_tag = get_image_tag()
    if image_tag:
        parts.append(f"image_tag={image_tag}")

    image_digest = get_image_digest()
    if image_digest:
        parts.append(f"image_digest={image_digest}")

    parts.append(f"module_path={get_import_path()}")
    logger.info("Runtime build info: " + " | ".join(parts))

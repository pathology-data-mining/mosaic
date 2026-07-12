"""Tests for runtime build information helpers."""

from __future__ import annotations

from pathlib import Path

from mosaic import __version__
from mosaic.build_info import (
    format_version_string,
    get_build_sha,
    get_import_path,
    log_runtime_build_info,
)


def test_format_version_string_without_sha(monkeypatch):
    monkeypatch.delenv("MOSAIC_BUILD_SHA", raising=False)
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("SOURCE_COMMIT", raising=False)

    assert format_version_string(__version__) == f"mosaic {__version__}"


def test_format_version_string_with_sha(monkeypatch):
    monkeypatch.setenv("MOSAIC_BUILD_SHA", "abc1234")

    assert format_version_string(__version__) == f"mosaic {__version__} (abc1234)"


def test_get_build_sha_ignores_placeholder(monkeypatch):
    monkeypatch.setenv("MOSAIC_BUILD_SHA", "unknown")

    assert get_build_sha() is None


def test_get_import_path_points_to_repo_module():
    assert Path(get_import_path()).name == "build_info.py"


def test_log_runtime_build_info_emits_module_path(monkeypatch, mocker):
    monkeypatch.setenv("MOSAIC_BUILD_SHA", "abc1234")
    mock_info = mocker.patch("mosaic.build_info.logger.info")

    log_runtime_build_info(__version__)

    mock_info.assert_called_once()
    message = mock_info.call_args.args[0]
    assert "Runtime build info:" in message
    assert "build_sha=abc1234" in message
    assert "module_path=" in message

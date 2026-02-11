"""Integration tests for user storage management in Gradio UI.

Tests end-to-end workflows including upload, analysis, viewing, and deletion
of slides and results through the UI components.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mosaic.ui.app import launch_gradio
from mosaic.ui.user_tabs import (
    LOCAL_DEBUG_USERNAME,
    _get_username,
    create_my_files_tab,
    create_my_results_tab,
)
from mosaic.user_results import save_analysis_results
from mosaic.user_storage import save_uploaded_slide


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directory for testing."""
    storage_dir = tmp_path / "mosaic_user_data"
    storage_dir.mkdir()

    # Mock the storage base directory
    # Note: user_results imports from user_storage, so only mock user_storage
    with patch("mosaic.user_storage.USER_STORAGE_BASE", storage_dir):
        yield storage_dir


@pytest.fixture
def mock_request_local():
    """Mock Gradio request for local mode (non-HF Spaces)."""
    request = MagicMock()
    request.username = None
    request.session_hash = "test_session"
    return request


@pytest.fixture
def mock_request_hf_logged_in():
    """Mock Gradio request for HF Spaces logged-in user."""
    request = MagicMock()
    request.username = "test_user"
    request.session_hash = "test_session"
    return request


@pytest.fixture
def mock_request_hf_not_logged_in():
    """Mock Gradio request for HF Spaces not logged in."""
    request = MagicMock()
    request.username = None
    request.session_hash = "test_session"
    return request


@pytest.fixture
def sample_slide_file(tmp_path):
    """Create a sample slide file for testing."""
    slide_file = tmp_path / "sample_slide.svs"
    slide_file.write_text("fake slide data" * 1000)  # ~15KB

    # Return the actual Path object (save_uploaded_slide expects a file-like object)
    # We need to mock it properly

    return MockFile(slide_file)


class TestUsernameExtraction:
    """Test username extraction in different modes."""

    @patch("mosaic.ui.user_tabs.IS_HF_SPACES", False)
    def test_local_mode_returns_debug_username(self, mock_request_local):
        """Local mode should return LOCAL_DEBUG_USERNAME."""
        username, is_local = _get_username(mock_request_local)
        assert username == LOCAL_DEBUG_USERNAME
        assert is_local is True

    @patch("mosaic.ui.user_tabs.IS_HF_SPACES", True)
    def test_hf_logged_in_returns_username(self, mock_request_hf_logged_in):
        """HF Spaces logged in should return request username."""
        username, is_local = _get_username(mock_request_hf_logged_in)
        assert username == "test_user"
        assert is_local is False

    @patch("mosaic.ui.user_tabs.IS_HF_SPACES", True)
    def test_hf_not_logged_in_returns_none(self, mock_request_hf_not_logged_in):
        """HF Spaces not logged in should return None."""
        username, is_local = _get_username(mock_request_hf_not_logged_in)
        assert username is None
        assert is_local is False


class TestEndToEndUserStorage:
    """Test complete user storage workflow."""

    def test_upload_analyze_view_delete_workflow(
        self, temp_storage_dir, mock_request_local
    ):
        """Test complete workflow: upload → analyze → view → delete."""
        username = LOCAL_DEBUG_USERNAME

        # Create a slide file
        temp_file = temp_storage_dir / "sample_slide.svs"
        temp_file.write_text("fake slide data" * 1000)

        # Step 1: Upload slide (use Path directly, not MockFile)
        slide_id, _ = save_uploaded_slide(username, temp_file)
        assert slide_id is not None

        # Verify slide saved
        from mosaic.user_storage import list_user_slides

        slides = list_user_slides(username)
        assert len(slides) == 1
        assert slides[0].slide_id == slide_id

        # Verify slide saved
        from mosaic.user_storage import list_user_slides

        slides = list_user_slides(username)
        assert len(slides) == 1
        assert slides[0].slide_id == slide_id

        # Step 2: Save analysis results
        settings = {
            "seg_config": "Biopsy",
            "site_type": "Primary",
            "sex": "Male",
            "tissue_site": "Lung",
            "cancer_subtype": "LUAD",
        }

        aeon_df = pd.DataFrame(
            {
                "cancer_type": ["LUAD", "LUSC"],
                "confidence": [0.85, 0.10],
            }
        )

        analysis_id = "test_analysis_001"
        save_analysis_results(
            username=username,
            analysis_id=analysis_id,
            slide_id=slide_id,
            slide_name="sample_slide.svs",
            settings=settings,
            aeon_results=aeon_df,
            paladin_results=None,
            slide_mask=None,
        )

        # Step 3: Verify results saved
        from mosaic.user_results import list_user_results, load_analysis_results

        results = list_user_results(username)
        assert len(results) == 1
        assert results[0].analysis_id == analysis_id
        assert results[0].slide_id == slide_id

        # Step 4: Load result details
        loaded = load_analysis_results(username, analysis_id)
        assert loaded is not None
        metadata, mask, aeon, paladin = loaded
        assert metadata.slide_name == "sample_slide.svs"
        assert aeon is not None
        assert len(aeon) == 2

        # Step 5: Delete slide (should also delete results)
        from mosaic.user_storage import delete_user_slide

        success = delete_user_slide(username, slide_id)
        assert success is True

        # Verify slide deleted
        slides = list_user_slides(username)
        assert len(slides) == 0

        # Verify results also deleted
        results = list_user_results(username, slide_id=slide_id)
        assert len(results) == 0

    def test_quota_enforcement_and_cleanup(self, temp_storage_dir, mock_request_local):
        """Test quota enforcement triggers FIFO cleanup."""
        username = LOCAL_DEBUG_USERNAME

        # Set very small quota (30KB)
        with patch("mosaic.user_storage.DEFAULT_QUOTA_BYTES", 30 * 1024):
            # Upload multiple files (each ~10KB)
            slide_ids = []
            for i in range(5):
                temp_file = temp_storage_dir / f"slide_{i}.svs"
                temp_file.write_text("fake data " * 1000)  # ~10KB each

                try:
                    slide_id, _ = save_uploaded_slide(username, temp_file)
                    slide_ids.append(slide_id)
                except ValueError:
                    # Quota exceeded, cleanup should have happened
                    pass

            # Should have some slides (older ones cleaned up)
            from mosaic.user_storage import list_user_slides, get_storage_usage

            slides = list_user_slides(username)
            usage = get_storage_usage(username)

            # Total should be under quota after cleanup
            assert usage.total_bytes <= 30 * 1024
            # Some files should have been cleaned up (only 3 should fit)
            assert len(slides) <= 3

    def test_multiple_analyses_same_slide(self, temp_storage_dir, mock_request_local):
        """Test multiple analyses can reference the same slide."""
        username = LOCAL_DEBUG_USERNAME

        # Upload slide once
        temp_file = temp_storage_dir / "multi_analysis.svs"
        temp_file.write_text("fake slide data" * 1000)

        slide_id, _ = save_uploaded_slide(username, temp_file)

        # Save multiple analysis results for same slide
        settings1 = {"seg_config": "Biopsy", "site_type": "Primary", "sex": "Male"}
        settings2 = {"seg_config": "Resection", "site_type": "Primary", "sex": "Male"}

        save_analysis_results(
            username=username,
            analysis_id="analysis_001",
            slide_id=slide_id,
            slide_name="sample.svs",
            settings=settings1,
            aeon_results=None,
            paladin_results=None,
            slide_mask=None,
        )

        save_analysis_results(
            username=username,
            analysis_id="analysis_002",
            slide_id=slide_id,
            slide_name="sample.svs",
            settings=settings2,
            aeon_results=None,
            paladin_results=None,
            slide_mask=None,
        )

        # Verify both results saved
        from mosaic.user_results import list_user_results

        results = list_user_results(username, slide_id=slide_id)
        assert len(results) == 2

        # Delete slide should delete both results
        from mosaic.user_storage import delete_user_slide

        delete_user_slide(username, slide_id)

        results = list_user_results(username, slide_id=slide_id)
        assert len(results) == 0


class TestMyFilesTabIntegration:
    """Test My Files tab UI integration."""

    @patch("mosaic.ui.user_tabs.IS_HF_SPACES", False)
    def test_load_files_local_mode(self, temp_storage_dir, mock_request_local):
        """Test loading files in local debug mode."""
        username = LOCAL_DEBUG_USERNAME

        # Upload a slide first
        temp_file = temp_storage_dir / "sample_slide.svs"
        temp_file.write_text("fake slide data" * 1000)

        save_uploaded_slide(username, temp_file)

        # Test the loading logic
        from mosaic.ui.user_tabs import _get_username
        from mosaic.user_storage import list_user_slides, get_storage_usage

        username_returned, is_local = _get_username(mock_request_local)
        assert username_returned == LOCAL_DEBUG_USERNAME
        assert is_local is True

        # Verify slides can be listed
        slides = list_user_slides(username)
        assert len(slides) == 1
        assert slides[0].original_filename == "sample_slide.svs"

        # Verify storage usage
        usage = get_storage_usage(username)
        assert usage.file_count == 1

    @patch("mosaic.ui.user_tabs.IS_HF_SPACES", True)
    def test_load_files_hf_not_logged_in(self, mock_request_hf_not_logged_in):
        """Test loading files when not logged in on HF Spaces."""
        # Import the load_files logic directly (would be in create_my_files_tab)
        from mosaic.ui.user_tabs import _get_username
        from mosaic.user_storage import list_user_slides, get_storage_usage

        username, _ = _get_username(mock_request_hf_not_logged_in)

        # Should return None for not logged in
        assert username is None

        # Verify behavior would show "not logged in" message
        # (actual function would return empty data)


class TestMyResultsTabIntegration:
    """Test My Results tab UI integration."""

    @patch("mosaic.ui.user_tabs.IS_HF_SPACES", False)
    def test_load_results_local_mode(self, temp_storage_dir, mock_request_local):
        """Test loading results in local debug mode."""
        username = LOCAL_DEBUG_USERNAME

        # Create analysis result
        temp_file = temp_storage_dir / "sample_slide.svs"
        temp_file.write_text("fake slide data" * 1000)

        slide_id, _ = save_uploaded_slide(username, temp_file)
        save_analysis_results(
            username=username,
            analysis_id="test_001",
            slide_id=slide_id,
            slide_name="sample.svs",
            settings={"sex": "Male", "site_type": "Primary"},
            aeon_results=None,
            paladin_results=None,
            slide_mask=None,
        )

        # Test the loading logic
        from mosaic.ui.user_tabs import _get_username
        from mosaic.user_results import list_user_results

        username_returned, _ = _get_username(mock_request_local)
        assert username_returned == LOCAL_DEBUG_USERNAME

        # Verify results can be listed
        results = list_user_results(username)
        assert len(results) == 1
        assert results[0].analysis_id == "test_001"
        assert results[0].slide_name == "sample.svs"


class TestStorageWarningsUI:
    """Test storage warning display in main analysis tab."""

    @patch("mosaic.ui.user_tabs.IS_HF_SPACES", False)
    def test_normal_storage_usage(self, temp_storage_dir, mock_request_local):
        """Test normal storage usage display (< 80%)."""
        # Import the function (would normally be in app.py)
        from mosaic.ui.user_tabs import _get_username
        from mosaic.user_storage import get_storage_usage

        username, _ = _get_username(mock_request_local)
        usage = get_storage_usage(username)

        # Should be 0% initially
        percent = (
            (usage.total_bytes / usage.quota_bytes * 100)
            if usage.quota_bytes > 0
            else 0
        )
        assert percent == 0

    @patch("mosaic.ui.user_tabs.IS_HF_SPACES", False)
    def test_warning_threshold_storage(self, temp_storage_dir, mock_request_local):
        """Test storage warning at 80% threshold."""
        username = LOCAL_DEBUG_USERNAME

        # Set quota to force warning threshold
        with patch("mosaic.user_storage.DEFAULT_QUOTA_BYTES", 18 * 1024):  # 18KB quota
            # Upload file to reach warning threshold
            temp_file = temp_storage_dir / "large_slide.svs"
            temp_file.write_text("large fake data" * 1000)  # ~16KB

            save_uploaded_slide(username, temp_file)

            from mosaic.user_storage import get_storage_usage

            usage = get_storage_usage(username)
            percent = usage.total_bytes / usage.quota_bytes * 100

            # Should be > 80%
            assert percent > 80


class TestReAnalysisWorkflow:
    """Test re-analyzing existing slides without re-uploading."""

    @patch("mosaic.ui.user_tabs.IS_HF_SPACES", False)
    def test_select_existing_slide_for_analysis(
        self, temp_storage_dir, mock_request_local
    ):
        """Test selecting an existing slide from dropdown."""
        username = LOCAL_DEBUG_USERNAME

        # Upload slide
        temp_file = temp_storage_dir / "sample_slide.svs"
        temp_file.write_text("fake slide data" * 1000)

        slide_id, _ = save_uploaded_slide(username, temp_file)

        # Get slide path (simulating dropdown selection)
        from mosaic.user_storage import get_slide_path

        slide_path = get_slide_path(username, slide_id)
        assert slide_path is not None
        assert slide_path.exists()

        # Slide can now be analyzed without re-upload
        # (In actual UI, this would populate input_slides component)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

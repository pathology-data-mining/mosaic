"""Unit tests for user_results module."""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from mosaic.user_results import (
    ResultMetadata,
    create_results_zip,
    delete_analysis_results,
    delete_results_for_slide,
    list_user_results,
    load_analysis_results,
    save_analysis_results,
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
def test_analysis_id():
    """Test analysis ID."""
    return "analysis-123"


@pytest.fixture
def test_slide_id():
    """Test slide ID."""
    return "slide-456"


@pytest.fixture
def test_results():
    """Test analysis results."""
    aeon_df = pd.DataFrame(
        {"Confidence": [0.95, 0.03, 0.02]}, index=["BRCA", "LUAD", "COAD"]
    )
    paladin_df = pd.DataFrame(
        {"Biomarker": ["ER", "PR", "HER2"], "Prediction": [1, 1, 0]}
    )
    mask = Image.new("RGB", (100, 100), color="red")

    return aeon_df, paladin_df, mask


def test_save_analysis_results(
    temp_storage_base, test_user, test_analysis_id, test_slide_id, test_results
):
    """Test saving analysis results."""
    aeon_df, paladin_df, mask = test_results

    settings = {
        "site_type": "Primary",
        "cancer_subtype": "Unknown",
        "ihc_subtype": "",
        "seg_config": "Biopsy",
    }

    result = save_analysis_results(
        username=test_user,
        analysis_id=test_analysis_id,
        slide_id=test_slide_id,
        slide_name="test_slide.svs",
        settings=settings,
        aeon_results=aeon_df,
        paladin_results=paladin_df,
        slide_mask=mask,
    )

    assert result is True

    # Verify files were created
    from mosaic.user_results import _get_results_dir

    results_dir = _get_results_dir(test_user, test_analysis_id)
    assert (results_dir / "metadata.json").exists()
    assert (results_dir / "aeon_results.csv").exists()
    assert (results_dir / "paladin_results.csv").exists()
    assert (results_dir / "slide_mask.png").exists()


def test_save_analysis_results_partial(
    temp_storage_base, test_user, test_analysis_id, test_slide_id
):
    """Test saving results with some None values."""
    settings = {"site_type": "Primary"}

    result = save_analysis_results(
        username=test_user,
        analysis_id=test_analysis_id,
        slide_id=test_slide_id,
        slide_name="test_slide.svs",
        settings=settings,
        aeon_results=None,
        paladin_results=None,
        slide_mask=None,
    )

    assert result is True

    # Metadata should still exist
    from mosaic.user_results import _get_results_dir

    results_dir = _get_results_dir(test_user, test_analysis_id)
    assert (results_dir / "metadata.json").exists()


def test_load_analysis_results(
    temp_storage_base, test_user, test_analysis_id, test_slide_id, test_results
):
    """Test loading analysis results."""
    aeon_df, paladin_df, mask = test_results
    settings = {"site_type": "Primary"}

    # Save results first
    save_analysis_results(
        username=test_user,
        analysis_id=test_analysis_id,
        slide_id=test_slide_id,
        slide_name="test_slide.svs",
        settings=settings,
        aeon_results=aeon_df,
        paladin_results=paladin_df,
        slide_mask=mask,
    )

    # Load results
    loaded = load_analysis_results(test_user, test_analysis_id)
    assert loaded is not None

    metadata, loaded_mask, loaded_aeon, loaded_paladin = loaded

    # Check metadata
    assert isinstance(metadata, ResultMetadata)
    assert metadata.analysis_id == test_analysis_id
    assert metadata.slide_id == test_slide_id
    assert metadata.slide_name == "test_slide.svs"
    assert metadata.cancer_subtype == "BRCA"  # Top prediction

    # Check results
    assert isinstance(loaded_mask, Image.Image)
    assert isinstance(loaded_aeon, pd.DataFrame)
    assert isinstance(loaded_paladin, pd.DataFrame)


def test_load_analysis_results_not_found(temp_storage_base, test_user):
    """Test loading non-existent results."""
    result = load_analysis_results(test_user, "nonexistent")
    assert result is None


def test_list_user_results(temp_storage_base, test_user, test_slide_id, test_results):
    """Test listing user results."""
    aeon_df, paladin_df, mask = test_results
    settings = {"site_type": "Primary"}

    # Save multiple results
    save_analysis_results(
        username=test_user,
        analysis_id="analysis-1",
        slide_id=test_slide_id,
        slide_name="slide1.svs",
        settings=settings,
        aeon_results=aeon_df,
        paladin_results=paladin_df,
        slide_mask=mask,
    )

    save_analysis_results(
        username=test_user,
        analysis_id="analysis-2",
        slide_id="slide-other",
        slide_name="slide2.svs",
        settings=settings,
        aeon_results=aeon_df,
        paladin_results=paladin_df,
        slide_mask=mask,
    )

    # List all results
    all_results = list_user_results(test_user)
    assert len(all_results) == 2

    # List results for specific slide
    slide_results = list_user_results(test_user, slide_id=test_slide_id)
    assert len(slide_results) == 1
    assert slide_results[0].analysis_id == "analysis-1"


def test_delete_analysis_results(
    temp_storage_base, test_user, test_analysis_id, test_slide_id, test_results
):
    """Test deleting analysis results."""
    aeon_df, paladin_df, mask = test_results
    settings = {"site_type": "Primary"}

    # Save results
    save_analysis_results(
        username=test_user,
        analysis_id=test_analysis_id,
        slide_id=test_slide_id,
        slide_name="test_slide.svs",
        settings=settings,
        aeon_results=aeon_df,
        paladin_results=paladin_df,
        slide_mask=mask,
    )

    # Verify results exist
    from mosaic.user_results import _get_results_dir

    results_dir = _get_results_dir(test_user, test_analysis_id)
    assert results_dir.exists()

    # Delete results
    assert delete_analysis_results(test_user, test_analysis_id) is True

    # Verify results are gone
    assert not results_dir.exists()

    # Try deleting again (should return False)
    assert delete_analysis_results(test_user, test_analysis_id) is False


def test_delete_results_for_slide(
    temp_storage_base, test_user, test_slide_id, test_results
):
    """Test deleting all results for a slide."""
    aeon_df, paladin_df, mask = test_results
    settings = {"site_type": "Primary"}

    # Save multiple results for the same slide
    save_analysis_results(
        username=test_user,
        analysis_id="analysis-1",
        slide_id=test_slide_id,
        slide_name="test_slide.svs",
        settings=settings,
        aeon_results=aeon_df,
        paladin_results=paladin_df,
        slide_mask=mask,
    )

    save_analysis_results(
        username=test_user,
        analysis_id="analysis-2",
        slide_id=test_slide_id,
        slide_name="test_slide.svs",
        settings=settings,
        aeon_results=aeon_df,
        paladin_results=paladin_df,
        slide_mask=mask,
    )

    # Save result for different slide
    save_analysis_results(
        username=test_user,
        analysis_id="analysis-3",
        slide_id="other-slide",
        slide_name="other.svs",
        settings=settings,
        aeon_results=aeon_df,
        paladin_results=paladin_df,
        slide_mask=mask,
    )

    # Delete results for test_slide_id
    deleted = delete_results_for_slide(test_user, test_slide_id)
    assert deleted == 2

    # Verify correct results were deleted
    remaining = list_user_results(test_user)
    assert len(remaining) == 1
    assert remaining[0].slide_id == "other-slide"


def test_create_results_zip(
    temp_storage_base, test_user, test_analysis_id, test_slide_id, test_results
):
    """Test creating ZIP file from results."""
    aeon_df, paladin_df, mask = test_results
    settings = {"site_type": "Primary"}

    # Save results
    save_analysis_results(
        username=test_user,
        analysis_id=test_analysis_id,
        slide_id=test_slide_id,
        slide_name="test_slide.svs",
        settings=settings,
        aeon_results=aeon_df,
        paladin_results=paladin_df,
        slide_mask=mask,
    )

    # Create ZIP
    zip_path = create_results_zip(test_user, test_analysis_id)
    assert zip_path is not None
    assert zip_path.exists()
    assert zip_path.suffix == ".zip"

    # Verify ZIP contains expected files
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zipf:
        names = zipf.namelist()
        assert "metadata.json" in names
        assert "aeon_results.csv" in names
        assert "paladin_results.csv" in names
        assert "slide_mask.png" in names


def test_create_results_zip_not_found(temp_storage_base, test_user):
    """Test creating ZIP for non-existent results."""
    zip_path = create_results_zip(test_user, "nonexistent")
    assert zip_path is None


def test_result_metadata_dataclass():
    """Test ResultMetadata dataclass."""
    metadata = ResultMetadata(
        analysis_id="test-analysis",
        slide_id="test-slide",
        slide_name="test.svs",
        timestamp="2024-01-01T00:00:00",
        settings={"site_type": "Primary"},
        cancer_subtype="BRCA",
    )

    assert metadata.analysis_id == "test-analysis"
    assert metadata.slide_id == "test-slide"
    assert metadata.cancer_subtype == "BRCA"

"""Shared fixtures and utilities for UI and CLI tests.

This module provides reusable fixtures for testing the Mosaic Gradio UI and CLI,
including mock file objects, settings DataFrames, cancer subtype mappings, and
utility functions for test setup/teardown.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import numpy as np
import pytest
from PIL import Image

# ============================================================================
# File and Path Fixtures
# ============================================================================


@pytest.fixture
def test_slide_path():
    """Path to actual test slide for integration tests."""
    return Path("tests/testdata/948176.svs")


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="mosaic_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_user_dir(temp_output_dir):
    """Mock user directory (same as temp_output_dir for simplicity)."""
    return temp_output_dir


# ============================================================================
# Mock File Upload Fixtures
# ============================================================================


@pytest.fixture
def sample_files_single():
    """Mock single file upload."""
    mock_file = Mock()
    mock_file.name = "test_slide_1.svs"
    return [mock_file]


@pytest.fixture
def sample_files_multiple():
    """Mock multiple file uploads (3 files)."""
    files = []
    for i in range(1, 4):
        mock_file = Mock()
        mock_file.name = f"test_slide_{i}.svs"
        files.append(mock_file)
    return files


def create_mock_file(filename):
    """Create a mock file object with specified filename.

    Args:
        filename: Name for the mock file

    Returns:
        Mock object with .name attribute
    """
    mock_file = Mock()
    mock_file.name = filename
    return mock_file


# ============================================================================
# Settings DataFrame Fixtures
# ============================================================================


@pytest.fixture
def sample_settings_df():
    """Sample settings DataFrame with 3 slides."""
    return pd.DataFrame(
        {
            "Slide": ["slide1.svs", "slide2.svs", "slide3.svs"],
            "Site Type": ["Primary", "Metastatic", "Primary"],
            "Sex": ["Male", "Female", "Male"],
            "Tissue Site": ["Lung", "Liver", "Unknown"],
            "Cancer Subtype": ["Unknown", "Lung Adenocarcinoma (LUAD)", "Unknown"],
            "IHC Subtype": ["", "", ""],
            "Segmentation Config": ["Biopsy", "Resection", "TCGA"],
        }
    )


def create_settings_df(n_rows, **kwargs):
    """Generate a test settings DataFrame with specified number of rows.

    Args:
        n_rows: Number of rows to generate
        **kwargs: Column overrides (e.g., site_type="Metastatic")

    Returns:
        DataFrame with SETTINGS_COLUMNS
    """
    defaults = {
        "Slide": [f"slide_{i}.svs" for i in range(1, n_rows + 1)],
        "Site Type": ["Primary"] * n_rows,
        "Sex": ["Unknown"] * n_rows,
        "Tissue Site": ["Unknown"] * n_rows,
        "Cancer Subtype": ["Unknown"] * n_rows,
        "IHC Subtype": [""] * n_rows,
        "Segmentation Config": ["Biopsy"] * n_rows,
    }

    # Override with any provided kwargs
    for key, value in kwargs.items():
        column_name = key.replace("_", " ").title()
        if column_name in defaults:
            if isinstance(value, list):
                defaults[column_name] = value
            else:
                defaults[column_name] = [value] * n_rows

    return pd.DataFrame(defaults)


# ============================================================================
# CSV File Fixtures
# ============================================================================


@pytest.fixture
def sample_csv_valid():
    """Temporary CSV file with valid settings."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(
            "Slide,Site Type,Sex,Tissue Site,Cancer Subtype,IHC Subtype,Segmentation Config\n"
        )
        f.write("slide1.svs,Primary,Unknown,Lung,Unknown,,Biopsy\n")
        f.write(
            "slide2.svs,Metastatic,Female,Liver,Lung Adenocarcinoma (LUAD),,Resection\n"
        )
        f.write("slide3.svs,Primary,Male,Unknown,Unknown,,TCGA\n")
        f.flush()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_csv_invalid():
    """Temporary CSV file with invalid values (for validation testing)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(
            "Slide,Site Type,Sex,Tissue Site,Cancer Subtype,IHC Subtype,Segmentation Config\n"
        )
        f.write(
            "slide1.svs,InvalidSite,InvalidSex,InvalidTissue,InvalidSubtype,InvalidIHC,InvalidConfig\n"
        )
        f.write(
            "slide2.svs,Primary,Unknown,Lung,BRCA,HR+/HER2+,Biopsy\n"
        )  # Valid breast cancer
        f.flush()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_csv_minimal():
    """Temporary CSV file with only required columns (missing optional columns)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Slide,Site Type,Cancer Subtype\n")
        f.write("slide1.svs,Primary,Unknown\n")
        f.write("slide2.svs,Metastatic,LUAD\n")
        f.flush()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# ============================================================================
# Cancer Subtype Mapping Fixtures
# ============================================================================


@pytest.fixture
def mock_cancer_subtype_maps():
    """Mock cancer subtype mappings for testing."""
    cancer_subtype_name_map = {
        "Unknown": "UNK",
        "Lung Adenocarcinoma (LUAD)": "LUAD",
        "Breast Invasive Carcinoma (BRCA)": "BRCA",
        "Colorectal Adenocarcinoma (COAD)": "COAD",
        "Prostate Adenocarcinoma (PRAD)": "PRAD",
    }

    reversed_cancer_subtype_name_map = {
        "UNK": "Unknown",
        "LUAD": "Lung Adenocarcinoma (LUAD)",
        "BRCA": "Breast Invasive Carcinoma (BRCA)",
        "COAD": "Colorectal Adenocarcinoma (COAD)",
        "PRAD": "Prostate Adenocarcinoma (PRAD)",
    }

    cancer_subtypes = ["LUAD", "BRCA", "COAD", "PRAD"]

    return cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes


# ============================================================================
# Mock Analysis Results Fixtures
# ============================================================================


@pytest.fixture
def mock_analyze_slide_results():
    """Mock results from analyze_slide function."""
    # Create a simple test mask image
    mask = Image.new("RGB", (100, 100), color="red")

    # Create Aeon results DataFrame
    aeon_results = pd.DataFrame(
        {
            "Cancer Subtype": ["LUAD"],
            "Confidence": [0.95],
        }
    )

    # Create Paladin results DataFrame (NOTE: No "Slide" column - that gets added by CLI/UI)
    paladin_results = pd.DataFrame(
        {
            "Cancer Subtype": ["LUAD", "LUAD", "LUAD"],
            "Biomarker": ["TP53", "KRAS", "EGFR"],
            "Score": [0.85, 0.72, 0.63],
        }
    )

    return (mask, aeon_results, paladin_results)


@pytest.fixture
def mock_model_cache():
    """Mock ModelCache with test models."""
    from unittest.mock import Mock

    cache = Mock()
    cache.ctranspath_model = Mock()
    cache.optimus_model = Mock()
    cache.marker_classifier = Mock()
    cache.aeon_model = Mock()
    cache.paladin_models = {}
    cache.device = Mock()
    cache.cleanup = Mock()

    return cache


# ============================================================================
# CLI Argument Fixtures
# ============================================================================


@pytest.fixture
def cli_args_single():
    """Mock argparse Namespace for single-slide mode."""
    from argparse import Namespace

    return Namespace(
        debug=False,
        server_name="0.0.0.0",
        server_port=None,
        share=False,
        slide_path="tests/testdata/948176.svs",
        slide_csv=None,
        output_dir="test_output",
        site_type="Primary",
        sex="Unknown",
        tissue_site="Unknown",
        cancer_subtype="Unknown",
        ihc_subtype="",
        segmentation_config="Biopsy",
        num_workers=4,
    )


@pytest.fixture
def cli_args_batch(sample_csv_valid):
    """Mock argparse Namespace for batch mode."""
    from argparse import Namespace

    return Namespace(
        debug=False,
        server_name="0.0.0.0",
        server_port=None,
        share=False,
        slide_path=None,
        slide_csv=sample_csv_valid,
        output_dir="test_output",
        site_type="Primary",
        sex="Unknown",
        tissue_site="Unknown",
        cancer_subtype="Unknown",
        ihc_subtype="",
        segmentation_config="Biopsy",
        num_workers=4,
    )


# ============================================================================
# Utility Functions
# ============================================================================


def verify_csv_output(path, expected_columns):
    """Validate CSV file structure.

    Args:
        path: Path to CSV file
        expected_columns: List of expected column names

    Returns:
        DataFrame loaded from CSV

    Raises:
        AssertionError: If CSV is invalid or missing columns
    """
    assert Path(path).exists(), f"CSV file not found: {path}"

    df = pd.read_csv(path)
    assert not df.empty, f"CSV file is empty: {path}"

    missing_cols = set(expected_columns) - set(df.columns)
    assert not missing_cols, f"Missing columns in CSV: {missing_cols}"

    return df


def mock_gradio_components():
    """Context manager to mock Gradio component classes.

    Usage:
        with mock_gradio_components() as mocks:
            # Gradio components are mocked
            result = function_that_returns_gr_components()
            # Verify mocks
            assert mocks['Dataframe'].called
    """
    from unittest.mock import patch, Mock

    mocks = {
        "Dataframe": Mock(return_value=Mock()),
        "File": Mock(return_value=Mock()),
        "DownloadButton": Mock(return_value=Mock()),
        "Dropdown": Mock(return_value=Mock()),
        "Gallery": Mock(return_value=Mock()),
        "Error": Exception,  # gr.Error is an exception
        "Warning": Mock(),
    }

    patches = []
    for name, mock_obj in mocks.items():
        patch_obj = patch(f"mosaic.ui.app.gr.{name}", mock_obj)
        patches.append(patch_obj)

    # Start all patches
    for p in patches:
        p.start()

    try:
        yield mocks
    finally:
        # Stop all patches
        for p in patches:
            p.stop()

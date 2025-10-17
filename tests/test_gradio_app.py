"""Unit tests for mosaic.gradio_app module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from mosaic.gradio_app import (
    IHC_SUBTYPES,
    SETTINGS_COLUMNS,
    load_settings,
    validate_settings,
)


class TestConstants:
    """Test constants in gradio_app."""

    def test_ihc_subtypes_list(self):
        """Test that IHC_SUBTYPES is a list."""
        assert isinstance(IHC_SUBTYPES, list)

    def test_ihc_subtypes_has_entries(self):
        """Test that IHC_SUBTYPES has entries."""
        assert len(IHC_SUBTYPES) > 0

    def test_ihc_subtypes_contains_expected_values(self):
        """Test that IHC_SUBTYPES contains expected breast cancer subtypes."""
        expected_subtypes = ["HR+/HER2+", "HR+/HER2-", "HR-/HER2+", "HR-/HER2-"]
        for subtype in expected_subtypes:
            assert subtype in IHC_SUBTYPES

    def test_ihc_subtypes_includes_empty_string(self):
        """Test that IHC_SUBTYPES includes empty string for non-breast cancers."""
        assert "" in IHC_SUBTYPES

    def test_settings_columns_list(self):
        """Test that SETTINGS_COLUMNS is a list."""
        assert isinstance(SETTINGS_COLUMNS, list)

    def test_settings_columns_required_fields(self):
        """Test that SETTINGS_COLUMNS contains required fields."""
        required_fields = [
            "Slide",
            "Site Type",
            "Cancer Subtype",
            "IHC Subtype",
            "Segmentation Config",
        ]
        for field in required_fields:
            assert field in SETTINGS_COLUMNS


class TestLoadSettings:
    """Test load_settings function."""

    @pytest.fixture
    def temp_settings_csv(self):
        """Create a temporary settings CSV file with all columns."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as f:
            f.write("Slide,Site Type,Cancer Subtype,IHC Subtype,Segmentation Config\n")
            f.write("slide1.svs,Primary,Unknown,,Biopsy\n")
            f.write("slide2.svs,Metastatic,Unknown,,Resection\n")
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()

    @pytest.fixture
    def temp_minimal_settings_csv(self):
        """Create a temporary settings CSV file with minimal columns."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as f:
            f.write("Slide,Site Type\n")
            f.write("slide1.svs,Primary\n")
            f.write("slide2.svs,Metastatic\n")
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()

    def test_load_settings_returns_dataframe(self, temp_settings_csv):
        """Test that load_settings returns a DataFrame."""
        df = load_settings(temp_settings_csv)
        assert isinstance(df, pd.DataFrame)

    def test_load_settings_has_all_columns(self, temp_settings_csv):
        """Test that all required columns are present."""
        df = load_settings(temp_settings_csv)
        for col in SETTINGS_COLUMNS:
            assert col in df.columns

    def test_load_settings_adds_missing_columns(self, temp_minimal_settings_csv):
        """Test that missing columns are added with defaults."""
        df = load_settings(temp_minimal_settings_csv)
        assert "Segmentation Config" in df.columns
        assert "Cancer Subtype" in df.columns
        assert "IHC Subtype" in df.columns
        assert df["Segmentation Config"].iloc[0] == "Biopsy"
        assert df["Cancer Subtype"].iloc[0] == "Unknown"
        assert df["IHC Subtype"].iloc[0] == ""

    def test_load_settings_preserves_data(self, temp_settings_csv):
        """Test that data is preserved correctly."""
        df = load_settings(temp_settings_csv)
        assert len(df) == 2
        assert df["Slide"].iloc[0] == "slide1.svs"
        assert df["Site Type"].iloc[0] == "Primary"

    def test_load_settings_missing_required_column_raises_error(self):
        """Test that missing required column raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        ) as f:
            f.write("RandomColumn\n")
            f.write("value\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Missing required column"):
                load_settings(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_settings_filters_to_settings_columns(self, temp_settings_csv):
        """Test that only SETTINGS_COLUMNS are returned."""
        df = load_settings(temp_settings_csv)
        assert list(df.columns) == SETTINGS_COLUMNS


class TestGetOncotreeCodeName:
    """Test get_oncotree_code_name function."""

    def test_oncotree_code_name_caching(self, mocker):
        """Test that oncotree code names are cached."""
        from mosaic.gradio_app import get_oncotree_code_name, oncotree_code_map

        # Mock the requests.get call
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"name": "Lung Adenocarcinoma"}]
        mocker.patch("requests.get", return_value=mock_response)

        # Clear the cache
        oncotree_code_map.clear()

        # First call should populate cache
        code = "LUAD"
        result1 = get_oncotree_code_name(code)

        # Cache should now contain the code
        assert code in oncotree_code_map

        # Second call should use cache
        result2 = get_oncotree_code_name(code)
        assert result1 == result2

    def test_oncotree_code_name_returns_string(self, mocker):
        """Test that function returns a string."""
        from mosaic.gradio_app import get_oncotree_code_name, oncotree_code_map

        # Mock the requests.get call
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"name": "Lung Adenocarcinoma"}]
        mocker.patch("requests.get", return_value=mock_response)

        # Clear cache first
        oncotree_code_map.clear()

        result = get_oncotree_code_name("LUAD")
        assert isinstance(result, str)

    def test_oncotree_invalid_code_returns_unknown(self, mocker):
        """Test that invalid code returns 'Unknown'."""
        from mosaic.gradio_app import get_oncotree_code_name, oncotree_code_map

        # Mock the requests.get call to return empty response (no matching codes)
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []  # Empty list means no matching codes found
        mocker.patch("requests.get", return_value=mock_response)

        # Clear cache and use an invalid code
        oncotree_code_map.clear()
        result = get_oncotree_code_name("INVALID_CODE_XYZ123")
        assert result == "Unknown"


class TestExportToCsv:
    """Test export_to_csv function."""

    def test_export_to_csv_returns_path(self):
        """Test that export_to_csv returns a file path."""
        from mosaic.gradio_app import export_to_csv

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = export_to_csv(df)
        assert isinstance(result, str)
        assert result.endswith(".csv")
        # Clean up
        Path(result).unlink(missing_ok=True)

    def test_export_to_csv_creates_file(self):
        """Test that export_to_csv creates a CSV file."""
        from mosaic.gradio_app import export_to_csv

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = export_to_csv(df)
        assert Path(result).exists()
        # Clean up
        Path(result).unlink()

    def test_export_to_csv_with_empty_dataframe_raises_error(self):
        """Test that exporting empty DataFrame raises error."""
        from mosaic.gradio_app import export_to_csv
        import gradio as gr

        df = pd.DataFrame()
        with pytest.raises(gr.Error):
            export_to_csv(df)

    def test_export_to_csv_with_none_raises_error(self):
        """Test that exporting None raises error."""
        from mosaic.gradio_app import export_to_csv
        import gradio as gr

        with pytest.raises(gr.Error):
            export_to_csv(None)

"""Tests for Gradio UI components and their interactions.

This module tests the Mosaic Gradio UI components, including:
- Settings validation
- Analysis workflow
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import after mocking (mocks are set up in conftest.py)
from mosaic.ui.app import (
    analyze_slides,
    set_cancer_subtype_maps,
)
from mosaic.ui.utils import SETTINGS_COLUMNS


class TestSettingsValidation:
    """Test settings validation logic."""

    @patch("mosaic.ui.utils.gr.Warning")
    def test_invalid_cancer_subtype_defaults_to_unknown(
        self, mock_warning, mock_cancer_subtype_maps
    ):
        """Test invalid cancer subtype generates warning and defaults to Unknown."""
        from mosaic.ui.utils import validate_settings

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )

        # Create DataFrame with invalid cancer subtype
        df = pd.DataFrame(
            {
                "Slide": ["test.svs"],
                "Site Type": ["Primary"],
                "Sex": ["Unknown"],
                "Tissue Site": ["Unknown"],
                "Cancer Subtype": ["InvalidSubtype"],
                "IHC Subtype": [""],
                "Segmentation Config": ["Biopsy"],
            }
        )

        result = validate_settings(
            df, cancer_subtype_name_map, cancer_subtypes, reversed_map
        )

        # Should default to Unknown
        assert result.iloc[0]["Cancer Subtype"] == "Unknown"
        # Warning should be called
        assert mock_warning.called

    @patch("mosaic.ui.utils.gr.Warning")
    def test_invalid_site_type_defaults_to_primary(
        self, mock_warning, mock_cancer_subtype_maps
    ):
        """Test invalid site type generates warning and defaults to Primary."""
        from mosaic.ui.utils import validate_settings

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )

        df = pd.DataFrame(
            {
                "Slide": ["test.svs"],
                "Site Type": ["InvalidSite"],
                "Sex": ["Unknown"],
                "Tissue Site": ["Unknown"],
                "Cancer Subtype": ["Unknown"],
                "IHC Subtype": [""],
                "Segmentation Config": ["Biopsy"],
            }
        )

        result = validate_settings(
            df, cancer_subtype_name_map, cancer_subtypes, reversed_map
        )

        assert result.iloc[0]["Site Type"] == "Primary"
        assert mock_warning.called


class TestAnalysisWorkflow:
    """Test analysis workflow with mocked analyze_slide."""

    @patch("mosaic.ui.app.analyze_slide")
    @patch("mosaic.ui.app.create_user_directory")
    def test_single_slide_analysis_no_model_cache(
        self,
        mock_create_dir,
        mock_analyze,
        sample_files_single,
        mock_analyze_slide_results,
        mock_cancer_subtype_maps,
        temp_output_dir,
    ):
        """Test single slide analysis doesn't load model cache."""
        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        # Setup mocks
        mock_create_dir.return_value = temp_output_dir
        mock_analyze.return_value = mock_analyze_slide_results

        # Generate settings DataFrame manually
        settings_df = pd.DataFrame(
            {
                "Slide": ["test_slide_1.svs"],
                "Site Type": ["Primary"],
                "Sex": ["Unknown"],
                "Tissue Site": ["Unknown"],
                "Cancer Subtype": ["Unknown"],
                "IHC Subtype": [""],
                "Segmentation Config": ["Biopsy"],
            }
        )

        # Call analyze_slides (generator)
        gen = analyze_slides(
            sample_files_single,
            settings_df,
            "Primary",
            "Unknown",
            "Unknown",
            "Unknown",
            "",
            "Biopsy",
            temp_output_dir,
        )

        # Consume generator
        results = list(gen)

        # Should yield at least once (intermediate + final)
        assert len(results) >= 1

        # analyze_slide should be called once
        assert mock_analyze.call_count == 1

        # Should be called with model_cache=None (single-slide mode)
        call_kwargs = mock_analyze.call_args[1]
        assert call_kwargs["model_cache"] is None

    @patch("mosaic.ui.app.load_all_models")
    @patch("mosaic.ui.app.analyze_slide")
    @patch("mosaic.ui.app.create_user_directory")
    def test_batch_analysis_loads_model_cache_once(
        self,
        mock_create_dir,
        mock_analyze,
        mock_load_models,
        sample_files_multiple,
        mock_analyze_slide_results,
        mock_model_cache,
        mock_cancer_subtype_maps,
        temp_output_dir,
    ):
        """Test batch analysis loads models once and reuses cache."""
        from PIL import Image

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        # Setup mocks - return new DataFrames on each call to avoid mutation issues
        def mock_analyze_side_effect(*args, **kwargs):
            mask = Image.new("RGB", (100, 100), color="red")
            aeon_results = pd.DataFrame(
                {"Cancer Subtype": ["LUAD"], "Confidence": [0.95]}
            )
            paladin_results = pd.DataFrame(
                {
                    "Cancer Subtype": ["LUAD", "LUAD", "LUAD"],
                    "Biomarker": ["TP53", "KRAS", "EGFR"],
                    "Score": [0.85, 0.72, 0.63],
                }
            )
            return (mask, aeon_results, paladin_results)

        mock_create_dir.return_value = temp_output_dir
        mock_load_models.return_value = mock_model_cache
        mock_analyze.side_effect = mock_analyze_side_effect

        # Generate settings DataFrame manually for 3 files
        settings_df = pd.DataFrame(
            {
                "Slide": ["test_slide_1.svs", "test_slide_2.svs", "test_slide_3.svs"],
                "Site Type": ["Primary", "Primary", "Primary"],
                "Sex": ["Unknown", "Unknown", "Unknown"],
                "Tissue Site": ["Unknown", "Unknown", "Unknown"],
                "Cancer Subtype": ["Unknown", "Unknown", "Unknown"],
                "IHC Subtype": ["", "", ""],
                "Segmentation Config": ["Biopsy", "Biopsy", "Biopsy"],
            }
        )

        # Call analyze_slides
        gen = analyze_slides(
            sample_files_multiple,
            settings_df,
            "Primary",
            "Unknown",
            "Unknown",
            "Unknown",
            "",
            "Biopsy",
            temp_output_dir,
        )

        # Consume generator
        results = list(gen)

        # load_all_models should be called once
        assert mock_load_models.call_count == 1

        # analyze_slide should be called 3 times (once per file)
        assert mock_analyze.call_count == 3

        # All calls should use the same model_cache
        for call in mock_analyze.call_args_list:
            assert call[1]["model_cache"] == mock_model_cache

        # cleanup should be called
        assert mock_model_cache.cleanup.called

    @patch("mosaic.ui.app.create_user_directory")
    def test_no_slides_raises_error(
        self, mock_create_dir, mock_cancer_subtype_maps, temp_output_dir
    ):
        """Test that no slides uploaded raises gr.Error."""
        import gradio as gr

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        mock_create_dir.return_value = temp_output_dir

        # Call with no slides
        gen = analyze_slides(
            None,
            None,
            "Primary",
            "Unknown",
            "Unknown",
            "Unknown",
            "",
            "Biopsy",
            temp_output_dir,
        )

        # Should raise gr.Error
        with pytest.raises(gr.Error):
            next(gen)

    @patch("mosaic.ui.app.create_user_directory")
    def test_settings_mismatch_raises_error(
        self,
        mock_create_dir,
        sample_files_multiple,
        sample_settings_df,
        mock_cancer_subtype_maps,
        temp_output_dir,
    ):
        """Test that settings count mismatch raises gr.Error."""
        import gradio as gr

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        mock_create_dir.return_value = temp_output_dir

        # sample_files_multiple has 3 files, sample_settings_df has 3 rows
        # Manually create mismatch by using only 2 files
        two_files = sample_files_multiple[:2]

        gen = analyze_slides(
            two_files,
            sample_settings_df,
            "Primary",
            "Unknown",
            "Unknown",
            "Unknown",
            "",
            "Biopsy",
            temp_output_dir,
        )

        # Should raise gr.Error about mismatch
        with pytest.raises(gr.Error):
            next(gen)

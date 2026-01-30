"""Tests for UI event handlers and state management.

This module tests complex event interactions in the Mosaic Gradio UI, including:
- Settings state management across events
- Generator behavior and incremental updates
- Error and warning display
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import inspect

from mosaic.ui.app import (
    analyze_slides,
    set_cancer_subtype_maps,
)
from mosaic.ui.utils import SETTINGS_COLUMNS, validate_settings, load_settings


class TestSettingsStateManagement:
    """Test settings state management across multiple events."""

    def test_csv_upload_replaces_settings(
        self, sample_csv_valid, mock_cancer_subtype_maps
    ):
        """Test CSV upload replaces existing settings."""
        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )

        # Load CSV
        loaded_df = load_settings(sample_csv_valid)
        validated_df = validate_settings(
            loaded_df, cancer_subtype_name_map, cancer_subtypes, reversed_map
        )

        # Verify new settings loaded
        assert len(validated_df) == 3
        assert validated_df.iloc[0]["Slide"] == "slide1.svs"
        assert validated_df.iloc[1]["Slide"] == "slide2.svs"


class TestGeneratorBehavior:
    """Test generator behavior for incremental updates."""

    @patch("mosaic.ui.app.analyze_slide")
    @patch("mosaic.ui.app.create_user_directory")
    def test_analyze_slides_is_generator(
        self,
        mock_create_dir,
        mock_analyze,
        sample_files_single,
        mock_analyze_slide_results,
        mock_cancer_subtype_maps,
        temp_output_dir,
    ):
        """Test analyze_slides returns a generator."""
        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        mock_create_dir.return_value = temp_output_dir
        mock_analyze.return_value = mock_analyze_slide_results

        settings_df = pd.DataFrame(
            {
                "Slide": ["test_slide_1.svs"],
                "Site Type": ["Primary"],
                "Sex": ["Male"],
                "Tissue Site": ["Unknown"],
                "Cancer Subtype": ["Unknown"],
                "IHC Subtype": [""],
                "Segmentation Config": ["Biopsy"],
            }
        )

        result = analyze_slides(
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

        # Verify it's a generator
        assert inspect.isgenerator(result)

    @patch("mosaic.ui.app.load_all_models")
    @patch("mosaic.ui.app.analyze_slide")
    @patch("mosaic.ui.app.create_user_directory")
    def test_single_final_yield_for_batch(
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
        """Test that analyze_slides yields only a single final result for batch."""
        from PIL import Image

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        mock_create_dir.return_value = temp_output_dir
        mock_load_models.return_value = mock_model_cache

        # Return fresh DataFrames on each call
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

        mock_analyze.side_effect = mock_analyze_side_effect

        settings_df = pd.DataFrame(
            {
                "Slide": ["test_slide_1.svs", "test_slide_2.svs", "test_slide_3.svs"],
                "Site Type": ["Primary", "Primary", "Primary"],
                "Sex": ["Male", "Female", "Male"],
                "Tissue Site": ["Unknown", "Unknown", "Unknown"],
                "Cancer Subtype": ["Unknown", "Unknown", "Unknown"],
                "IHC Subtype": ["", "", ""],
                "Segmentation Config": ["Biopsy", "Biopsy", "Biopsy"],
            }
        )

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

        # Generator should produce exactly one yield (final results only)
        results = list(gen)
        assert len(results) == 1

        final_yield = results[0]

        # Should be tuple with 9 elements
        assert len(final_yield) == 9

        # Analyze button should be re-enabled
        analyze_button = final_yield[7]
        assert hasattr(analyze_button, "interactive") and analyze_button.interactive

        # Clear button should be re-enabled
        clear_button = final_yield[8]
        assert hasattr(clear_button, "interactive") and clear_button.interactive

    @patch("mosaic.ui.app.load_all_models")
    @patch("mosaic.ui.app.analyze_slide")
    @patch("mosaic.ui.app.create_user_directory")
    def test_final_yield_has_complete_results(
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
        """Test final yield contains complete results."""
        from PIL import Image

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        mock_create_dir.return_value = temp_output_dir
        mock_load_models.return_value = mock_model_cache

        # Return fresh DataFrames on each call
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

        mock_analyze.side_effect = mock_analyze_side_effect

        settings_df = pd.DataFrame(
            {
                "Slide": ["test_slide_1.svs", "test_slide_2.svs", "test_slide_3.svs"],
                "Site Type": ["Primary", "Primary", "Primary"],
                "Sex": ["Male", "Female", "Male"],
                "Tissue Site": ["Unknown", "Unknown", "Unknown"],
                "Cancer Subtype": ["Unknown", "Unknown", "Unknown"],
                "IHC Subtype": ["", "", ""],
                "Segmentation Config": ["Biopsy", "Biopsy", "Biopsy"],
            }
        )

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

        # Consume generator to get final yield
        results = list(gen)
        final_yield = results[-1]

        # Final yield should have all results (9 elements with analyze and clear buttons)
        assert len(final_yield) == 9

        # First element is settings_input (should be visible for 3 slides)
        settings = final_yield[0]
        assert (
            hasattr(settings, "visible") and settings.visible
        )  # Visible for multiple slides

        # Second element is slide_masks
        slide_masks = final_yield[1]
        assert len(slide_masks) == 3  # All 3 slides

        # AEON download button should be visible on final yield (4th element, index 3)
        aeon_download = final_yield[3]
        assert hasattr(aeon_download, "visible") and aeon_download.visible

        # PALADIN download button should be visible on final yield (6th element, index 5)
        paladin_download = final_yield[5]
        assert hasattr(paladin_download, "visible") and paladin_download.visible

        # Eighth element should be analyze button (re-enabled after processing)
        analyze_button = final_yield[7]
        assert hasattr(analyze_button, "interactive") and analyze_button.interactive

        # Ninth element should be clear button (re-enabled after processing)
        clear_button = final_yield[8]
        assert hasattr(clear_button, "interactive") and clear_button.interactive


    @patch("mosaic.ui.app.load_all_models")
    @patch("mosaic.ui.app.analyze_slide")
    @patch("mosaic.ui.app.create_user_directory")
    def test_aeon_hidden_when_cancer_subtype_selected(
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
        """Test that Aeon table and download are hidden when cancer subtype is selected."""
        from PIL import Image

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        mock_create_dir.return_value = temp_output_dir
        mock_load_models.return_value = mock_model_cache

        def mock_analyze_side_effect(*args, **kwargs):
            mask = Image.new("RGB", (100, 100), color="red")
            # Synthetic Aeon result (user supplied cancer subtype)
            aeon_results = pd.DataFrame(
                {"Cancer Subtype": ["LUAD"], "Confidence": [1.0]}
            )
            paladin_results = pd.DataFrame(
                {
                    "Cancer Subtype": ["LUAD", "LUAD", "LUAD"],
                    "Biomarker": ["TP53", "KRAS", "EGFR"],
                    "Score": [0.85, 0.72, 0.63],
                }
            )
            return (mask, aeon_results, paladin_results)

        mock_analyze.side_effect = mock_analyze_side_effect

        # All slides have a known cancer subtype (not "Unknown")
        settings_df = pd.DataFrame(
            {
                "Slide": ["test_slide_1.svs", "test_slide_2.svs", "test_slide_3.svs"],
                "Site Type": ["Primary", "Primary", "Primary"],
                "Sex": ["Male", "Female", "Male"],
                "Tissue Site": ["Lung", "Lung", "Lung"],
                "Cancer Subtype": ["LUAD", "LUAD", "LUAD"],
                "IHC Subtype": ["", "", ""],
                "Segmentation Config": ["Biopsy", "Biopsy", "Biopsy"],
            }
        )

        gen = analyze_slides(
            sample_files_multiple,
            settings_df,
            "Primary",
            "Male",
            "Lung",
            "LUAD",
            "",
            "Biopsy",
            temp_output_dir,
        )

        results = list(gen)
        final_yield = results[-1]

        # Aeon table (index 2) should be hidden
        aeon_table = final_yield[2]
        assert hasattr(aeon_table, "visible") and not aeon_table.visible

        # Aeon download button (index 3) should be hidden
        aeon_download = final_yield[3]
        assert hasattr(aeon_download, "visible") and not aeon_download.visible

        # Paladin results should still be visible
        paladin_download = final_yield[5]
        assert hasattr(paladin_download, "visible") and paladin_download.visible


class TestErrorDisplay:
    """Test error and warning display behavior."""

    @patch("mosaic.ui.app.create_user_directory")
    def test_no_slides_raises_gr_error(
        self, mock_create_dir, mock_cancer_subtype_maps, temp_output_dir
    ):
        """Test that no slides raises gr.Error."""
        import gradio as gr

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        mock_create_dir.return_value = temp_output_dir

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

    @patch("mosaic.ui.utils.gr.Warning")
    def test_validation_warnings_shown(self, mock_warning, mock_cancer_subtype_maps):
        """Test validation warnings are displayed."""
        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )

        # Create DataFrame with multiple invalid values
        df = pd.DataFrame(
            {
                "Slide": ["test1.svs", "test2.svs"],
                "Site Type": ["InvalidSite", "Primary"],
                "Sex": ["Male", "InvalidSex"],
                "Tissue Site": ["Unknown", "Unknown"],
                "Cancer Subtype": ["InvalidSubtype", "Unknown"],
                "IHC Subtype": ["", ""],
                "Segmentation Config": ["Biopsy", "InvalidConfig"],
            }
        )

        result = validate_settings(
            df, cancer_subtype_name_map, cancer_subtypes, reversed_map
        )

        # Should have warning calls (at least 1 for the multiple invalid values)
        assert mock_warning.call_count >= 1

        # Verify defaults applied
        assert result.iloc[0]["Site Type"] == "Primary"  # Invalid → Primary
        assert result.iloc[0]["Cancer Subtype"] == "Unknown"  # Invalid → Unknown
        assert result.iloc[1]["Sex"] == ""  # Invalid → empty string
        assert result.iloc[1]["Segmentation Config"] == "Biopsy"  # Invalid → Biopsy

    @patch("mosaic.ui.app.create_user_directory")
    def test_settings_mismatch_raises_gr_error(
        self,
        mock_create_dir,
        sample_files_multiple,
        sample_settings_df,
        mock_cancer_subtype_maps,
        temp_output_dir,
    ):
        """Test settings/files count mismatch raises gr.Error."""
        import gradio as gr

        cancer_subtype_name_map, reversed_map, cancer_subtypes = (
            mock_cancer_subtype_maps
        )
        set_cancer_subtype_maps(cancer_subtype_name_map, reversed_map, cancer_subtypes)

        mock_create_dir.return_value = temp_output_dir

        # Create mismatch: 2 files but 3 settings rows
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

"""Tests for CLI execution modes and argument handling.

This module tests the Mosaic CLI, including:
- Argument parsing and routing
- Single-slide processing mode
- Batch CSV processing mode
- Model download behavior
- Output file generation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import pandas as pd


class TestArgumentParsing:
    """Test CLI argument parsing and mode routing."""

    @patch("mosaic.gradio_app.launch_gradio")
    @patch("mosaic.gradio_app.download_and_process_models")
    @patch("sys.argv", ["mosaic"])
    def test_no_arguments_launches_web_interface(self, mock_download, mock_launch):
        """Test no arguments routes to web interface mode."""
        mock_download.return_value = ({}, {}, [])

        from mosaic.gradio_app import main

        main()

        # Should call launch_gradio
        assert mock_launch.called
        assert mock_launch.call_count == 1

    @patch("mosaic.gradio_app.analyze_slide")
    @patch("mosaic.gradio_app.download_and_process_models")
    @patch(
        "sys.argv",
        ["mosaic", "--slide-path", "test.svs", "--output-dir", "out", "--sex", "Male"],
    )
    def test_slide_path_routes_to_single_mode(self, mock_download, mock_analyze):
        """Test --slide-path routes to single-slide mode."""
        mock_download.return_value = ({"Unknown": "UNK"}, {"UNK": "Unknown"}, [])
        mock_analyze.return_value = (None, None, None)

        from mosaic.gradio_app import main

        with patch("mosaic.gradio_app.Path.mkdir"):
            main()

        # Should call analyze_slide
        assert mock_analyze.called

    @patch("mosaic.gradio_app.load_all_models")
    @patch("mosaic.gradio_app.load_settings")
    @patch("mosaic.gradio_app.validate_settings")
    @patch("mosaic.gradio_app.analyze_slide")
    @patch("mosaic.gradio_app.download_and_process_models")
    @patch("sys.argv", ["mosaic", "--slide-csv", "test.csv", "--output-dir", "out"])
    def test_slide_csv_routes_to_batch_mode(
        self,
        mock_download,
        mock_analyze,
        mock_validate,
        mock_load_settings,
        mock_load_models,
    ):
        """Test --slide-csv routes to batch mode."""
        mock_download.return_value = ({"Unknown": "UNK"}, {"UNK": "Unknown"}, [])
        mock_load_settings.return_value = pd.DataFrame(
            {
                "Slide": ["test.svs"],
                "Site Type": ["Primary"],
                "Sex": ["Unknown"],
                "Tissue Site": ["Unknown"],
                "Cancer Subtype": ["Unknown"],
                # "IHC Subtype": [""],  # Not yet in use
                "Segmentation Config": ["Biopsy"],
            }
        )
        mock_validate.return_value = mock_load_settings.return_value
        mock_analyze.return_value = (None, None, None)

        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_load_models.return_value = mock_cache

        from mosaic.gradio_app import main

        with patch("mosaic.gradio_app.Path.mkdir"):
            main()

        # Should call load_all_models (batch mode)
        assert mock_load_models.called


class TestSingleSlideMode:
    """Test single-slide processing mode."""

    @patch("mosaic.gradio_app.Path.mkdir")
    @patch("mosaic.gradio_app.analyze_slide")
    @patch("mosaic.gradio_app.download_and_process_models")
    def test_analyze_slide_called_with_correct_params(
        self, mock_download, mock_analyze, mock_mkdir, cli_args_single
    ):
        """Test analyze_slide called with correct parameters in single mode."""
        mock_download.return_value = ({"Unknown": "UNK"}, {"UNK": "Unknown"}, [])
        mock_analyze.return_value = (None, None, None)

        # Patch ArgumentParser to return our test args
        with patch(
            "mosaic.gradio_app.ArgumentParser.parse_args", return_value=cli_args_single
        ):
            from mosaic.gradio_app import main

            main()

        # Verify analyze_slide was called
        assert mock_analyze.called
        call_args = mock_analyze.call_args[0]  # Positional args

        # Check key parameters (analyze_slide uses positional args)
        assert call_args[0] == cli_args_single.slide_path  # slide_path
        assert call_args[1] == cli_args_single.segmentation_config  # seg_config
        assert call_args[2] == cli_args_single.site_type  # site_type

    @patch("PIL.Image.Image.save")
    @patch("mosaic.gradio_app.Path.mkdir")
    @patch("mosaic.gradio_app.analyze_slide")
    @patch("mosaic.gradio_app.download_and_process_models")
    def test_output_files_saved_correctly(
        self,
        mock_download,
        mock_analyze,
        mock_mkdir,
        mock_save,
        cli_args_single,
        mock_analyze_slide_results,
    ):
        """Test output files are saved with correct names."""
        from PIL import Image

        mock_download.return_value = ({"Unknown": "UNK"}, {"UNK": "Unknown"}, [])

        # Mock analyze_slide to return results
        mask, aeon_results, paladin_results = mock_analyze_slide_results
        mock_analyze.return_value = (mask, aeon_results, paladin_results)

        # Patch ArgumentParser
        with patch(
            "mosaic.gradio_app.ArgumentParser.parse_args", return_value=cli_args_single
        ):
            # Patch DataFrame.to_csv to avoid actual file writes
            with patch("pandas.DataFrame.to_csv"):
                from mosaic.gradio_app import main

                main()

        # Verify save was called for mask
        assert mock_save.called


class TestBatchCsvMode:
    """Test batch CSV processing mode."""

    @patch("mosaic.gradio_app.Path.mkdir")
    @patch("mosaic.gradio_app.load_all_models")
    @patch("mosaic.gradio_app.analyze_slide")
    @patch("mosaic.gradio_app.validate_settings")
    @patch("mosaic.gradio_app.load_settings")
    @patch("mosaic.gradio_app.download_and_process_models")
    def test_load_all_models_called_once(
        self,
        mock_download,
        mock_load_settings,
        mock_validate,
        mock_analyze,
        mock_load_models,
        mock_mkdir,
        cli_args_batch,
        sample_settings_df,
        mock_analyze_slide_results,
    ):
        """Test load_all_models called once in batch mode."""
        from PIL import Image

        mock_download.return_value = ({"Unknown": "UNK"}, {"UNK": "Unknown"}, [])
        mock_load_settings.return_value = sample_settings_df
        mock_validate.return_value = sample_settings_df

        # Return fresh DataFrames on each call to avoid mutation
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

        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_load_models.return_value = mock_cache

        with patch(
            "mosaic.gradio_app.ArgumentParser.parse_args", return_value=cli_args_batch
        ):
            with patch("pandas.DataFrame.to_csv"):
                with patch("PIL.Image.Image.save"):
                    from mosaic.gradio_app import main

                    main()

        # load_all_models should be called exactly once
        assert mock_load_models.call_count == 1

        # analyze_slide should be called for each slide (3 times)
        assert mock_analyze.call_count == 3

        # All analyze_slide calls should receive the model_cache
        for call in mock_analyze.call_args_list:
            assert call[1]["model_cache"] == mock_cache

        # cleanup should be called
        assert mock_cache.cleanup.called

    @patch("mosaic.gradio_app.Path.mkdir")
    @patch("mosaic.gradio_app.load_all_models")
    @patch("mosaic.gradio_app.analyze_slide")
    @patch("mosaic.gradio_app.validate_settings")
    @patch("mosaic.gradio_app.load_settings")
    @patch("mosaic.gradio_app.download_and_process_models")
    def test_combined_outputs_generated(
        self,
        mock_download,
        mock_load_settings,
        mock_validate,
        mock_analyze,
        mock_load_models,
        mock_mkdir,
        cli_args_batch,
        sample_settings_df,
        mock_analyze_slide_results,
    ):
        """Test combined output files are generated in batch mode."""
        from PIL import Image

        mock_download.return_value = (
            {"Unknown": "UNK", "Lung Adenocarcinoma (LUAD)": "LUAD"},
            {"UNK": "Unknown", "LUAD": "Lung Adenocarcinoma (LUAD)"},
            ["LUAD"],
        )
        mock_load_settings.return_value = sample_settings_df
        mock_validate.return_value = sample_settings_df

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

        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_load_models.return_value = mock_cache

        csv_calls = []

        def track_csv_write(path, *args, **kwargs):
            """Track CSV file writes."""
            csv_calls.append(str(path))

        with patch(
            "mosaic.gradio_app.ArgumentParser.parse_args", return_value=cli_args_batch
        ):
            with patch("pandas.DataFrame.to_csv", side_effect=track_csv_write):
                with patch("PIL.Image.Image.save"):
                    from mosaic.gradio_app import main

                    main()

        # Should have combined files
        combined_files = [c for c in csv_calls if "combined" in c]
        assert len(combined_files) >= 2  # combined_aeon and combined_paladin

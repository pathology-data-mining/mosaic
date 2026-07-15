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
from argparse import Namespace


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

    @patch("mosaic.cli.runners.analyze_slide")
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

        with patch("mosaic.cli.runners.Path.mkdir"):
            main()

        # Should call analyze_slide
        assert mock_analyze.called

    @patch("mosaic.cli.runners.load_all_models")
    @patch("mosaic.cli.runners.load_settings")
    @patch("mosaic.cli.runners.normalize_settings")
    @patch("mosaic.cli.runners.analyze_slide")
    @patch("mosaic.gradio_app.download_and_process_models")
    @patch("sys.argv", ["mosaic", "--slide-csv", "test.csv", "--output-dir", "out"])
    def test_slide_csv_routes_to_batch_mode(
        self,
        mock_download,
        mock_analyze,
        mock_normalize,
        mock_load_settings,
        mock_load_models,
    ):
        """Test --slide-csv routes to batch mode."""
        mock_download.return_value = ({"Unknown": "UNK"}, {"UNK": "Unknown"}, [])
        settings_df = pd.DataFrame(
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
        mock_load_settings.return_value = settings_df
        mock_normalize.return_value = (settings_df, [])
        mock_analyze.return_value = (None, None, None)

        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_load_models.return_value = mock_cache

        from mosaic.gradio_app import main

        with patch("mosaic.cli.runners.Path.mkdir"):
            main()

        # Should call load_all_models (batch mode)
        assert mock_load_models.called


class TestSingleSlideMode:
    """Test single-slide processing mode."""

    @patch("mosaic.cli.runners.Path.mkdir")
    @patch("mosaic.cli.runners.analyze_slide")
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
    @patch("mosaic.cli.runners.Path.mkdir")
    @patch("mosaic.cli.runners.analyze_slide")
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

    @patch("mosaic.gradio_app.download_and_process_models")
    def test_known_cancer_subtype_does_not_require_sex(self, mock_download, cli_args_single):
        """Known subtype should bypass the single-slide sex requirement."""
        args = Namespace(**vars(cli_args_single))
        args.sex = None
        args.cancer_subtype = "Lung Adenocarcinoma (LUAD)"

        mock_download.return_value = (
            {"Unknown": "UNK", "Lung Adenocarcinoma (LUAD)": "LUAD"},
            {"UNK": "Unknown", "LUAD": "Lung Adenocarcinoma (LUAD)"},
            ["LUAD"],
        )

        with patch("mosaic.cli.runners.Path.mkdir"), patch(
            "mosaic.cli.runners.analyze_slide", return_value=(None, None, None)
        ) as mock_analyze:
            with patch(
                "mosaic.gradio_app.ArgumentParser.parse_args", return_value=args
            ):
                from mosaic.gradio_app import main

                main()

        assert mock_analyze.called

    def test_unknown_cancer_subtype_requires_sex(self, cli_args_single):
        """Unknown subtype should still require sex for single-slide Aeon."""
        from mosaic.cli.runners import run_single_slide

        args = Namespace(**vars(cli_args_single))
        args.sex = None
        args.cancer_subtype = "Unknown"

        with pytest.raises(
            ValueError,
            match="Sex is required for single-slide processing when Cancer Subtype is Unknown",
        ):
            run_single_slide(
                args,
                {"Unknown": "UNK"},
                [],
                {"UNK": "Unknown"},
            )


class TestBatchCsvMode:
    """Test batch CSV processing mode."""

    @patch("mosaic.cli.runners.Path.mkdir")
    @patch("mosaic.cli.runners.load_all_models")
    @patch("mosaic.cli.runners.analyze_slide")
    @patch("mosaic.cli.runners.normalize_settings")
    @patch("mosaic.cli.runners.load_settings")
    @patch("mosaic.gradio_app.download_and_process_models")
    def test_load_all_models_called_once(
        self,
        mock_download,
        mock_load_settings,
        mock_normalize,
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
        mock_normalize.return_value = (sample_settings_df, [])

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
                    "Prob. Present": [0.85, 0.72, 0.63],
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

    def test_batch_allows_missing_sex_when_cancer_subtype_is_known(self, cli_args_batch):
        """Batch CSV should accept missing sex for known subtype rows."""
        from mosaic.cli.runners import run_batch

        settings_df = pd.DataFrame(
            {
                "Slide": ["test.svs"],
                "Site Type": ["Primary"],
                "Sex": [""],
                "Tissue Site": ["Unknown"],
                "Cancer Subtype": ["Lung Adenocarcinoma (LUAD)"],
                "Segmentation Config": ["Biopsy"],
                "Sample ID": ["test"],
            }
        )

        mock_cache = Mock()
        mock_cache.cleanup = Mock()

        with patch("mosaic.cli.runners.Path.mkdir"), patch(
            "mosaic.cli.runners.load_settings", return_value=settings_df
        ), patch(
            "mosaic.cli.runners.normalize_settings", return_value=(settings_df, [])
        ), patch(
            "mosaic.cli.runners.load_all_models", return_value=mock_cache
        ), patch(
            "mosaic.cli.runners.analyze_slide", return_value=(None, None, None)
        ):
            run_batch(
                cli_args_batch,
                {"Unknown": "UNK", "Lung Adenocarcinoma (LUAD)": "LUAD"},
                ["LUAD"],
                {"UNK": "Unknown", "LUAD": "Lung Adenocarcinoma (LUAD)"},
            )

    def test_batch_requires_sex_when_cancer_subtype_is_unknown(self, cli_args_batch):
        """Batch CSV should reject missing sex when Aeon inference is required."""
        from mosaic.cli.runners import run_batch

        settings_df = pd.DataFrame(
            {
                "Slide": ["test.svs"],
                "Site Type": ["Primary"],
                "Sex": [""],
                "Tissue Site": ["Unknown"],
                "Cancer Subtype": ["Unknown"],
                "Segmentation Config": ["Biopsy"],
                "Sample ID": ["test"],
            }
        )

        with patch("mosaic.cli.runners.Path.mkdir"), patch(
            "mosaic.cli.runners.load_settings", return_value=settings_df
        ), patch(
            "mosaic.cli.runners.normalize_settings", return_value=(settings_df, [])
        ):
            with pytest.raises(
                ValueError,
                match=r"Sex is required for slide 'test\.svs' when Cancer Subtype is Unknown",
            ):
                run_batch(
                    cli_args_batch,
                    {"Unknown": "UNK"},
                    [],
                    {"UNK": "Unknown"},
                )

    @patch("mosaic.cli.runners.Path.mkdir")
    @patch("mosaic.cli.runners.load_all_models")
    @patch("mosaic.cli.runners.analyze_slide")
    @patch("mosaic.cli.runners.normalize_settings")
    @patch("mosaic.cli.runners.load_settings")
    @patch("mosaic.gradio_app.download_and_process_models")
    def test_combined_outputs_generated(
        self,
        mock_download,
        mock_load_settings,
        mock_normalize,
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
        mock_normalize.return_value = (sample_settings_df, [])

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
                    "Prob. Present": [0.85, 0.72, 0.63],
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

    @patch("mosaic.cli.runners.Path.mkdir")
    @patch("mosaic.cli.runners.load_all_models")
    @patch("mosaic.cli.runners.analyze_slide")
    @patch("mosaic.cli.runners.normalize_settings")
    @patch("mosaic.cli.runners.load_settings")
    @patch("mosaic.gradio_app.download_and_process_models")
    def test_batch_mode_logs_normalize_settings_warnings(
        self,
        mock_download,
        mock_load_settings,
        mock_normalize,
        mock_analyze,
        mock_load_models,
        mock_mkdir,
        cli_args_batch,
        sample_settings_df,
    ):
        """normalize_settings warnings are logged (not silently dropped) in CLI mode.

        The CLI path calls normalize_settings directly (no Gradio), so warnings
        must be emitted via logger rather than gr.Warning.  This test verifies
        the batch path processes all slides even when warnings are present.
        """
        from PIL import Image

        mock_download.return_value = ({"Unknown": "UNK"}, {"UNK": "Unknown"}, [])
        mock_load_settings.return_value = sample_settings_df
        # normalize_settings always returns a (df, warnings) tuple — never a sentinel.
        mock_normalize.return_value = (sample_settings_df, ["Some validation warning"])

        def mock_analyze_side_effect(*args, **kwargs):
            mask = Image.new("RGB", (100, 100), color="red")
            aeon_results = pd.DataFrame(
                {"Cancer Subtype": ["LUAD"], "Confidence": [0.95]}
            )
            paladin_results = pd.DataFrame(
                {
                    "Cancer Subtype": ["LUAD", "LUAD"],
                    "Biomarker": ["TP53", "KRAS"],
                    "Prob. Present": [0.85, 0.72],
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

                    # Must not raise even when warnings are returned
                    main()

        # All slides from the normalized DataFrame should still be processed.
        assert mock_analyze.call_count == len(sample_settings_df)

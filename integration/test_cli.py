"""CLI integration tests.

These tests call mosaic.gradio_app.main() directly in-process with sys.argv
patched, while mocking only the heavy ML operations (model download, analyze_slide,
load_all_models).  Everything else — argument parsing, settings validation, output
file creation — runs for real.
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAKE_CANCER_SUBTYPE_MAP = {
    "Unknown": "UNK",
    "Lung Adenocarcinoma (LUAD)": "LUAD",
}
FAKE_REVERSED_MAP = {v: k for k, v in FAKE_CANCER_SUBTYPE_MAP.items()}
FAKE_CANCER_SUBTYPES = ["LUAD"]


@pytest.fixture()
def fake_slide(tmp_path) -> Path:
    """A tiny 256×256 TIFF saved with an .svs extension."""
    path = tmp_path / "test_slide.svs"
    Image.new("RGB", (256, 256), color=(200, 180, 160)).save(str(path), format="TIFF")
    return path


@pytest.fixture()
def fake_mask() -> Image.Image:
    return Image.new("RGB", (256, 256), color=(0, 128, 0))


@pytest.fixture()
def fake_aeon_df() -> pd.DataFrame:
    # analyze_slide returns aeon_results with "Cancer Subtype" as the index
    # and "Confidence" as the only column (matching analysis.py set_index call)
    df = pd.DataFrame({"Cancer Subtype": ["LUAD", "LUSC"], "Confidence": [0.85, 0.15]})
    df.set_index("Cancer Subtype", inplace=True)
    return df


@pytest.fixture()
def fake_paladin_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Cancer Subtype": ["LUAD"], "Biomarker": ["TP53"], "Score": [0.85]},
    )


@pytest.fixture()
def output_dir(tmp_path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


def _make_download_metadata_mock():
    """Return a side-effectful mock for download_metadata that also sets pipeline maps."""
    from mosaic.ui.pipeline import set_cancer_subtype_maps

    def _mock(*_args, **_kwargs):
        set_cancer_subtype_maps(
            FAKE_CANCER_SUBTYPE_MAP,
            FAKE_REVERSED_MAP,
            FAKE_CANCER_SUBTYPES,
        )
        return FAKE_CANCER_SUBTYPE_MAP, FAKE_REVERSED_MAP, FAKE_CANCER_SUBTYPES

    return _mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMMON_PATCHES = [
    # Prevent any real model/metadata downloads
    patch("mosaic.gradio_app.download_and_process_models"),
    patch("mosaic.gradio_app.download_metadata"),
    # Never hit HuggingFace
    patch("mosaic.gradio_app.snapshot_download", return_value="/fake/cache"),
    # Prevent oncotree HTTP calls in validate_settings
    patch(
        "mosaic.gradio_app.get_oncotree_code_name",
        side_effect=lambda code: f"Mock {code}",
    ),
]


def _apply_patches(patches):
    """Context manager that enters all patches and returns their mocks."""
    from contextlib import ExitStack

    stack = ExitStack()
    mocks = [stack.enter_context(p) for p in patches]
    return stack, mocks


# ---------------------------------------------------------------------------
# Single-slide CLI tests
# ---------------------------------------------------------------------------


class TestSingleSlideCLI:
    """Integration tests for `mosaic --slide-path <file> --output-dir <dir>`."""

    def _run_single(self, fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        """Call main() with single-slide args; return the output dir path."""
        mock_download = _make_download_metadata_mock()

        argv = [
            "mosaic",
            "--slide-path", str(fake_slide),
            "--output-dir", str(output_dir),
            "--sex", "Male",
            "--site-type", "Primary",
            "--tissue-site", "Lung",
            "--cancer-subtype", "Unknown",
            "--segmentation-config", "Biopsy",
            "--skip-model-download",
        ]

        with patch.object(sys, "argv", argv), \
             patch("mosaic.gradio_app.analyze_slide",
                   return_value=(fake_mask, fake_aeon_df, fake_paladin_df)), \
             patch("mosaic.gradio_app.get_oncotree_code_name",
                   side_effect=lambda code: f"Mock {code}"), \
             patch("mosaic.gradio_app.download_and_process_models",
                   side_effect=mock_download), \
             patch("mosaic.gradio_app.download_metadata",
                   side_effect=mock_download), \
             patch("mosaic.data_directory.get_data_directory",
                   return_value=str(Path(__file__).parent.parent / "data")), \
             patch("mosaic.gradio_app.set_data_directory"):
            from mosaic.gradio_app import main
            main()

        return output_dir

    def test_mask_file_created(self, fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        """A PNG mask file is written to the output directory."""
        self._run_single(fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df)
        mask_path = output_dir / "test_slide_mask.png"
        assert mask_path.exists(), f"Expected mask at {mask_path}"

    def test_aeon_csv_created(self, fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        """An Aeon results CSV is written to the output directory."""
        self._run_single(fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df)
        aeon_path = output_dir / "test_slide_aeon_results.csv"
        assert aeon_path.exists(), f"Expected Aeon CSV at {aeon_path}"

    def test_aeon_csv_contains_cancer_subtype(self, fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        """Aeon results CSV contains the mocked cancer subtype."""
        self._run_single(fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df)
        df = pd.read_csv(output_dir / "test_slide_aeon_results.csv")
        assert "Cancer Subtype" in df.columns
        assert "LUAD" in df["Cancer Subtype"].values
    def test_paladin_csv_created(self, fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        """A Paladin results CSV is written to the output directory."""
        self._run_single(fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df)
        paladin_path = output_dir / "test_slide_paladin_results.csv"
        assert paladin_path.exists(), f"Expected Paladin CSV at {paladin_path}"

    def test_paladin_csv_contains_biomarkers(self, fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        """Paladin results CSV contains biomarker rows."""
        self._run_single(fake_slide, output_dir, fake_mask, fake_aeon_df, fake_paladin_df)
        df = pd.read_csv(output_dir / "test_slide_paladin_results.csv")
        assert len(df) >= 1
        assert "Biomarker" in df.columns

    def test_missing_sex_raises(self, fake_slide, output_dir):
        """CLI raises ValueError when --sex is omitted for single-slide mode."""
        mock_download = _make_download_metadata_mock()
        argv = [
            "mosaic",
            "--slide-path", str(fake_slide),
            "--output-dir", str(output_dir),
            "--skip-model-download",
        ]
        with patch.object(sys, "argv", argv), \
             patch("mosaic.gradio_app.download_and_process_models",
                   side_effect=mock_download), \
             patch("mosaic.gradio_app.download_metadata",
                   side_effect=mock_download), \
             patch("mosaic.data_directory.get_data_directory",
                   return_value=str(Path(__file__).parent.parent / "data")), \
             patch("mosaic.gradio_app.set_data_directory"):
            with pytest.raises(ValueError, match="--sex"):
                from mosaic.gradio_app import main
                main()


# ---------------------------------------------------------------------------
# Batch CSV CLI tests
# ---------------------------------------------------------------------------


class TestBatchCLI:
    """Integration tests for `mosaic --slide-csv <file> --output-dir <dir>`."""

    @pytest.fixture()
    def batch_csv(self, fake_slide, tmp_path) -> Path:
        """A minimal batch CSV referencing our fake slide by name only."""
        csv_path = tmp_path / "batch_settings.csv"
        # Use just the filename (not absolute path) so output files land in output_dir
        csv_path.write_text(
            textwrap.dedent(f"""\
                Slide,Site Type,Sex,Tissue Site,Cancer Subtype,Segmentation Config
                {fake_slide.name},Primary,Male,Lung,Unknown,Biopsy
            """)
        )
        return csv_path

    def _run_batch(self, batch_csv, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        mock_download = _make_download_metadata_mock()
        mock_model_cache = MagicMock()

        argv = [
            "mosaic",
            "--slide-csv", str(batch_csv),
            "--output-dir", str(output_dir),
            "--skip-model-download",
        ]

        with patch.object(sys, "argv", argv), \
             patch("mosaic.gradio_app.analyze_slide",
                   return_value=(fake_mask, fake_aeon_df, fake_paladin_df)), \
             patch("mosaic.gradio_app.load_all_models",
                   return_value=mock_model_cache), \
             patch("mosaic.gradio_app.get_oncotree_code_name",
                   side_effect=lambda code: f"Mock {code}"), \
             patch("mosaic.gradio_app.download_and_process_models",
                   side_effect=mock_download), \
             patch("mosaic.gradio_app.download_metadata",
                   side_effect=mock_download), \
             patch("mosaic.data_directory.get_data_directory",
                   return_value=str(Path(__file__).parent.parent / "data")), \
             patch("mosaic.gradio_app.set_data_directory"):
            from mosaic.gradio_app import main
            main()

        return output_dir

    def test_mask_file_created(self, batch_csv, output_dir, fake_mask, fake_aeon_df, fake_paladin_df, fake_slide):
        """A PNG mask is created for the slide in the batch."""
        self._run_batch(batch_csv, output_dir, fake_mask, fake_aeon_df, fake_paladin_df)
        # Batch code uses Path(slide_name).stem for the file stem
        slide_stem = fake_slide.stem
        mask_path = output_dir / f"{slide_stem}_mask.png"
        assert mask_path.exists(), f"Expected mask at {mask_path}; found: {list(output_dir.iterdir())}"

    def test_aeon_csv_created(self, batch_csv, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        """An Aeon CSV is written per-slide and a combined CSV is written."""
        self._run_batch(batch_csv, output_dir, fake_mask, fake_aeon_df, fake_paladin_df)
        combined = output_dir / "combined_aeon_results.csv"
        assert combined.exists(), f"Expected combined Aeon CSV; found: {list(output_dir.iterdir())}"

    def test_paladin_csv_created(self, batch_csv, output_dir, fake_mask, fake_aeon_df, fake_paladin_df, fake_slide):
        """A Paladin CSV is written per-slide."""
        self._run_batch(batch_csv, output_dir, fake_mask, fake_aeon_df, fake_paladin_df)
        slide_stem = fake_slide.stem
        paladin_path = output_dir / f"{slide_stem}_paladin_results.csv"
        assert paladin_path.exists(), f"Expected Paladin CSV; found: {list(output_dir.iterdir())}"

    def test_load_all_models_called_once(self, batch_csv, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        """Models are loaded exactly once for the entire batch."""
        mock_download = _make_download_metadata_mock()
        mock_model_cache = MagicMock()
        mock_load = MagicMock(return_value=mock_model_cache)

        argv = [
            "mosaic",
            "--slide-csv", str(batch_csv),
            "--output-dir", str(output_dir),
            "--skip-model-download",
        ]

        with patch.object(sys, "argv", argv), \
             patch("mosaic.gradio_app.analyze_slide",
                   return_value=(fake_mask, fake_aeon_df, fake_paladin_df)), \
             patch("mosaic.gradio_app.load_all_models", mock_load), \
             patch("mosaic.gradio_app.get_oncotree_code_name",
                   side_effect=lambda code: f"Mock {code}"), \
             patch("mosaic.gradio_app.download_and_process_models",
                   side_effect=mock_download), \
             patch("mosaic.gradio_app.download_metadata",
                   side_effect=mock_download), \
             patch("mosaic.data_directory.get_data_directory",
                   return_value=str(Path(__file__).parent.parent / "data")), \
             patch("mosaic.gradio_app.set_data_directory"):
            from mosaic.gradio_app import main
            main()

        mock_load.assert_called_once()

    def test_model_cache_cleanup_called(self, batch_csv, output_dir, fake_mask, fake_aeon_df, fake_paladin_df):
        """Model cache cleanup is called after batch processing."""
        mock_download = _make_download_metadata_mock()
        mock_model_cache = MagicMock()

        argv = [
            "mosaic",
            "--slide-csv", str(batch_csv),
            "--output-dir", str(output_dir),
            "--skip-model-download",
        ]

        with patch.object(sys, "argv", argv), \
             patch("mosaic.gradio_app.analyze_slide",
                   return_value=(fake_mask, fake_aeon_df, fake_paladin_df)), \
             patch("mosaic.gradio_app.load_all_models",
                   return_value=mock_model_cache), \
             patch("mosaic.gradio_app.get_oncotree_code_name",
                   side_effect=lambda code: f"Mock {code}"), \
             patch("mosaic.gradio_app.download_and_process_models",
                   side_effect=mock_download), \
             patch("mosaic.gradio_app.download_metadata",
                   side_effect=mock_download), \
             patch("mosaic.data_directory.get_data_directory",
                   return_value=str(Path(__file__).parent.parent / "data")), \
             patch("mosaic.gradio_app.set_data_directory"):
            from mosaic.gradio_app import main
            main()

        mock_model_cache.cleanup.assert_called_once()

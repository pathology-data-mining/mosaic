"""End-to-end CLI tests for the Mosaic command-line interface.

These tests run the full analysis pipeline via the CLI using real models and a
real slide. They are intentionally excluded from CI and must be run manually on
a system with a CUDA-capable GPU and pre-downloaded models.

Prerequisites:
    1. CUDA GPU available
    2. Model weights pre-downloaded (run ``mosaic --download-models-only`` first)
    3. tests/testdata/948176.svs present

Usage:
    pytest tests/e2e/ --e2e -v
    pytest tests/e2e/test_cli_e2e.py --e2e -v -s   # with stdout

The tests cover:
    - Single-slide mode (--slide-path): output files, Aeon/Paladin CSV content
    - Batch CSV mode (--slide-csv): per-sample and combined output files
    - Multi-slide batch mode: same slide listed twice → feature aggregation
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_SLIDE = Path(__file__).parent.parent / "testdata" / "948176.svs"

# Ground-truth metadata for TEST_SLIDE (948176.svs — HGSOC omentum metastasis).
TEST_SLIDE_METADATA = {
    "sex": "Female",
    "site_type": "Metastatic",
    "tissue_site": "Omentum",
}

# Expected Aeon top-1 cancer type (oncotree code) for TEST_SLIDE.
TEST_SLIDE_EXPECTED_CANCER_TYPE = "HGSOC"

# Maximum seconds allowed for a single CLI run (real models are slow).
CLI_TIMEOUT = 900  # 15 minutes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(*args: str, timeout: int = CLI_TIMEOUT) -> subprocess.CompletedProcess:
    """Invoke the Mosaic CLI and return the completed process."""
    cmd = [sys.executable, "-m", "mosaic.gradio_app", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_aeon_csv_sensible(
    csv_path: Path, expected_cancer_type: str | None = None
) -> pd.DataFrame:
    """Assert an Aeon results CSV is structurally sound and optionally check top-3.

    Checks:
    - File exists and has at least 5 rows (model outputs 100+ cancer types)
    - Required columns present
    - All confidence values are floats in [0, 1]
    - Top confidence is meaningfully non-zero (> 0.05)
    - If *expected_cancer_type* given, it appears (as substring) in top-3 rows

    Returns the loaded DataFrame for further inspection.
    """
    assert csv_path.exists(), f"Aeon results CSV not found: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) >= 5, f"Aeon CSV has only {len(df)} rows; expected ≥ 5"
    assert (
        "Cancer Subtype" in df.columns
    ), f"Missing 'Cancer Subtype' column: {df.columns.tolist()}"

    # Find the confidence column (may be named 'Confidence' or similar).
    conf_cols = [
        c for c in df.columns if "confidence" in c.lower() or "prob" in c.lower()
    ]
    assert conf_cols, f"No confidence column found in Aeon CSV: {df.columns.tolist()}"
    conf_col = conf_cols[0]

    confidences = df[conf_col].dropna().tolist()
    assert all(
        0.0 <= c <= 1.0 for c in confidences
    ), f"Aeon confidence values out of [0, 1]: {confidences[:10]}"
    assert max(confidences) > 0.05, (
        f"Max Aeon confidence {max(confidences):.4f} is suspiciously low; "
        "the model may not have run correctly"
    )

    if expected_cancer_type:
        top3_subtypes = df["Cancer Subtype"].head(3).tolist()
        matches = [st for st in top3_subtypes if expected_cancer_type in str(st)]
        assert matches, (
            f"Expected cancer type code '{expected_cancer_type}' not found in "
            f"top-3 Aeon predictions: {top3_subtypes}"
        )

    return df


def _assert_paladin_csv_sensible(csv_path: Path) -> pd.DataFrame:
    """Assert a Paladin results CSV is structurally sound.

    Checks:
    - File exists and has at least 1 row
    - A 'Biomarker' column exists with non-empty names
    - A probability column exists with values in [0, 1]

    Returns the loaded DataFrame for further inspection.
    """
    assert csv_path.exists(), f"Paladin results CSV not found: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) > 0, f"Paladin CSV is empty: {csv_path}"
    assert (
        "Biomarker" in df.columns
    ), f"Missing 'Biomarker' column: {df.columns.tolist()}"

    prob_cols = [c for c in df.columns if "prob" in c.lower()]
    assert (
        prob_cols
    ), f"No probability column found in Paladin CSV: {df.columns.tolist()}"
    prob_col = prob_cols[0]

    probs = df[prob_col].dropna().tolist()
    assert all(
        0.0 <= p <= 1.0 for p in probs
    ), f"Paladin probability values out of [0, 1]: {probs[:10]}"
    assert df["Biomarker"].notna().any(), "All Paladin biomarker names are empty"

    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _require_test_slide():
    """Skip the entire module if the test slide is missing."""
    if not TEST_SLIDE.exists():
        pytest.skip(
            f"Test slide not found: {TEST_SLIDE}. "
            "Download it and place it in tests/testdata/ to run CLI e2e tests."
        )


# ---------------------------------------------------------------------------
# Single-slide mode tests
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_require_test_slide")
class TestCLISingleSlide:
    """End-to-end tests for single-slide CLI mode (--slide-path).

    The CLI is invoked once per class (via the ``cli_output`` fixture) and all
    tests in this class read from that shared output directory.
    """

    @pytest.fixture(scope="class")
    def cli_output(self, tmp_path_factory):
        """Run the single-slide CLI once and return the output directory."""
        output_dir = tmp_path_factory.mktemp("cli_single")
        result = _run_cli(
            "--slide-path",
            str(TEST_SLIDE),
            "--sex",
            TEST_SLIDE_METADATA["sex"],
            "--site-type",
            TEST_SLIDE_METADATA["site_type"],
            "--tissue-site",
            TEST_SLIDE_METADATA["tissue_site"],
            "--output-dir",
            str(output_dir),
            "--skip-model-download",
        )
        assert result.returncode == 0, (
            f"Single-slide CLI exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
        return output_dir

    # -- Output file existence --------------------------------------------------

    def test_mask_png_created(self, cli_output):
        """Tissue mask PNG is written next to the slide stem."""
        mask = cli_output / f"{TEST_SLIDE.stem}_mask.png"
        assert mask.exists(), f"Expected mask at {mask}"
        assert mask.stat().st_size > 0, "Mask file is empty"

    def test_aeon_csv_created(self, cli_output):
        """Aeon results CSV is written for the slide."""
        aeon_csv = cli_output / f"{TEST_SLIDE.stem}_aeon_results.csv"
        assert aeon_csv.exists(), f"Expected Aeon CSV at {aeon_csv}"

    def test_paladin_csv_created(self, cli_output):
        """Paladin results CSV is written for the slide."""
        paladin_csv = cli_output / f"{TEST_SLIDE.stem}_paladin_results.csv"
        assert paladin_csv.exists(), f"Expected Paladin CSV at {paladin_csv}"

    # -- Content correctness ---------------------------------------------------

    def test_aeon_results_sensible(self, cli_output):
        """Aeon CSV contains ≥ 5 predictions with valid confidences."""
        _assert_aeon_csv_sensible(cli_output / f"{TEST_SLIDE.stem}_aeon_results.csv")

    def test_aeon_top_cancer_type(self, cli_output):
        """Expected cancer type appears in the top-3 Aeon predictions."""
        _assert_aeon_csv_sensible(
            cli_output / f"{TEST_SLIDE.stem}_aeon_results.csv",
            expected_cancer_type=TEST_SLIDE_EXPECTED_CANCER_TYPE,
        )

    def test_paladin_results_sensible(self, cli_output):
        """Paladin CSV contains biomarkers with valid probabilities."""
        _assert_paladin_csv_sensible(
            cli_output / f"{TEST_SLIDE.stem}_paladin_results.csv"
        )

    # -- Error-path validation (fast, no real inference) -----------------------

    def test_missing_sex_exits_nonzero(self, tmp_path):
        """CLI exits with a non-zero code when --sex is omitted."""
        result = _run_cli(
            "--slide-path",
            str(TEST_SLIDE),
            "--output-dir",
            str(tmp_path),
            "--skip-model-download",
            timeout=60,
        )
        assert (
            result.returncode != 0
        ), "--sex is required; CLI should have exited non-zero"

    def test_missing_output_dir_exits_nonzero(self):
        """CLI exits with a non-zero code when --output-dir is omitted."""
        result = _run_cli(
            "--slide-path",
            str(TEST_SLIDE),
            "--sex",
            TEST_SLIDE_METADATA["sex"],
            "--skip-model-download",
            timeout=60,
        )
        assert (
            result.returncode != 0
        ), "--output-dir is required; CLI should have exited non-zero"


# ---------------------------------------------------------------------------
# Batch CSV mode tests (single slide per sample)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_require_test_slide")
class TestCLIBatchMode:
    """End-to-end tests for batch CSV mode (--slide-csv).

    Uses a single-entry CSV so the run is comparable in duration to single-slide
    mode; the main assertions are that batch-specific outputs (combined CSVs)
    are produced alongside the per-sample files.
    """

    @pytest.fixture(scope="class")
    def batch_csv(self, tmp_path_factory):
        """Write a minimal batch CSV and return its path."""
        csv_path = tmp_path_factory.mktemp("batch_csv") / "settings.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Slide",
                    "Site Type",
                    "Sex",
                    "Tissue Site",
                    "Cancer Subtype",
                    "Segmentation Config",
                ]
            )
            writer.writerow(
                [
                    str(TEST_SLIDE),
                    TEST_SLIDE_METADATA["site_type"],
                    TEST_SLIDE_METADATA["sex"],
                    TEST_SLIDE_METADATA["tissue_site"],
                    "Unknown",
                    "Biopsy",
                ]
            )
        return csv_path

    @pytest.fixture(scope="class")
    def cli_output(self, batch_csv, tmp_path_factory):
        """Run the batch-mode CLI once and return the output directory."""
        output_dir = tmp_path_factory.mktemp("cli_batch")
        result = _run_cli(
            "--slide-csv",
            str(batch_csv),
            "--output-dir",
            str(output_dir),
            "--skip-model-download",
        )
        assert result.returncode == 0, (
            f"Batch CLI exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
        return output_dir

    # -- Output file existence --------------------------------------------------

    def test_per_sample_aeon_csv_created(self, cli_output):
        """Per-sample Aeon results CSV is produced."""
        aeon_files = list(cli_output.glob("*_aeon_results.csv"))
        # Exclude the combined file if present
        per_sample = [f for f in aeon_files if "combined" not in f.name]
        assert per_sample, f"No per-sample Aeon CSV found in {cli_output}"

    def test_per_sample_paladin_csv_created(self, cli_output):
        """Per-sample Paladin results CSV is produced."""
        paladin_files = list(cli_output.glob("*_paladin_results.csv"))
        per_sample = [f for f in paladin_files if "combined" not in f.name]
        assert per_sample, f"No per-sample Paladin CSV found in {cli_output}"

    def test_combined_aeon_csv_created(self, cli_output):
        """Combined Aeon results CSV is written for the whole batch."""
        combined = cli_output / "combined_aeon_results.csv"
        assert combined.exists(), f"combined_aeon_results.csv not found in {cli_output}"

    def test_combined_paladin_csv_created(self, cli_output):
        """Combined Paladin results CSV is written for the whole batch."""
        combined = cli_output / "combined_paladin_results.csv"
        assert (
            combined.exists()
        ), f"combined_paladin_results.csv not found in {cli_output}"

    # -- Content correctness ---------------------------------------------------

    def test_per_sample_aeon_results_sensible(self, cli_output):
        """Per-sample Aeon CSV has valid predictions."""
        per_sample = [
            f for f in cli_output.glob("*_aeon_results.csv") if "combined" not in f.name
        ]
        _assert_aeon_csv_sensible(
            per_sample[0],
            expected_cancer_type=TEST_SLIDE_EXPECTED_CANCER_TYPE,
        )

    def test_per_sample_paladin_results_sensible(self, cli_output):
        """Per-sample Paladin CSV has valid biomarker probabilities."""
        per_sample = [
            f
            for f in cli_output.glob("*_paladin_results.csv")
            if "combined" not in f.name
        ]
        _assert_paladin_csv_sensible(per_sample[0])

    def test_combined_aeon_csv_content(self, cli_output):
        """Combined Aeon CSV includes the full cancer subtype name (not just code)."""
        df = pd.read_csv(cli_output / "combined_aeon_results.csv")
        assert len(df) >= 5, "Combined Aeon CSV has too few rows"
        # The combined file adds full names like "High-Grade Serous Ovarian Cancer (HGSOC)"
        top_subtypes = df["Cancer Subtype"].tolist()
        assert any(
            TEST_SLIDE_EXPECTED_CANCER_TYPE in str(st) for st in top_subtypes[:5]
        ), (
            f"Expected '{TEST_SLIDE_EXPECTED_CANCER_TYPE}' not found in top-5 of combined "
            f"Aeon CSV: {top_subtypes[:5]}"
        )

    def test_combined_paladin_csv_has_sample_column(self, cli_output):
        """Combined Paladin CSV includes a 'Sample' column for traceability."""
        df = pd.read_csv(cli_output / "combined_paladin_results.csv")
        assert (
            "Sample" in df.columns
        ), f"'Sample' column missing from combined Paladin CSV; columns: {df.columns.tolist()}"

    # -- Error-path validation -------------------------------------------------

    def test_missing_output_dir_exits_nonzero(self, batch_csv):
        """CLI exits with a non-zero code when --output-dir is omitted."""
        result = _run_cli(
            "--slide-csv",
            str(batch_csv),
            "--skip-model-download",
            timeout=60,
        )
        assert (
            result.returncode != 0
        ), "--output-dir is required; CLI should have exited non-zero"


# ---------------------------------------------------------------------------
# Multi-slide batch mode tests (feature aggregation)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_require_test_slide")
class TestCLIMultiSlideBatch:
    """End-to-end tests for multi-slide aggregation in batch CSV mode.

    When the same Sample ID appears on multiple rows in the CSV, Mosaic
    extracts features per-slide and then aggregates them before inference.
    Here we use the same slide listed twice under one sample ID.
    """

    SAMPLE_ID = "HGSOC_sample_1"

    @pytest.fixture(scope="class")
    def multi_slide_csv(self, tmp_path_factory):
        """Write a CSV with the test slide listed twice under one Sample ID."""
        csv_path = tmp_path_factory.mktemp("multi_csv") / "settings.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Slide",
                    "Site Type",
                    "Sex",
                    "Tissue Site",
                    "Cancer Subtype",
                    "Segmentation Config",
                    "Sample ID",
                ]
            )
            for _ in range(2):
                writer.writerow(
                    [
                        str(TEST_SLIDE),
                        TEST_SLIDE_METADATA["site_type"],
                        TEST_SLIDE_METADATA["sex"],
                        TEST_SLIDE_METADATA["tissue_site"],
                        "Unknown",
                        "Biopsy",
                        self.SAMPLE_ID,
                    ]
                )
        return csv_path

    @pytest.fixture(scope="class")
    def cli_output(self, multi_slide_csv, tmp_path_factory):
        """Run the multi-slide batch CLI once and return the output directory."""
        output_dir = tmp_path_factory.mktemp("cli_multi")
        result = _run_cli(
            "--slide-csv",
            str(multi_slide_csv),
            "--output-dir",
            str(output_dir),
            "--skip-model-download",
        )
        assert result.returncode == 0, (
            f"Multi-slide CLI exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
        return output_dir

    def test_aggregated_aeon_csv_created(self, cli_output):
        """Aeon results for the aggregated sample are written."""
        aeon_csv = cli_output / f"{self.SAMPLE_ID}_aeon_results.csv"
        assert aeon_csv.exists(), (
            f"Expected aggregated Aeon CSV at {aeon_csv}. "
            f"Files in output dir: {list(cli_output.iterdir())}"
        )

    def test_aggregated_paladin_csv_created(self, cli_output):
        """Paladin results for the aggregated sample are written."""
        paladin_csv = cli_output / f"{self.SAMPLE_ID}_paladin_results.csv"
        assert paladin_csv.exists(), (
            f"Expected aggregated Paladin CSV at {paladin_csv}. "
            f"Files in output dir: {list(cli_output.iterdir())}"
        )

    def test_aggregated_aeon_results_sensible(self, cli_output):
        """Aggregated Aeon predictions are structurally sound."""
        _assert_aeon_csv_sensible(
            cli_output / f"{self.SAMPLE_ID}_aeon_results.csv",
            expected_cancer_type=TEST_SLIDE_EXPECTED_CANCER_TYPE,
        )

    def test_aggregated_paladin_results_sensible(self, cli_output):
        """Aggregated Paladin predictions have valid biomarker probabilities."""
        _assert_paladin_csv_sensible(
            cli_output / f"{self.SAMPLE_ID}_paladin_results.csv"
        )

    def test_combined_aeon_csv_created(self, cli_output):
        """Combined Aeon CSV is produced even in multi-slide batch mode."""
        assert (cli_output / "combined_aeon_results.csv").exists()

    def test_combined_paladin_csv_created(self, cli_output):
        """Combined Paladin CSV is produced even in multi-slide batch mode."""
        assert (cli_output / "combined_paladin_results.csv").exists()

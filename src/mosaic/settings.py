"""Pure slide-analysis settings — constants, loading, and normalization.

This module has **no Gradio dependency** and can be safely imported from CLI
code, background workers, and tests without triggering the Gradio import chain.

UI code that needs to surface validation feedback through Gradio components
should use :func:`mosaic.ui.utils.validate_settings`, which wraps
:func:`normalize_settings` with ``gr.Warning`` / ``gr.skip`` semantics.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from mosaic.data_directory import get_data_directory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IHC_SUBTYPES = ["", "HR+/HER2+", "HR+/HER2-", "HR-/HER2+", "HR-/HER2-"]
SEX_OPTIONS = ["Male", "Female"]

# Column name constants — single source of truth for CSV column names.
COL_SLIDE = "Slide"
COL_SITE_TYPE = "Site Type"
COL_SEX = "Sex"
COL_TISSUE_SITE = "Tissue Site"
COL_CANCER_SUBTYPE = "Cancer Subtype"
COL_IHC_SUBTYPE = "IHC Subtype"
COL_SEG_CONFIG = "Segmentation Config"
COL_SAMPLE_ID = "Sample ID"

SETTINGS_COLUMNS = [
    COL_SLIDE,
    COL_SITE_TYPE,
    COL_SEX,
    COL_TISSUE_SITE,
    COL_CANCER_SUBTYPE,
    # COL_IHC_SUBTYPE,  # Not yet in use
    COL_SEG_CONFIG,
    COL_SAMPLE_ID,  # Optional: slides sharing a Sample ID are aggregated before inference
]

# ---------------------------------------------------------------------------
# Tissue-site helper
# ---------------------------------------------------------------------------

_tissue_site_list = None


def get_tissue_sites() -> list[str]:
    """Return the list of valid tissue site names.

    Loads ``tissue_site_original_to_idx.csv`` from the data directory on first
    call and caches the result.  Falls back to ``["Unknown"]`` and emits a log
    warning if the file is not found (e.g. models not yet downloaded).

    Returns:
        Sorted list of tissue site names, always starting with ``"Unknown"``.
    """
    global _tissue_site_list
    if _tissue_site_list is not None:
        return _tissue_site_list

    try:
        data_dir = get_data_directory()
        tissue_site_map_path = data_dir / "tissue_site_original_to_idx.csv"
        df = pd.read_csv(tissue_site_map_path)
        _tissue_site_list = ["Unknown"] + sorted(df["TISSUE_SITE"].unique().tolist())
    except FileNotFoundError:
        logger.warning(
            f"Tissue site mapping file not found at {tissue_site_map_path}. "
            "Only 'Unknown' will be available for tissue site selection. "
            "Ensure the model data files are downloaded from the model repository."
        )
        _tissue_site_list = ["Unknown"]

    return _tissue_site_list


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_settings(slide_csv_path: str) -> pd.DataFrame:
    """Load slide analysis settings from a CSV file.

    Ensures all required columns are present, adding safe defaults for optional
    columns when absent, and backfills blank Sample IDs from slide filename
    stems.

    Args:
        slide_csv_path: Path to the settings CSV.

    Returns:
        ``pd.DataFrame`` with exactly the columns in :data:`SETTINGS_COLUMNS`.

    Raises:
        ValueError: If a required column is missing from the CSV.
    """
    settings_df = pd.read_csv(slide_csv_path, na_filter=False)
    if COL_SEG_CONFIG not in settings_df.columns:
        settings_df[COL_SEG_CONFIG] = "Biopsy"
    if COL_CANCER_SUBTYPE not in settings_df.columns:
        settings_df[COL_CANCER_SUBTYPE] = "Unknown"
    if COL_TISSUE_SITE not in settings_df.columns:
        settings_df[COL_TISSUE_SITE] = "Unknown"
    if COL_SAMPLE_ID not in settings_df.columns:
        settings_df[COL_SAMPLE_ID] = ""
    required = [c for c in SETTINGS_COLUMNS if c != COL_SAMPLE_ID]
    if not set(required).issubset(settings_df.columns):
        raise ValueError("Missing required column in CSV file")
    settings_df = settings_df[SETTINGS_COLUMNS]
    # Backfill blank Sample IDs with the slide filename stem
    mask = settings_df[COL_SAMPLE_ID].astype(str).str.strip() == ""
    settings_df.loc[mask, COL_SAMPLE_ID] = settings_df.loc[mask, COL_SLIDE].apply(
        lambda p: Path(p).stem
    )
    return settings_df


# ---------------------------------------------------------------------------
# Pure normalization (no Gradio)
# ---------------------------------------------------------------------------


def normalize_settings(
    settings_df: pd.DataFrame,
    cancer_subtype_name_map: dict,
    cancer_subtypes: list,
    reversed_cancer_subtype_name_map: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and normalize slide analysis settings.

    Corrects invalid field values to safe defaults and collects a human-readable
    warning message for every correction made.  Never raises; all errors are
    handled gracefully.

    The caller is responsible for surfacing the warnings in an appropriate way
    — ``gr.Warning`` in UI code, ``logger.warning`` in CLI code.

    Args:
        settings_df: DataFrame whose columns already match
            :data:`SETTINGS_COLUMNS` (i.e. after :func:`load_settings`).
        cancer_subtype_name_map: Mapping of display names → oncotree codes,
            e.g. ``{"Unknown": "UNK", "Lung Adenocarcinoma (LUAD)": "LUAD"}``.
        cancer_subtypes: List of valid oncotree code strings (e.g. ``["LUAD"]``).
        reversed_cancer_subtype_name_map: Inverse of *cancer_subtype_name_map*.

    Returns:
        ``(normalized_df, warnings)`` where *normalized_df* is a new DataFrame
        with corrected values and *warnings* is a (possibly empty) list of
        human-readable warning strings.
    """
    settings_df = settings_df.copy()
    warnings: list[str] = []
    tissue_sites = get_tissue_sites()

    for idx, row in settings_df.iterrows():
        slide_name = row[COL_SLIDE]
        subtype = row[COL_CANCER_SUBTYPE]

        # Normalize raw oncotree code → display name
        if subtype in cancer_subtypes:
            settings_df.at[idx, COL_CANCER_SUBTYPE] = reversed_cancer_subtype_name_map[
                subtype
            ]
        if settings_df.at[idx, COL_CANCER_SUBTYPE] not in cancer_subtype_name_map:
            warnings.append(
                f"Slide {slide_name}: Unknown cancer subtype. "
                f"Valid subtypes are: {', '.join(cancer_subtype_name_map)}. "
            )
            settings_df.at[idx, COL_CANCER_SUBTYPE] = "Unknown"

        if row[COL_SITE_TYPE] not in ["Metastatic", "Primary"]:
            warnings.append(
                f"Slide {slide_name}: Unknown site type. "
                "Valid types are: Metastatic, Primary. "
            )
            settings_df.at[idx, COL_SITE_TYPE] = "Primary"

        sex_value = row[COL_SEX]
        if sex_value == "Unknown":
            # Silently convert legacy "Unknown" to the canonical empty string
            settings_df.at[idx, COL_SEX] = ""
        elif sex_value and sex_value not in SEX_OPTIONS:
            warnings.append(
                f"Slide {slide_name}: Invalid sex value '{sex_value}'. "
                f"Valid options are: {', '.join(SEX_OPTIONS)}. "
            )
            settings_df.at[idx, COL_SEX] = ""

        if row[COL_TISSUE_SITE] not in tissue_sites:
            warnings.append(
                f"Slide {slide_name}: Unknown tissue site. "
                f"Valid tissue sites are: {', '.join(tissue_sites)}. "
            )
            settings_df.at[idx, COL_TISSUE_SITE] = "Unknown"

        if row[COL_SEG_CONFIG] not in ["Biopsy", "Resection", "TCGA"]:
            warnings.append(
                f"Slide {slide_name}: Unknown segmentation config. "
                "Valid configs are: Biopsy, Resection, TCGA. "
            )
            settings_df.at[idx, COL_SEG_CONFIG] = "Biopsy"

    return settings_df, warnings


def row_to_slide_kwargs(row) -> dict:
    """Extract the standard ``analyze_slide`` keyword arguments from a settings row.

    Args:
        row: A row from a settings DataFrame (e.g. ``settings_df.iloc[i]``).

    Returns:
        Dict ready for ``**``-unpacking into :func:`mosaic.analysis.analyze_slide`,
        :func:`mosaic.tcga.compute_settings_hash`, and similar functions.
    """
    return {
        "seg_config": row[COL_SEG_CONFIG],
        "site_type": row[COL_SITE_TYPE],
        "sex": row[COL_SEX],
        "tissue_site": row.get(COL_TISSUE_SITE, "Unknown"),
        "cancer_subtype": row[COL_CANCER_SUBTYPE],
        "ihc_subtype": row.get(COL_IHC_SUBTYPE, ""),
    }

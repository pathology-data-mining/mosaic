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

SETTINGS_COLUMNS = [
    "Slide",
    "Site Type",
    "Sex",
    "Tissue Site",
    "Cancer Subtype",
    # "IHC Subtype",  # Not yet in use
    "Segmentation Config",
    "Sample ID",  # Optional: slides sharing a Sample ID are aggregated before inference
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
    if "Segmentation Config" not in settings_df.columns:
        settings_df["Segmentation Config"] = "Biopsy"
    if "Cancer Subtype" not in settings_df.columns:
        settings_df["Cancer Subtype"] = "Unknown"
    if "Tissue Site" not in settings_df.columns:
        settings_df["Tissue Site"] = "Unknown"
    if "Sample ID" not in settings_df.columns:
        settings_df["Sample ID"] = ""
    required = [c for c in SETTINGS_COLUMNS if c != "Sample ID"]
    if not set(required).issubset(settings_df.columns):
        raise ValueError("Missing required column in CSV file")
    settings_df = settings_df[SETTINGS_COLUMNS]
    # Backfill blank Sample IDs with the slide filename stem
    mask = settings_df["Sample ID"].astype(str).str.strip() == ""
    settings_df.loc[mask, "Sample ID"] = settings_df.loc[mask, "Slide"].apply(
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
        slide_name = row["Slide"]
        subtype = row["Cancer Subtype"]

        # Normalize raw oncotree code → display name
        if subtype in cancer_subtypes:
            settings_df.at[idx, "Cancer Subtype"] = reversed_cancer_subtype_name_map[
                subtype
            ]
        if settings_df.at[idx, "Cancer Subtype"] not in cancer_subtype_name_map:
            warnings.append(
                f"Slide {slide_name}: Unknown cancer subtype. "
                f"Valid subtypes are: {', '.join(cancer_subtype_name_map)}. "
            )
            settings_df.at[idx, "Cancer Subtype"] = "Unknown"

        if row["Site Type"] not in ["Metastatic", "Primary"]:
            warnings.append(
                f"Slide {slide_name}: Unknown site type. "
                "Valid types are: Metastatic, Primary. "
            )
            settings_df.at[idx, "Site Type"] = "Primary"

        sex_value = row["Sex"]
        if sex_value == "Unknown":
            # Silently convert legacy "Unknown" to the canonical empty string
            settings_df.at[idx, "Sex"] = ""
        elif sex_value and sex_value not in SEX_OPTIONS:
            warnings.append(
                f"Slide {slide_name}: Invalid sex value '{sex_value}'. "
                f"Valid options are: {', '.join(SEX_OPTIONS)}. "
            )
            settings_df.at[idx, "Sex"] = ""

        if row["Tissue Site"] not in tissue_sites:
            warnings.append(
                f"Slide {slide_name}: Unknown tissue site. "
                f"Valid tissue sites are: {', '.join(tissue_sites)}. "
            )
            settings_df.at[idx, "Tissue Site"] = "Unknown"

        if row["Segmentation Config"] not in ["Biopsy", "Resection", "TCGA"]:
            warnings.append(
                f"Slide {slide_name}: Unknown segmentation config. "
                "Valid configs are: Biopsy, Resection, TCGA. "
            )
            settings_df.at[idx, "Segmentation Config"] = "Biopsy"

    return settings_df, warnings

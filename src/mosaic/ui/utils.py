"""UI utility functions for the Mosaic Gradio interface.

This module provides helper functions for:
- OncoTree code lookup and caching
- User session directory management
- Settings CSV loading and validation
- Data export functionality
"""

import tempfile
from pathlib import Path
import pandas as pd
import gradio as gr
import requests

from mosaic.data_directory import get_data_directory

# This path should be outside your project directory if running locally
TEMP_USER_DATA_DIR = Path(tempfile.gettempdir()) / "mosaic_user_data"

IHC_SUBTYPES = ["", "HR+/HER2+", "HR+/HER2-", "HR-/HER2+", "HR-/HER2-"]
SEX_OPTIONS = ["Male", "Female"]

SETTINGS_COLUMNS = [
    "Slide",
    "Site Type",
    "Sex",
    "Tissue Site",
    "Cancer Subtype",
    "IHC Subtype",
    "Segmentation Config",
]

oncotree_code_map = {}
_oncotree_prefetched = False
tissue_site_list = None


def get_tissue_sites():
    """Get the list of tissue sites from the tissue site map file.

    Returns:
        List of tissue site names. Returns ["Unknown"] if the CSV file is not found.
    """
    global tissue_site_list
    if tissue_site_list is None:
        try:
            data_dir = get_data_directory()
            tissue_site_map_path = data_dir / "tissue_site_original_to_idx.csv"
            df = pd.read_csv(tissue_site_map_path)
            # Get unique tissue sites and sort them
            tissue_site_list = ["Unknown"] + sorted(df["TISSUE_SITE"].unique().tolist())
        except FileNotFoundError:
            gr.Warning(
                f"Tissue site mapping file not found at {tissue_site_map_path}. "
                "Only 'Unknown' option will be available for tissue site selection. "
                "Please ensure the data files are downloaded from the model repository."
            )
            tissue_site_list = ["Unknown"]
    return tissue_site_list


def _prefetch_oncotree_codes():
    """Fetch all OncoTree codes in a single bulk API request.

    Populates the oncotree_code_map cache so that individual
    get_oncotree_code_name() calls never need to make HTTP requests.
    """
    global oncotree_code_map, _oncotree_prefetched
    if _oncotree_prefetched:
        return
    _oncotree_prefetched = True
    try:
        url = "https://oncotree.mskcc.org/api/tumorTypes?version=oncotree_2025_04_08"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            for entry in response.json():
                if "code" in entry and "name" in entry:
                    oncotree_code_map[entry["code"]] = entry["name"]
    except Exception:
        pass  # Fall back to per-code lookups


def get_oncotree_code_name(code):
    """Retrieve the human-readable name for an OncoTree code.

    On first call, fetches all codes from the OncoTree API in a single
    bulk request. Subsequent calls use the cached map.

    Args:
        code: OncoTree code (e.g., "LUAD", "BRCA")

    Returns:
        Human-readable cancer subtype name, or "Unknown" if not found
    """
    global oncotree_code_map
    if not _oncotree_prefetched:
        _prefetch_oncotree_codes()
    if code in oncotree_code_map:
        return oncotree_code_map[code]

    # Fallback for codes not in bulk response
    url = f"https://oncotree.mskcc.org/api/tumorTypes/search/code/{code}?exactMatch=true&version=oncotree_2025_04_08"
    try:
        response = requests.get(url, timeout=10)
        code_name = "Unknown"
        if response.status_code == 200:
            data = response.json()
            if data:
                code_name = data[0]["name"]
    except Exception:
        code_name = "Unknown"

    oncotree_code_map[code] = code_name
    return code_name


def create_user_directory(state, request: gr.Request):
    """Create a unique directory for each user session.

    Args:
        state: Gradio state object (unused)
        request: Gradio request object containing session hash

    Returns:
        Path to user's session directory, or None if no session hash available
    """
    session_hash = request.session_hash
    if session_hash is None:
        return None

    user_dir = TEMP_USER_DATA_DIR / session_hash
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def load_settings(slide_csv_path):
    """Load slide analysis settings from CSV file.

    Loads the CSV and ensures all required columns are present, adding defaults
    for optional columns if they are missing.

    Args:
        slide_csv_path: Path to the CSV file containing slide settings

    Returns:
        DataFrame with columns: Slide, Site Type, Cancer Subtype, IHC Subtype, Segmentation Config

    Raises:
        ValueError: If required columns are missing from the CSV
    """
    settings_df = pd.read_csv(slide_csv_path, na_filter=False)
    if "Segmentation Config" not in settings_df.columns:
        settings_df["Segmentation Config"] = "Biopsy"
    if "Cancer Subtype" not in settings_df.columns:
        settings_df["Cancer Subtype"] = "Unknown"
    if "IHC Subtype" not in settings_df.columns:
        settings_df["IHC Subtype"] = ""
    if "Tissue Site" not in settings_df.columns:
        settings_df["Tissue Site"] = "Unknown"
    if not set(SETTINGS_COLUMNS).issubset(settings_df.columns):
        raise ValueError("Missing required column in CSV file")
    settings_df = settings_df[SETTINGS_COLUMNS]
    return settings_df


def validate_settings(
    settings_df,
    cancer_subtype_name_map,
    cancer_subtypes,
    reversed_cancer_subtype_name_map,
):
    """Validate and normalize slide analysis settings.

    Checks each row for valid values and normalizes cancer subtype names.
    Generates warnings for invalid entries and replaces them with defaults.

    Args:
        settings_df: DataFrame with slide settings to validate
        cancer_subtype_name_map: Dict mapping subtype display names to codes
        cancer_subtypes: List of valid cancer subtype codes
        reversed_cancer_subtype_name_map: Dict mapping codes to display names

    Returns:
        Validated DataFrame with normalized values

    Note:
        Invalid entries are replaced with defaults and warnings are displayed
        to the user via Gradio warnings.
    """
    settings_df.columns = SETTINGS_COLUMNS
    warnings = []
    tissue_sites = get_tissue_sites()

    for idx, row in settings_df.iterrows():
        slide_name = row["Slide"]
        subtype = row["Cancer Subtype"]
        if subtype in cancer_subtypes:
            settings_df.at[idx, "Cancer Subtype"] = reversed_cancer_subtype_name_map[
                subtype
            ]
        if settings_df.at[idx, "Cancer Subtype"] not in cancer_subtype_name_map.keys():
            warnings.append(
                f"Slide {slide_name}: Unknown cancer subtype. Valid subtypes are: {', '.join(cancer_subtype_name_map.keys())}. "
            )
            settings_df.at[idx, "Cancer Subtype"] = "Unknown"
        if row["Site Type"] not in ["Metastatic", "Primary"]:
            warnings.append(
                f"Slide {slide_name}: Unknown site type. Valid types are: Metastatic, Primary. "
            )
            settings_df.at[idx, "Site Type"] = "Primary"
        # Only warn about invalid sex values that are not empty/None
        # Empty/None will be validated at analysis time
        # Convert old "Unknown" values to empty string silently
        sex_value = row["Sex"]
        if sex_value == "Unknown":
            settings_df.at[idx, "Sex"] = ""
        elif sex_value and sex_value not in SEX_OPTIONS:
            warnings.append(
                f"Slide {slide_name}: Invalid sex value '{sex_value}'. Valid options are: {', '.join(SEX_OPTIONS)}. "
            )
            settings_df.at[idx, "Sex"] = ""
        if row["Tissue Site"] not in tissue_sites:
            warnings.append(
                f"Slide {slide_name}: Unknown tissue site. Valid tissue sites are: {', '.join(tissue_sites)}. "
            )
            settings_df.at[idx, "Tissue Site"] = "Unknown"
        if (
            "Breast" not in settings_df.at[idx, "Cancer Subtype"]
            and row["IHC Subtype"] != ""
        ):
            warnings.append(
                f"Slide {slide_name}: IHC subtype should be empty for non-breast cancer subtypes. "
            )
            settings_df.at[idx, "IHC Subtype"] = ""
        if row["IHC Subtype"] not in IHC_SUBTYPES:
            warnings.append(
                f"Slide {slide_name}: Unknown IHC subtype. Valid subtypes are: {', '.join(IHC_SUBTYPES)}. "
            )
            settings_df.at[idx, "IHC Subtype"] = ""
        if row["Segmentation Config"] not in ["Biopsy", "Resection", "TCGA"]:
            warnings.append(
                f"Slide {slide_name}: Unknown segmentation config. Valid configs are: Biopsy, Resection, TCGA. "
            )
            settings_df.at[idx, "Segmentation Config"] = "Biopsy"

    if warnings:
        gr.Warning("\n".join(warnings))

    return settings_df


def export_to_csv(df):
    """Export a DataFrame to CSV file for download.

    Args:
        df: DataFrame to export

    Returns:
        Path to the exported CSV file

    Raises:
        gr.Error: If the DataFrame is None or empty
    """
    if df is None or df.empty:
        raise gr.Error("No data to export.")
    csv_path = "paladin_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

"""UI utility functions for the Mosaic Gradio interface.

This module provides helper functions for:
- OncoTree code lookup and caching
- User session directory management
- Settings CSV loading and validation
- Data export functionality

Pure settings logic (constants, loading, normalization) lives in
:mod:`mosaic.settings` with no Gradio dependency.  The symbols are
re-exported here so existing imports continue to work unchanged.
"""

import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

from mosaic.settings import (
    IHC_SUBTYPES,
    SETTINGS_COLUMNS,
    SEX_OPTIONS,
    get_tissue_sites,
    load_settings,
    normalize_settings,
)

# Re-export so callers that do ``from mosaic.ui.utils import SETTINGS_COLUMNS``
# (and similar) continue to work without modification.
__all__ = [
    "IHC_SUBTYPES",
    "SETTINGS_COLUMNS",
    "SEX_OPTIONS",
    "get_tissue_sites",
    "load_settings",
    "normalize_settings",
    "validate_settings",
    "get_oncotree_code_name",
    "format_cancer_subtype",
    "expand_cancer_subtype_column",
    "create_user_directory",
    "export_to_csv",
    "TEMP_USER_DATA_DIR",
]

# This path should be outside your project directory if running locally
TEMP_USER_DATA_DIR = Path(tempfile.gettempdir()) / "mosaic_user_data"

oncotree_code_map = {}
_oncotree_prefetched = False


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


def format_cancer_subtype(code: str) -> str:
    """Format an OncoTree code as a human-readable display string.

    Args:
        code: OncoTree cancer subtype code (e.g. ``"LUAD"``).

    Returns:
        Display string of the form ``"<Name> (<CODE>)"``
        (e.g. ``"Lung Adenocarcinoma (LUAD)"``).
    """
    return f"{get_oncotree_code_name(code)} ({code})"


def expand_cancer_subtype_column(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw OncoTree codes in the ``"Cancer Subtype"`` column with display names.

    Args:
        df: DataFrame with a ``"Cancer Subtype"`` column of OncoTree codes.

    Returns:
        The same DataFrame (mutated in-place) with ``"Cancer Subtype"`` values
        replaced by ``"<Name> (<CODE>)"`` display strings.
    """
    df["Cancer Subtype"] = df["Cancer Subtype"].map(format_cancer_subtype)
    return df


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


def validate_settings(
    settings_df,
    cancer_subtype_name_map,
    cancer_subtypes,
    reversed_cancer_subtype_name_map,
):
    """Gradio-aware wrapper around :func:`mosaic.settings.normalize_settings`.

    Handles the UI-specific case where the Gradio DataFrame component emits a
    6-column DataFrame (without "Sample ID"), normalizes the schema, then
    delegates to :func:`~mosaic.settings.normalize_settings` for the pure
    validation logic.

    Validation warnings are surfaced via ``gr.Warning``; returns ``gr.skip()``
    when no changes were made to prevent the ``settings_input.change`` handler
    from re-firing (which would cause a Svelte reactive cascade and browser hang).

    Args:
        settings_df: DataFrame with slide settings to validate.
        cancer_subtype_name_map: Dict mapping subtype display names to codes.
        cancer_subtypes: List of valid cancer subtype codes.
        reversed_cancer_subtype_name_map: Dict mapping codes to display names.

    Returns:
        Validated ``pd.DataFrame`` with normalized values, or ``gr.skip()`` if
        the settings were already valid and no changes were made.
    """
    # The UI path builds a 6-column DataFrame (no Sample ID); backfill it here
    # so the rest of this function and downstream code use SETTINGS_COLUMNS uniformly.
    if "Sample ID" not in settings_df.columns:
        if len(settings_df.columns) == len(SETTINGS_COLUMNS) - 1:
            # Rename positional columns, then add Sample ID
            settings_df.columns = [c for c in SETTINGS_COLUMNS if c != "Sample ID"]
        settings_df["Sample ID"] = settings_df["Slide"].apply(lambda p: Path(p).stem)
    settings_df = settings_df[SETTINGS_COLUMNS]

    # Capture original values *after* schema normalisation so the equality
    # check below operates on the same column layout.
    original_values = settings_df.copy(deep=True)

    normalized, warnings = normalize_settings(
        settings_df,
        cancer_subtype_name_map,
        cancer_subtypes,
        reversed_cancer_subtype_name_map,
    )

    if warnings:
        gr.Warning("\n".join(warnings))

    # Return gr.skip() to prevent the settings_input.change feedback loop from
    # re-firing (repeated gr.DataFrame re-renders → Svelte reactive cascade →
    # ~420-second browser hang).
    try:
        if normalized.equals(original_values):
            return gr.skip()
    except Exception:
        pass

    return normalized


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

import tempfile
from pathlib import Path
import pandas as pd
import gradio as gr
import requests

# This path should be outside your project directory if running locally
TEMP_USER_DATA_DIR = Path(tempfile.gettempdir()) / "mosaic_user_data"

IHC_SUBTYPES = ["", "HR+/HER2+", "HR+/HER2-", "HR-/HER2+", "HR-/HER2-"]

SETTINGS_COLUMNS = [
    "Slide",
    "Site Type",
    "Cancer Subtype",
    "IHC Subtype",
    "Segmentation Config",
]

oncotree_code_map = {}


def get_oncotree_code_name(code):
    global oncotree_code_map
    if code in oncotree_code_map.keys():
        return oncotree_code_map[code]

    url = f"https://oncotree.mskcc.org/api/tumorTypes/search/code/{code}?exactMatch=true&version=oncotree_2025_04_08"
    response = requests.get(url)
    code_name = "Unknown"
    if response.status_code == 200:
        data = response.json()
        if data:
            code_name = data[0]["name"]

    oncotree_code_map[code] = code_name
    return code_name


def create_user_directory(state, request: gr.Request):
    """Create a unique directory for each user session."""
    session_hash = request.session_hash
    if session_hash is None:
        return None, None

    user_dir = TEMP_USER_DATA_DIR / session_hash
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def load_settings(slide_csv_path):
    """Load settings from CSV file and validate columns."""
    settings_df = pd.read_csv(slide_csv_path, na_filter=False)
    if "Segmentation Config" not in settings_df.columns:
        settings_df["Segmentation Config"] = "Biopsy"
    if "Cancer Subtype" not in settings_df.columns:
        settings_df["Cancer Subtype"] = "Unknown"
    if "IHC Subtype" not in settings_df.columns:
        settings_df["IHC Subtype"] = ""
    if not set(SETTINGS_COLUMNS).issubset(settings_df.columns):
        raise ValueError("Missing required column in CSV file")
    settings_df = settings_df[SETTINGS_COLUMNS]
    return settings_df


def validate_settings(settings_df, cancer_subtype_name_map, cancer_subtypes, reversed_cancer_subtype_name_map):
    """Validate settings DataFrame and provide warnings for invalid entries."""
    settings_df.columns = SETTINGS_COLUMNS
    warnings = []
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
    if df is None or df.empty:
        raise gr.Error("No data to export.")
    csv_path = "paladin_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

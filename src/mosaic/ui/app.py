"""Gradio web interface for Mosaic.

This module provides the web-based user interface for analyzing whole slide images.
It includes functionality for:
- Multi-slide upload and analysis
- TCGA slide fetching by ID (UUID or barcode)
- Settings configuration (site type, cancer subtype, IHC subtype, segmentation)
- Results visualization and export
- CSV-based batch processing

The analysis pipeline (``analyze_slides``, model-download state, cancer-subtype
maps) lives in ``mosaic.ui.pipeline`` and is re-exported here for backward
compatibility.
"""

import os
import tempfile
import threading
import time
from typing import Optional

import gradio as gr
import pandas as pd
from pathlib import Path
from loguru import logger

from mosaic.ui.utils import (
    create_user_directory,
    load_settings,
    validate_settings,
    get_tissue_sites,
    IHC_SUBTYPES,
    SEX_OPTIONS,
    SETTINGS_COLUMNS,
)
from mosaic.hardware import DEFAULT_CONCURRENCY_LIMIT, IS_HF_SPACES
from mosaic.tcga import (
    download_results_from_hf,
    fetch_slide,
    get_case_metadata,
    validate_slide_id,
    GDCError,
    GDCInvalidIDError,
    GDCNotFoundError,
    GDCAccessDeniedError,
)

# Pipeline module owns globals (cancer subtype maps, model-download state).
# Access via ``_pipeline.<name>`` so that mutations in pipeline.py are visible
# here without re-importing.
from mosaic.ui import pipeline as _pipeline

# Backward-compatible re-exports — external callers (tests, gradio_app.py) import
# these names from ``mosaic.ui.app`` and must continue to work unchanged.
from mosaic.ui.pipeline import (  # noqa: F401
    analyze_slides,
    set_cancer_subtype_maps,
    cancer_subtype_name_map,
)

current_dir = Path(__file__).parent.parent


def launch_gradio(server_name, server_port, share):
    # Initialize telemetry and log app start
    from mosaic.telemetry import TelemetryTracker
    import os

    tracker = TelemetryTracker.get_instance()
    tracker.log_app_start()

    # Start downloading models in background
    _pipeline.start_background_model_download()

    with gr.Blocks(title="Mosaic") as demo:
        user_dir_state = gr.State(None)
        slide_ids_state = gr.State(
            {}
        )  # Maps slide filename to slide_id for user storage

        # Add login button for OAuth.
        # On HF Spaces the LoginButton is always hidden because sdk:docker
        # injects the Space owner's identity into request.username, causing
        # the button to show "Logged in as <owner>" for all visitors.
        # Instead we use a custom Markdown with direct OAuth route links.
        # The LoginButton must still exist in the layout so Gradio registers
        # the /login/huggingface and /logout OAuth routes (via expects_oauth).
        login_button = gr.LoginButton(visible=not IS_HF_SPACES)

        if IS_HF_SPACES:
            # LoginButton.activate() registers a load event (_check_login_status)
            # that returns a NEW LoginButton with visible=True, overriding our
            # visible=False. It reads session oauth_info (populated by the HF
            # reverse proxy with the Space owner's identity) and shows
            # "Logout (raylim)" to everyone. Clear the load event so the button
            # stays permanently hidden.
            login_button.load_event_to_attach = None

            # Custom login display — sign-in via markdown link,
            # sign-out via a Gradio button (stays in Gradio event system,
            # avoiding navigation away from the SPA).
            with gr.Row():
                login_status_md = gr.Markdown(
                    value="[Sign in with HuggingFace](/api/auth/login) to save slides and results",
                    visible=True,
                )
                logout_button = gr.Button(
                    "Sign out",
                    visible=False,
                    size="sm",
                    scale=0,
                    min_width=80,
                )

        gr.Markdown(
            "# Mosaic: H&E Whole Slide Image Cancer Subtype and Biomarker Inference"
        )
        gr.Markdown(
            "Upload an H&E whole slide image in SVS or TIFF format, or fetch a slide directly from TCGA. The slide will be processed to infer cancer subtype and relevant biomarkers."
        )

        # Comprehensive documentation section
        with gr.Accordion("📖 Documentation & Help", open=False):
            gr.Markdown(
                """
## Quick Start Guide

**Single Slide Analysis:**
1. Upload an H&E whole slide image (.svs, .tiff, or .tif)
2. Configure settings:
   - **Sex**: Required - Patient biological sex (affects analysis)
   - **Site Type**: Primary tumor or metastatic site
   - **Tissue Site**: Anatomical location (if known)
   - **Cancer Subtype**: Leave as "Unknown" for automatic inference
   - **Segmentation Config**: Choose based on slide type
     - **Biopsy**: For small tissue samples
     - **Resection**: For surgical specimens
     - **TCGA**: For TCGA-compatible slides
3. Click "Analyze" and wait for results

**Fetch from TCGA:**
1. Select "Fetch from TCGA" input mode
2. Enter a TCGA slide ID (UUID or barcode like TCGA-A1-A0SB-01Z-00-DX1)
3. Click "Fetch Slide" - metadata will auto-fill the dropdowns
4. Review/adjust settings, then click "Analyze"

**Batch Processing:**
- Upload multiple slides at once to process them sequentially
- Use "Settings CSV Upload" for batch configuration with a CSV file

---

## Input Requirements

**Supported Formats:**
- SVS (Aperio ScanScope Virtual Slide)
- TIFF/TIF (Multi-resolution pyramid TIFF)

**File Requirements:**
- H&E stained whole slide images
- Standard microscopy scanning resolution
- Compatible with OpenSlide library

**Configuration Fields:**
- **Sex** (Required): Male/Female - Used for sex-specific cancer analysis
- **Site Type**: Primary vs Metastatic - Affects biomarker interpretation
- **Tissue Site**: 60+ anatomical sites supported (see dropdown)
- **Cancer Subtype**: Use "Unknown" for automatic inference via Aeon model

---

## Understanding Your Results

**Slide Masks:**
Visual representation of tissue segmentation showing analyzed regions

**Cancer Subtype Predictions:**
- Uses **Aeon foundation model** for cancer type classification
- Confidence scores range 0-1 (higher = more confident)
- Displays top predicted subtypes with OncoTree codes

**Biomarker Predictions:**
- Uses **Paladin model** for biomarker inference
- Biomarkers are cancer-subtype specific
- Scores indicate predicted expression/mutation status

**OncoTree Codes:**
Standardized cancer classification system (e.g., LUAD = Lung Adenocarcinoma, BRCA = Breast Invasive Carcinoma). Learn more at [oncotree.mskcc.org](https://oncotree.mskcc.org/)

---

## Processing Information

**Expected Processing Time:**
- Single slide: 5-15 minutes (depending on slide size)
- Multiple slides: Processed sequentially
- Results appear progressively as each slide completes

**Hardware:**
- Running on GPU-accelerated infrastructure
- Models: CTransPath, Optimus, Aeon (subtype), Paladin (biomarker)
- Background model loading may occur on first use

---

## CSV Batch Upload Format

A settings CSV lets you specify per-slide metadata for batch processing. Upload the CSV alongside your slide files.

**Columns:**

| Column | Required | Valid Values | Default |
|--------|----------|--------------|---------|
| `Slide` | ✅ Yes | Filename matching an uploaded slide (e.g., `slide1.svs`) | — |
| `Site Type` | ✅ Yes | `Primary`, `Metastatic` | `Primary` if invalid |
| `Sex` | ✅ Yes | `Male`, `Female` | Must be provided; empty = prompted at analysis |
| `Tissue Site` | No | Any valid anatomical site (see dropdown), or `Unknown` | `Unknown` |
| `Cancer Subtype` | No | OncoTree code (e.g., `LUAD`) or display name, or `Unknown` | `Unknown` (auto-inferred by Aeon) |
| `Segmentation Config` | No | `Biopsy`, `Resection`, `TCGA` | `Biopsy` |

**Example:**
```
Slide,Site Type,Sex,Tissue Site,Cancer Subtype,Segmentation Config
slide1.svs,Primary,Female,Breast,Unknown,Resection
slide2.tif,Metastatic,Male,Lung,LUAD,Biopsy
slide3.svs,Primary,Female,Unknown,Unknown,Biopsy
```

**Notes:**
- The CSV file itself can have any name (e.g., `my_batch.csv`, `settings.csv`)
- `Slide` values must exactly match the filenames of slides uploaded in the same session
- Omitting `Cancer Subtype` (or using `Unknown`) lets the Aeon model infer the subtype automatically
- Omitting optional columns entirely is allowed — defaults are applied automatically
- Rows with invalid `Site Type` or `Segmentation Config` values fall back to their defaults with a warning

---

## TCGA Slide Fetching

**Supported ID Formats:**
- **UUID**: 36-character identifier (e.g., `9c960533-ce6d-4a77-9227-eb974710a607`)
- **TCGA Barcode**: Standard TCGA format (e.g., `TCGA-E2-A1LS-01Z-00-DX1`)

**Example Slide IDs (Public Access, with Paladin biomarker models):**
| ID | Cancer Type |
|----|-------------|
| `TCGA-86-8074-01Z-00-DX1` | Lung Adenocarcinoma (LUAD) |
| `TCGA-AZ-4313-01Z-00-DX1` | Colon Adenocarcinoma (COAD) |
| `TCGA-EJ-7123-01Z-00-DX1` | Prostate Adenocarcinoma (PRAD) |
| `TCGA-G2-A2ES-01Z-00-DX1` | Bladder Cancer (BLCA) |
| `TCGA-HU-A4GJ-01Z-00-DX1` | Stomach Adenocarcinoma (STAD) |
| `TCGA-EY-A1GK-01Z-00-DX1` | Endometrial Carcinoma (UCEC) |

**How It Works:**
1. Slides are downloaded from the GDC (Genomic Data Commons) API
2. Downloaded slides are cached locally to avoid re-downloading
3. Clinical metadata (sex, tissue site, cancer subtype) is extracted and pre-filled

**Authentication:**
- Public TCGA slides can be fetched without authentication
- Controlled-access data requires a GDC token (set via `GDC_TOKEN` environment variable)

**Cache Location:**
- Default: `~/.cache/mosaic/tcga_slides/`
- Override with `MOSAIC_TCGA_CACHE_DIR` environment variable

---

## Important Limitations

**Research Use Only:**
This tool is for research purposes only and not approved for clinical diagnosis.

**Model Limitations:**
- Predictions are probabilistic and may contain errors
- Performance varies by tissue type and slide quality
- Best results with standard H&E staining protocols
- Not validated for all rare cancer subtypes

**Quality Requirements:**
- Slides should have clear H&E staining
- Adequate tissue present (not mostly blank)
- Standard microscopy resolution

---

## Models & Technical Details

**Models Used:**
- **[CTransPath](https://github.com/Xiyue-Wang/TransPath)**: Used for filtering marker tiles (not the main analysis model)
- **[Optimus-0](https://huggingface.co/bioptimus/H-optimus-0)**: Vision-language foundation model (main feature extraction)
- **Aeon**: Cancer subtype classification
- **Paladin**: Biomarker prediction

**Learn More:**
- [Mosaic GitHub Repository](https://github.com/pathology-data-mining/mosaic)

---

## Login & User Storage
"""
            )

            if IS_HF_SPACES:
                gr.Markdown(
                    """
**Logging In:**
- Click **Sign in with HuggingFace** at the top of the page to sign in with your HuggingFace account
- Login uses HuggingFace OAuth — you will be redirected to HuggingFace to authorize access
- After logging in, the page reloads and unlocks additional features

**Features Available After Login:**
- **My Files** tab: View, download, and delete your previously uploaded slides
- **My Results** tab: View, download, and delete your past analysis results
- **Storage Usage**: See how much of your storage quota is used
- **Re-analyze Existing Slides**: Select a previously uploaded slide from the dropdown without re-uploading
"""
                )
            else:
                gr.Markdown(
                    """
**Local Mode:**
- Login is not required — all user storage features are available automatically
- Files are stored under a debug username (`local_user`) in a temporary directory

**User Storage Features:**
- **My Files** tab: View, download, and delete your previously uploaded slides
- **My Results** tab: View, download, and delete your past analysis results
- **Storage Usage**: See how much of your storage quota is used
- **Re-analyze Existing Slides**: Select a previously uploaded slide from the dropdown without re-uploading
"""
                )

            gr.Markdown(
                """
---

## Troubleshooting

**Common Issues:**
- **"Sex is required"**: Select Male or Female before analyzing
- **Slow processing**: Large slides take longer; be patient
- **Upload failed**: Check file format (.svs, .tiff, .tif only)
- **No results**: Ensure slide contains sufficient tissue
            """
            )

        with gr.Row():
            with gr.Column():
                # Input mode selection
                input_mode = gr.Radio(
                    choices=["Upload Files", "Fetch from TCGA"],
                    value="Upload Files",
                    label="Input Mode",
                    info="Choose how to provide the slide",
                )

                # File upload component (visible by default)
                input_slides = gr.File(
                    label="Upload H&E Whole Slide Image (SVS or TIFF format)",
                    file_types=[".svs", ".tiff", ".tif"],
                    file_count="multiple",
                )

                # Storage usage indicator (shown for logged-in users or local mode)
                # Initially visible in local mode, hidden on HF Spaces until login
                storage_usage_warning = gr.Markdown(visible=not IS_HF_SPACES)

                # Analyze Existing File option (shown for logged-in users or local mode)
                # Initially visible in local mode, hidden on HF Spaces until login
                with gr.Row(visible=not IS_HF_SPACES) as existing_file_row:
                    gr.Markdown("**Or select an existing slide:**")
                    existing_slides_dropdown = gr.Dropdown(
                        label="Previously Uploaded Slides",
                        choices=[],
                        value=None,
                        interactive=True,
                        multiselect=True,
                        info="Select one or more slides you've already uploaded to re-analyze",
                    )
                    refresh_existing_btn = gr.Button(
                        "🔄 Refresh List", size="sm", variant="secondary"
                    )

                # TCGA fetch components (hidden by default)
                with gr.Group(visible=False) as tcga_input_group:
                    tcga_id_input = gr.Textbox(
                        label="TCGA Slide ID",
                        placeholder="Enter UUID or TCGA barcode",
                        info="Supports UUID format or TCGA barcode format",
                    )
                    gr.Examples(
                        examples=[
                            ["TCGA-86-8074-01Z-00-DX1"],  # LUAD (Lung)
                            ["TCGA-AZ-4313-01Z-00-DX1"],  # COAD (Colon)
                            ["TCGA-EJ-7123-01Z-00-DX1"],  # PRAD (Prostate)
                            ["TCGA-G2-A2ES-01Z-00-DX1"],  # BLCA (Bladder)
                            ["TCGA-HU-A4GJ-01Z-00-DX1"],  # STAD (Stomach)
                            ["TCGA-EY-A1GK-01Z-00-DX1"],  # UCEC (Endometrial)
                        ],
                        inputs=tcga_id_input,
                        label="Example TCGA Slide IDs (click to use)",
                    )
                    with gr.Row():
                        fetch_button = gr.Button("Fetch Slide", variant="primary")
                    tcga_status = gr.Textbox(
                        label="Fetch Status",
                        interactive=False,
                        visible=False,
                    )

                # State to track fetched slide path
                tcga_slide_path_state = gr.State(None)

                site_dropdown = gr.Dropdown(
                    choices=["Primary", "Metastatic"],
                    label="Site Type",
                    value="Primary",
                    info="Primary tumor or metastatic site - affects biomarker interpretation",
                )
                sex_dropdown = gr.Dropdown(
                    choices=SEX_OPTIONS,
                    label="Sex",
                    value=None,
                    info="Required - Patient biological sex (used for sex-specific cancer analysis)",
                )
                tissue_site_dropdown = gr.Dropdown(
                    choices=get_tissue_sites(),
                    label="Tissue Site",
                    value="Unknown",
                    info="Anatomical location of the tissue sample (if known)",
                )
                cancer_subtype_dropdown = gr.Dropdown(
                    choices=[name for name in _pipeline.cancer_subtype_name_map.keys()],
                    label="Cancer Subtype",
                    value="Unknown",
                    info="Leave as 'Unknown' for automatic inference using the Aeon model",
                )
                # TODO: Enable IHC subtype dropdown in future work
                # ihc_subtype_dropdown = gr.Dropdown(
                #     choices=IHC_SUBTYPES,
                #     label="IHC Subtype (if applicable)",
                #     value="",
                #     visible=False,
                # )
                ihc_subtype_dropdown = gr.State("")  # Placeholder state for now
                seg_config_dropdown = gr.Dropdown(
                    choices=["Biopsy", "Resection", "TCGA"],
                    label="Segmentation Config",
                    value="Biopsy",
                    info="Biopsy: small samples | Resection: surgical specimens | TCGA: TCGA-compatible slides",
                )
                with gr.Row():
                    settings_input = gr.Dataframe(
                        headers=SETTINGS_COLUMNS,
                        label="Current Settings (editable for multiple slides)",
                        datatype=["str"] * len(SETTINGS_COLUMNS),
                        visible=False,
                        interactive=True,
                        static_columns="Slide",
                    )

                with gr.Row():
                    settings_csv = gr.File(
                        file_types=[".csv"],
                        label="Upload Settings CSV",
                        visible=False,
                    )

                with gr.Row():
                    clear_button = gr.Button("Clear")
                    analyze_button = gr.Button("Analyze", variant="primary")
            with gr.Column():
                gr.Markdown(
                    """
                    **Tissue Segmentation Masks**: Cyan outlines show detected tissue regions. Red fill indicates tissue area.
                    """
                )
                slide_masks = gr.Gallery(
                    label="Slide Masks (tissue segmentation)",
                    columns=3,
                    object_fit="contain",
                    height="auto",
                )
                aeon_output_table = gr.Dataframe(
                    headers=["Cancer Subtype", "Slide Name"],
                    label="Aeon — Cancer Subtype Predictions",
                    datatype=["str", "number"],
                    column_widths=["60%", "40%"],
                    wrap=True,
                    visible=False,
                )
                aeon_note = gr.Markdown(
                    "> **Confidence (0–1):** Higher = more likely the correct diagnosis.",
                    visible=False,
                )
                aeon_download_button = gr.DownloadButton(
                    "Download Aeon Results as CSV",
                    label="Download Aeon Results",
                    visible=False,
                )
                paladin_output_table = gr.Dataframe(
                    headers=["Slide", "Cancer Subtype", "Biomarker", "Prob. Present"],
                    label="Paladin — Biomarker Predictions",
                    datatype=["str", "str", "str", "number"],
                    column_widths=["20%", "25%", "45%", "10%"],
                    wrap=True,
                )
                paladin_note = gr.Markdown(
                    "> **Prob. Present (0–1):** Probability a genomic alteration is present. "
                    "Near 1.0 = likely present, near 0.0 = likely absent.",
                    visible=False,
                )
                paladin_download_button = gr.DownloadButton(
                    "Download Paladin Results as CSV",
                    label="Download Paladin Results",
                    visible=False,
                )

        def get_settings(
            files, site_type, sex, tissue_site, cancer_subtype, seg_config
        ):
            """Generate initial settings DataFrame from uploaded files and dropdown values."""
            if files is None:
                return pd.DataFrame()
            settings = []
            for file in files:
                filename = file.name if hasattr(file, "name") else file
                slide_name = filename.split("/")[-1]
                settings.append(
                    [
                        slide_name,
                        site_type,
                        sex if sex is not None else "",
                        tissue_site,
                        cancer_subtype,
                        seg_config,
                        Path(slide_name).stem,
                    ]
                )
            df = pd.DataFrame(settings, columns=SETTINGS_COLUMNS)
            return df

        def update_settings_column(settings_df, column_name, new_value):
            """Update a specific column in the settings DataFrame."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            # Create a copy to avoid modifying the original
            updated_df = settings_df.copy()
            # Convert None to empty string for display (especially for Sex column)
            if new_value is None:
                new_value = ""
            # Convert legacy "Unknown" sex values to empty string
            if column_name == "Sex" and new_value == "Unknown":
                new_value = ""
            updated_df[column_name] = new_value
            return updated_df

        # Handle input mode switching
        @input_mode.change(
            inputs=[input_mode],
            outputs=[
                input_slides,
                tcga_input_group,
                settings_input,
                settings_csv,
                existing_file_row,
            ],
        )
        def switch_input_mode(mode):
            """Switch between file upload and TCGA fetch modes."""
            if mode == "Upload Files":
                return (
                    gr.File(visible=True),
                    gr.Group(visible=False),
                    gr.Dataframe(visible=False),
                    gr.File(visible=False),
                    gr.Row(visible=True),  # Show existing files row in upload mode
                )
            else:
                return (
                    gr.File(visible=False),
                    gr.Group(visible=True),
                    gr.Dataframe(visible=False),
                    gr.File(visible=False),
                    gr.Row(visible=False),  # Hide existing files row in TCGA mode
                )

        # Load storage usage and show warnings
        def load_storage_usage(
            session_id: str = "",
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
        ):
            """Load and display storage usage with warnings."""
            from mosaic.ui.user_tabs import _get_username

            username, is_local = _get_username(request, profile, session_id)

            if not username:
                # Not logged in on HF Spaces - hide storage info
                return gr.Markdown(visible=False), gr.Row(visible=False)

            try:
                from mosaic.user_storage import get_storage_usage

                usage = get_storage_usage(username)
                quota_gb = usage.quota_bytes / (1024**3)
                used_gb = usage.total_bytes / (1024**3)
                percent = (
                    (usage.total_bytes / usage.quota_bytes * 100)
                    if usage.quota_bytes > 0
                    else 0
                )

                mode_indicator = " 🔧 [Local Debug]" if is_local else ""

                # Build warning message
                if percent >= 100:
                    warning_md = (
                        f"⛔ **Storage quota exceeded!** {used_gb:.2f} GB / {quota_gb:.0f} GB "
                        f"({percent:.0f}%){mode_indicator}\n\n"
                        f"Delete old files from the **My Files** tab before uploading."
                    )
                    warning_color = "red"
                elif percent >= 80:
                    warning_md = (
                        f"⚠️ **Storage nearly full:** {used_gb:.2f} GB / {quota_gb:.0f} GB "
                        f"({percent:.0f}%){mode_indicator}\n\n"
                        f"Consider deleting old files from the **My Files** tab."
                    )
                    warning_color = "orange"
                else:
                    warning_md = (
                        f"💾 **Storage:** {used_gb:.2f} GB / {quota_gb:.0f} GB "
                        f"({percent:.0f}%) | {usage.file_count} files{mode_indicator}"
                    )
                    warning_color = "gray"

                return gr.Markdown(value=warning_md, visible=True), gr.Row(visible=True)

            except Exception as e:
                logger.error(f"Failed to load storage usage: {e}")
                return gr.Markdown(visible=False), gr.Row(visible=False)

        # Load list of existing slides for re-analysis
        def load_existing_slides(
            session_id: str = "",
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
        ):
            """Load dropdown with user's existing slides."""
            from mosaic.ui.user_tabs import _get_username

            username, _ = _get_username(request, profile, session_id)

            if not username:
                return gr.Dropdown(choices=[], value=None)

            try:
                from mosaic.user_storage import list_user_slides

                slides = list_user_slides(username)

                if not slides:
                    return gr.Dropdown(
                        choices=[], value=None, info="No slides uploaded yet"
                    )

                # Create dropdown choices: display name → slide_id
                choices = [
                    (
                        f"{slide.original_filename} ({slide.size_bytes / (1024**2):.1f} MB)",
                        slide.slide_id,
                    )
                    for slide in slides
                ]

                return gr.Dropdown(
                    choices=choices,
                    value=None,
                    info=f"{len(slides)} slide(s) available for re-analysis",
                )

            except Exception as e:
                logger.error(f"Failed to load existing slides: {e}")
                return gr.Dropdown(choices=[], value=None)

        # Handle slide upload and save to user storage
        def handle_slide_upload(
            files,
            slide_ids,
            session_id: str = "",
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
        ):
            """Save uploaded slides to user storage and track slide IDs."""
            if not files:
                return gr.skip(), slide_ids or {}

            # Determine username via OAuthProfile (primary) or request (fallback)
            from mosaic.ui.user_tabs import _get_username

            username, _ = _get_username(request, profile, session_id)
            if not username:
                return gr.skip(), slide_ids or {}

            try:
                from mosaic.user_storage import save_uploaded_slide

                new_slide_ids = slide_ids.copy() if slide_ids else {}

                for file in files:
                    try:
                        # Save slide to user storage
                        slide_id, _ = save_uploaded_slide(username, file)

                        # Track mapping from filename to slide_id
                        filename = (
                            Path(file.name).name if hasattr(file, "name") else str(file)
                        )
                        new_slide_ids[filename] = slide_id

                        logger.info(
                            f"Saved slide {filename} with ID {slide_id} for user {username}"
                        )
                    except Exception as e:
                        # Best effort - don't fail upload if storage fails
                        logger.warning(
                            f"Failed to save slide {file} to user storage: {e}"
                        )

                return gr.skip(), new_slide_ids

            except Exception as e:
                logger.error(f"Error in slide upload handler: {e}")
                # Don't fail the upload, just log the error
                return gr.skip(), slide_ids or {}

        # Handle TCGA slide fetch
        def fetch_tcga_slide(tcga_id, seg_config, progress=gr.Progress()):
            """Fetch a slide from TCGA and pre-fill metadata."""
            if not tcga_id or not tcga_id.strip():
                raise gr.Error("Please enter a TCGA slide ID (UUID or barcode).")

            try:
                # Validate ID format first
                tcga_id = validate_slide_id(tcga_id)
                progress(0.1, desc="Validating slide ID...")

                # Determine file UUID for HF download
                from mosaic.tcga import is_valid_uuid, query_slide_uuid

                if is_valid_uuid(tcga_id):
                    file_uuid = tcga_id
                else:
                    # For barcodes, we need to query first to get UUID
                    file_uuid = query_slide_uuid(tcga_id)

                # Try to download cached results from HF Dataset (on demand)
                try:
                    download_results_from_hf(file_uuid)
                except Exception as e:
                    logger.warning(
                        f"Failed to download HF cache for {file_uuid}: {e}. "
                        f"Will analyze from scratch if no local cache exists."
                    )

                # Get tissue sites for metadata mapping
                tissue_sites = get_tissue_sites()

                # Fetch the slide (will use cache if available)
                progress(
                    0.2,
                    desc="Fetching slide from TCGA (this may take a while for large files)...",
                )

                def update_progress(downloaded, total):
                    if total > 0:
                        pct = 0.2 + (downloaded / total) * 0.7
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        progress(
                            pct, desc=f"Downloading: {size_mb:.1f} / {total_mb:.1f} MB"
                        )

                slide_path, file_uuid = fetch_slide(
                    tcga_id,
                    progress_callback=update_progress,
                )

                # Check if metadata is cached
                from mosaic.tcga import load_metadata_cache

                cached_metadata = load_metadata_cache(file_uuid)
                if cached_metadata is not None:
                    progress(0.9, desc="Loading cached metadata...")
                else:
                    progress(0.9, desc="Extracting metadata from TCGA...")

                # Get case metadata and convert to Mosaic values
                metadata = get_case_metadata(file_uuid, tissue_sites)

                progress(1.0, desc="Slide fetched successfully!")

                # Determine cancer subtype display value
                cancer_subtype_value = "Unknown"
                if metadata.cancer_subtype and metadata.cancer_subtype != "Unknown":
                    # Try to find the display name in the map
                    if (
                        metadata.cancer_subtype
                        in _pipeline.reversed_cancer_subtype_name_map
                    ):
                        cancer_subtype_value = (
                            _pipeline.reversed_cancer_subtype_name_map[
                                metadata.cancer_subtype
                            ]
                        )
                    else:
                        # Code not in map - keep as Unknown, log warning
                        logger.warning(
                            f"Cancer subtype code '{metadata.cancer_subtype}' not found in "
                            "available subtypes. Using 'Unknown'."
                        )

                status_msg = f"Slide fetched: {metadata.filename or slide_path.name}"
                if metadata.sex or metadata.tissue_site != "Unknown":
                    status_msg += f"\nPre-filled: Sex={metadata.sex or 'N/A'}, Site={metadata.tissue_site}, Subtype={cancer_subtype_value}"
                status_msg += (
                    "\nSegmentation Config: TCGA (auto-selected for TCGA slides)"
                )

                return (
                    str(slide_path),  # tcga_slide_path_state
                    metadata.site_type or "Primary",  # site_dropdown
                    metadata.sex,  # sex_dropdown
                    metadata.tissue_site or "Unknown",  # tissue_site_dropdown
                    cancer_subtype_value,  # cancer_subtype_dropdown
                    "TCGA",  # seg_config_dropdown - Auto-select TCGA config for TCGA slides
                    gr.Textbox(value=status_msg, visible=True),  # tcga_status
                    gr.Dataframe(value=None, visible=False),  # settings_input
                    gr.File(visible=False),  # settings_csv
                )

            except GDCInvalidIDError as e:
                raise gr.Error(str(e))
            except GDCNotFoundError as e:
                raise gr.Error(str(e))
            except GDCAccessDeniedError as e:
                raise gr.Error(str(e))
            except GDCError as e:
                raise gr.Error(f"Error fetching slide: {e}")
            except Exception as e:
                logger.exception("Unexpected error fetching TCGA slide")
                raise gr.Error(f"Unexpected error: {e}")

        fetch_button.click(
            fetch_tcga_slide,
            inputs=[tcga_id_input, seg_config_dropdown],
            outputs=[
                tcga_slide_path_state,
                site_dropdown,
                sex_dropdown,
                tissue_site_dropdown,
                cancer_subtype_dropdown,
                seg_config_dropdown,  # Include seg_config_dropdown to auto-select TCGA
                tcga_status,
                settings_input,
                settings_csv,
            ],
        )

        # Hidden textbox that carries the localStorage session ID into
        # every Gradio event handler that needs the visitor's identity.
        # On HF Spaces the js= parameter on demo.load() fills this from
        # localStorage; locally it stays empty and _get_username() falls
        # back to LOCAL_DEBUG_USERNAME.
        session_id_input = gr.Textbox(visible=False, elem_id="session_id_input")
        # Hidden flag set to "1" only by the demo.load js= parameter.
        # On WebSocket reconnections JS doesn't re-run, so this stays
        # at its default "".  The Python handler uses it to detect
        # reconnections and return gr.skip() to preserve UI state.
        _js_ran_flag = gr.Textbox(visible=False, value="")

        # Handle file uploads - regenerate entire settings table
        # Save uploaded slides to user storage (HF Spaces only)
        input_slides.upload(
            handle_slide_upload,
            inputs=[input_slides, slide_ids_state, session_id_input],
            outputs=[input_slides, slide_ids_state],
        ).then(
            # Refresh storage usage after upload
            load_storage_usage,
            inputs=[session_id_input],
            outputs=[storage_usage_warning, existing_file_row],
        ).then(
            # Refresh existing slides list after upload
            load_existing_slides,
            inputs=[session_id_input],
            outputs=[existing_slides_dropdown],
        )

        # Refresh existing slides dropdown
        refresh_existing_btn.click(
            load_existing_slides,
            inputs=[session_id_input],
            outputs=[existing_slides_dropdown],
        )

        # Handle selection of existing slide(s) for re-analysis
        def select_existing_slide(
            slide_ids,
            session_id: str = "",
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
        ):
            """When user selects existing slide(s), load them for analysis.

            Args:
                slide_ids: Single slide_id (str) or list of slide_ids (multiselect)
                session_id: Server-side session ID from localStorage
                request: Gradio request object
                profile: OAuthProfile for HF Spaces login

            Returns:
                gr.File update with selected slide(s)
            """
            if not slide_ids:
                return gr.skip()

            # Handle both single selection and multiselect
            if isinstance(slide_ids, str):
                slide_ids = [slide_ids]

            from mosaic.ui.user_tabs import _get_username

            username, _ = _get_username(request, profile, session_id)
            if not username:
                return gr.skip()

            try:
                from mosaic.user_storage import get_slide_path, _load_slide_metadata

                metadata = _load_slide_metadata(username)
                slide_paths = []

                for slide_id in slide_ids:
                    slide_path = get_slide_path(username, slide_id)
                    if slide_path is None:
                        logger.warning(
                            f"Slide {slide_id} not found for user {username}"
                        )
                        continue

                    # Create a symlink with the original filename so the File
                    # component displays the real name instead of the UUID storage name
                    if slide_id in metadata:
                        original_name = metadata[slide_id].original_filename
                        if slide_path.name != original_name:
                            tmp_dir = Path(tempfile.mkdtemp(prefix="mosaic_slide_"))
                            symlink_path = tmp_dir / original_name
                            symlink_path.symlink_to(slide_path)
                            slide_paths.append(str(symlink_path))
                        else:
                            slide_paths.append(str(slide_path))
                    else:
                        slide_paths.append(str(slide_path))

                if not slide_paths:
                    return gr.skip()

                return gr.File(value=slide_paths)

            except Exception as e:
                logger.error(f"Failed to load existing slides {slide_ids}: {e}")
                return gr.skip()

        existing_slides_dropdown.change(
            select_existing_slide,
            inputs=[existing_slides_dropdown, session_id_input],
            outputs=[input_slides],
        )

        @input_slides.change(
            inputs=[
                input_slides,
                site_dropdown,
                sex_dropdown,
                tissue_site_dropdown,
                cancer_subtype_dropdown,
                ihc_subtype_dropdown,
                seg_config_dropdown,
            ],
            outputs=[settings_input, settings_csv, ihc_subtype_dropdown],
        )
        def update_files(
            files, site_type, sex, tissue_site, cancer_subtype, ihc_subtype, seg_config
        ):
            """Handle file upload - regenerate settings table from scratch."""
            # has_ihc = "Breast" in cancer_subtype  # TODO: Enable when IHC dropdown is active
            if not files:
                return None, None, ""  # Return empty string for State component
            settings_df = get_settings(
                files,
                site_type,
                sex,
                tissue_site,
                cancer_subtype,
                seg_config,
            )
            # if settings_df is not None:
            #     has_ihc = any("Breast" in cs for cs in settings_df["Cancer Subtype"])
            visible = files and len(files) > 1
            return (
                gr.Dataframe(value=settings_df, visible=visible),
                gr.File(visible=visible),
                "",  # Return empty string for State component
            )

        # Handle individual dropdown changes - only update the relevant column
        @site_dropdown.change(
            inputs=[settings_input, site_dropdown],
            outputs=[settings_input],
        )
        def update_site_type(settings_df, site_type):
            """Update Site Type column when dropdown changes."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            updated_df = update_settings_column(settings_df, "Site Type", site_type)
            return gr.Dataframe(value=updated_df)

        @sex_dropdown.change(
            inputs=[settings_input, sex_dropdown],
            outputs=[settings_input],
        )
        def update_sex(settings_df, sex):
            """Update Sex column when dropdown changes."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            updated_df = update_settings_column(settings_df, "Sex", sex)
            return gr.Dataframe(value=updated_df)

        @tissue_site_dropdown.change(
            inputs=[settings_input, tissue_site_dropdown],
            outputs=[settings_input],
        )
        def update_tissue_site(settings_df, tissue_site):
            """Update Tissue Site column when dropdown changes."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            updated_df = update_settings_column(settings_df, "Tissue Site", tissue_site)
            return gr.Dataframe(value=updated_df)

        @cancer_subtype_dropdown.change(
            inputs=[settings_input, cancer_subtype_dropdown],
            outputs=[settings_input, ihc_subtype_dropdown],
        )
        def update_cancer_subtype(settings_df, cancer_subtype):
            """Update Cancer Subtype column when dropdown changes."""
            # has_ihc = "Breast" in cancer_subtype  # TODO: Enable when IHC dropdown is active
            if settings_df is None or len(settings_df) == 0:
                return settings_df, ""  # Return empty string for State component
            updated_df = update_settings_column(
                settings_df, "Cancer Subtype", cancer_subtype
            )
            return (
                gr.Dataframe(value=updated_df),
                "",
            )  # Return empty string for State component

        # TODO: Enable IHC subtype change handler in future work
        # @ihc_subtype_dropdown.change(
        #     inputs=[settings_input, ihc_subtype_dropdown],
        #     outputs=[settings_input],
        # )
        # def update_ihc_subtype(settings_df, ihc_subtype):
        #     """Update IHC Subtype column when dropdown changes."""
        #     if settings_df is None or len(settings_df) == 0:
        #         return settings_df
        #     updated_df = update_settings_column(settings_df, "IHC Subtype", ihc_subtype)
        #     return gr.Dataframe(value=updated_df)

        @seg_config_dropdown.change(
            inputs=[settings_input, seg_config_dropdown],
            outputs=[settings_input],
        )
        def update_seg_config(settings_df, seg_config):
            """Update Segmentation Config column when dropdown changes."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            updated_df = update_settings_column(
                settings_df, "Segmentation Config", seg_config
            )
            return gr.Dataframe(value=updated_df)

        @settings_csv.upload(
            inputs=[settings_csv],
            outputs=[settings_input],
        )
        def read_settings(file):
            if file is None:
                return None
            df = load_settings(file.name if hasattr(file, "name") else file)
            return gr.Dataframe(df, visible=True)

        def analyze_with_tcga_support(
            input_mode_val,
            slides,
            tcga_slide_path,
            settings_input_val,
            site_type,
            sex,
            tissue_site,
            cancer_subtype,
            ihc_subtype,
            seg_config,
            user_dir,
            slide_ids,
            session_id,
            progress=gr.Progress(track_tqdm=True),
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
        ):
            """Wrapper to handle both file upload and TCGA slide inputs."""
            # Determine the slides to analyze based on input mode
            if input_mode_val == "Fetch from TCGA":
                if not tcga_slide_path:
                    raise gr.Error(
                        "Please fetch a TCGA slide first using the 'Fetch Slide' button."
                    )
                # Create a mock file-like object for the TCGA slide
                slides = [Path(tcga_slide_path)]

                # Regenerate settings for the single TCGA slide to avoid issues
                # with stale settings from previously uploaded slides
                slide_name = Path(tcga_slide_path).name
                settings_input_val = pd.DataFrame(
                    [
                        [
                            slide_name,
                            site_type,
                            sex,
                            tissue_site,
                            cancer_subtype,
                            seg_config,
                            Path(tcga_slide_path).stem,
                        ]
                    ],
                    columns=SETTINGS_COLUMNS,
                )
            elif slides is None or len(slides) == 0:
                raise gr.Error("Please upload at least one slide.")

            # Call the original analyze_slides function
            yield from analyze_slides(
                slides,
                settings_input_val,
                site_type,
                sex,
                tissue_site,
                cancer_subtype,
                ihc_subtype,
                seg_config,
                user_dir,
                slide_ids=slide_ids,
                session_id=session_id,
                progress=progress,
                request=request,
                profile=profile,
            )

        analyze_button.click(
            analyze_with_tcga_support,
            inputs=[
                input_mode,
                input_slides,
                tcga_slide_path_state,
                settings_input,
                site_dropdown,
                sex_dropdown,
                tissue_site_dropdown,
                cancer_subtype_dropdown,
                ihc_subtype_dropdown,
                seg_config_dropdown,
                user_dir_state,
                slide_ids_state,
                session_id_input,
            ],
            outputs=[
                settings_input,
                slide_masks,
                aeon_output_table,
                aeon_note,
                aeon_download_button,
                paladin_output_table,
                paladin_note,
                paladin_download_button,
                user_dir_state,
                analyze_button,
                clear_button,
            ],
            queue=True,
            show_progress_on=paladin_output_table,
        )
        settings_input.change(
            lambda df: validate_settings(
                df,
                _pipeline.cancer_subtype_name_map,
                _pipeline.cancer_subtypes,
                _pipeline.reversed_cancer_subtype_name_map,
            ),
            inputs=[settings_input],
            outputs=[settings_input],
        )

        @clear_button.click(
            outputs=[
                input_slides,
                slide_masks,
                paladin_output_table,
                paladin_note,
                paladin_download_button,
                aeon_output_table,
                aeon_note,
                aeon_download_button,
                settings_input,
                settings_csv,
                tcga_id_input,
                tcga_status,
                tcga_slide_path_state,
                input_mode,
                tcga_input_group,
            ],
        )
        def clear_fn():
            return (
                None,  # input_slides
                None,  # slide_masks
                None,  # paladin_output_table
                gr.Markdown(visible=False),  # paladin_note
                gr.DownloadButton(visible=False),  # paladin_download_button
                gr.Dataframe(visible=False),  # aeon_output_table
                gr.Markdown(visible=False),  # aeon_note
                gr.DownloadButton(visible=False),  # aeon_download_button
                gr.Dataframe(visible=False),  # settings_input
                gr.File(visible=False),  # settings_csv
                "",  # tcga_id_input
                gr.Textbox(visible=False),  # tcga_status
                None,  # tcga_slide_path_state
                "Upload Files",  # input_mode - reset to default
                gr.Group(visible=False),  # tcga_input_group - hide TCGA inputs
            )

        demo.load(
            create_user_directory,
            inputs=[user_dir_state],
            outputs=[user_dir_state],
        )

        # Add My Files and My Results tabs for user storage
        # On HF Spaces: visible for logged-in users
        # Locally: always visible with "local_user" for debugging
        gr.Markdown("---")
        if IS_HF_SPACES:
            user_storage_header = gr.Markdown(
                "## User Storage (Logged-In Users Only)", visible=False
            )
            user_storage_info = gr.Markdown(
                "The tabs below are only accessible to logged-in users. "
                "Upload and view your slides and analysis results.",
                visible=False,
            )
        else:
            gr.Markdown("## User Storage (Local Debug Mode)")
            gr.Markdown(
                "🔧 **Debug Mode**: Files stored as user 'local_user' in `/tmp/mosaic_user_data/local_user/`"
            )

        # User storage tabs - initially hidden on HF Spaces, visible locally
        with gr.Tabs(visible=not IS_HF_SPACES) as user_storage_tabs:
            with gr.Tab("My Files"):
                from mosaic.ui.user_tabs import create_my_files_tab

                my_files_components = create_my_files_tab(session_id_input)

            with gr.Tab("My Results"):
                from mosaic.ui.user_tabs import create_my_results_tab

                my_results_components = create_my_results_tab(session_id_input)

        # On HF Spaces: demo.load(js=...) reads the session ID from
        # localStorage AND sets a "JS ran" flag.  Gradio re-fires
        # demo.load() on WebSocket reconnections WITHOUT re-executing
        # the JS — so the flag stays at its default "".  The Python
        # handler detects this and returns gr.skip() to preserve the
        # UI state instead of overwriting the logged-in display.
        if IS_HF_SPACES:
            _load_files = my_files_components["load_files"]
            _load_results = my_results_components["load_results"]

            _LS_JS = "() => [localStorage.getItem('mosaic_session_id') || '', '1']"

            def on_page_load(session_id_from_browser, js_ran):
                """Handle page load: check login, load files, load results.

                On WebSocket reconnections, Gradio re-fires demo.load()
                but does NOT re-execute the js= parameter.  The js_ran
                flag stays at its default "" on reconnections, allowing
                us to return gr.skip() and preserve the current UI state.
                """
                from mosaic.ui.oauth import get_user_by_session_id, _sessions

                # Reconnection detection: JS didn't run → skip all updates
                if js_ran != "1":
                    logger.debug("on_page_load: reconnection detected, skipping")
                    return gr.skip()

                sid_repr = (
                    session_id_from_browser[:8] + "..."
                    if session_id_from_browser
                    else "(empty)"
                )
                logger.info(
                    f"on_page_load: sid={sid_repr}, "
                    f"active_sessions={len(_sessions)}"
                )

                if not session_id_from_browser:
                    return _build_logged_out_result()

                userinfo = get_user_by_session_id(session_id_from_browser)

                if userinfo is not None:
                    username = userinfo.get("preferred_username", "User")
                    login_out = (
                        gr.update(
                            value=f"Signed in as **{username}**",
                            visible=True,
                        ),
                        gr.update(visible=True),  # logout_button
                        gr.update(visible=True),  # user_storage_tabs
                        gr.update(visible=True),  # user_storage_header
                        gr.update(visible=True),  # user_storage_info
                    )
                else:
                    return _build_logged_out_result()

                # Load storage usage + existing slides for the analysis page
                storage_out = load_storage_usage(session_id=session_id_from_browser)
                existing_out = load_existing_slides(session_id=session_id_from_browser)

                # Load user files and results tabs
                try:
                    files_out = _load_files(session_id=session_id_from_browser)
                except Exception as e:
                    logger.error(f"on_page_load: _load_files error: {e}")
                    files_out = (
                        f"**Storage:** Error: {e}",
                        [],
                        gr.Textbox(visible=False),
                        gr.File(visible=False),
                    )
                try:
                    results_out = _load_results(session_id=session_id_from_browser)
                except Exception as e:
                    logger.error(f"on_page_load: _load_results error: {e}")
                    results_out = (
                        [],
                        gr.Textbox(visible=False),
                        gr.File(visible=False),
                    )

                # login(5) + session_id(1) + storage(2) + existing(1)
                # + files(4) + results(3) = 16
                return (
                    *login_out,
                    session_id_from_browser,
                    *storage_out,
                    existing_out,
                    *files_out,
                    *results_out,
                )

            def _build_logged_out_result():
                """Build the full logged-out output tuple (16 elements)."""
                return (
                    gr.update(
                        value="[Sign in with HuggingFace](/api/auth/login) to save slides and results",
                        visible=True,
                    ),
                    gr.update(visible=False),  # logout_button
                    gr.update(visible=False),  # user_storage_tabs
                    gr.update(visible=False),  # user_storage_header
                    gr.update(visible=False),  # user_storage_info
                    "",  # session_id_input
                    gr.Markdown(visible=False),  # storage_usage_warning
                    gr.Row(visible=False),  # existing_file_row
                    gr.Dropdown(choices=[], value=None),  # existing_slides_dropdown
                    "**Storage:** Not logged in",
                    [],  # files_table
                    gr.Textbox(visible=False),
                    gr.File(visible=False),
                    [],  # results_table
                    gr.Textbox(visible=False),
                    gr.File(visible=False),
                )

            demo.load(
                on_page_load,
                inputs=[session_id_input, _js_ran_flag],
                outputs=[
                    # login state (5)
                    login_status_md,
                    logout_button,
                    user_storage_tabs,
                    user_storage_header,
                    user_storage_info,
                    # persist session_id for subsequent button clicks (1)
                    session_id_input,
                    # analysis page: storage + existing slides (3)
                    storage_usage_warning,
                    existing_file_row,
                    existing_slides_dropdown,
                    # files tab (4)
                    my_files_components["storage_usage"],
                    my_files_components["files_table"],
                    my_files_components["file_action_status"],
                    my_files_components["slide_download_file"],
                    # results tab (3)
                    my_results_components["results_table"],
                    my_results_components["result_action_status"],
                    my_results_components["result_download_file"],
                ],
                js=_LS_JS,
            )

            # Sign-out: clear localStorage and reload.
            # On reload, demo.load fires; JS reads empty localStorage
            # and returns ['', '1']; on_page_load sees empty session_id
            # and returns the logged-out state.
            logout_button.click(
                fn=None,
                inputs=None,
                js="() => { localStorage.removeItem('mosaic_session_id'); location.reload(); }",
            )

    # Use hardware-specific concurrency limit
    # T4 GPUs (16GB) can only handle one analysis at a time to prevent OOM
    # Higher-memory GPUs and ZeroGPU can handle multiple concurrent analyses
    demo.queue(max_size=10, default_concurrency_limit=DEFAULT_CONCURRENCY_LIMIT)

    # OAuth routes for HF Spaces (sdk:docker):
    # Custom OAuth routes (/api/auth/login, /api/auth/callback, /api/auth/logout)
    # are mounted via the lifespan protocol in create_oauth_lifespan(). This
    # ensures routes survive launch() recreating the app. Identity is read from
    # our server-side session (mosaic_auth cookie), not gr.OAuthProfile, which
    # returns the Space owner's identity on Docker SDK Spaces.

    # Register cleanup handler for graceful shutdown
    import atexit

    def cleanup_on_exit():
        """Signal background thread to stop and log telemetry shutdown."""
        _pipeline.request_shutdown()
        logger.info("Shutdown signal sent to background model download thread")

        # Log app shutdown for telemetry
        tracker.log_app_shutdown()

        # Upload telemetry to HF Dataset if configured
        if os.environ.get("MOSAIC_TELEMETRY_HF_REPO"):
            tracker.upload_to_hf()

    atexit.register(cleanup_on_exit)

    # Start heartbeat thread for telemetry (every 5 minutes)
    # Also uploads telemetry on each heartbeat to ensure data is saved before HF Spaces sleep
    def heartbeat_worker():
        while not _pipeline.is_shutdown_requested():
            time.sleep(tracker.config.heartbeat_interval_sec)
            tracker.log_heartbeat()
            # Upload telemetry on each heartbeat to avoid data loss when HF Spaces goes to sleep
            # (atexit handlers may not run when the container is frozen/killed)
            tracker.upload_to_hf()

    heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    heartbeat_thread.start()

    # On HF Spaces, mount custom OAuth routes via the lifespan protocol.
    # Gradio's launch() recreates the app, but the lifespan handler receives
    # the final App instance, so routes added there survive.
    launch_kwargs = {}
    if IS_HF_SPACES:
        from mosaic.ui.oauth import create_oauth_lifespan

        launch_kwargs["app_kwargs"] = {"lifespan": create_oauth_lifespan}

    demo.launch(
        server_name=server_name,
        share=share,
        server_port=server_port,
        show_error=True,
        favicon_path=current_dir / "favicon.svg",
        **launch_kwargs,
    )

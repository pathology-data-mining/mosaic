"""My Files and My Results tabs for HuggingFace Spaces user storage.

This module provides UI components for browsing and managing user-uploaded
slides and analysis results in ephemeral storage.
"""

import gradio as gr
import pandas as pd
from loguru import logger
from pathlib import Path

from mosaic.hardware import IS_HF_SPACES
from mosaic.user_storage import (
    list_user_slides,
    get_storage_usage,
    delete_user_slide,
    get_slide_path,
    find_slide_by_name,
    DEFAULT_QUOTA_BYTES,
)
from mosaic.user_results import (
    list_user_results,
    load_analysis_results,
    delete_analysis_results,
    create_results_zip,
)

# Local debug username for non-HF Spaces environments
LOCAL_DEBUG_USERNAME = "local_user"


def _get_username(request: gr.Request = None, profile=None) -> tuple[str, bool]:
    """Get username for storage operations.

    On HF Spaces, uses OAuthProfile from gr.LoginButton() to identify
    the logged-in user. Does NOT use request.username, which returns
    the Space owner's username rather than the visitor's.

    Returns:
        Tuple of (username, is_local_mode)
        - In HF Spaces: (profile.username, False) if logged in
        - Locally: (LOCAL_DEBUG_USERNAME, True)
        - HF Spaces not logged in: (None, False)
    """
    if IS_HF_SPACES:
        # On HF Spaces, only OAuthProfile reliably identifies the logged-in user.
        # request.username returns the Space owner's username (not the visitor's),
        # so we must NOT fall back to it.
        if profile is not None:
            try:
                username = getattr(profile, "username", None)
                if username:
                    return (username, False)
            except Exception:
                pass

        # Fallback: check server-side session store (custom OAuth flow)
        if request is not None:
            try:
                from mosaic.ui.oauth import get_user_from_server_session

                userinfo = get_user_from_server_session(request)
                if userinfo:
                    username = userinfo.get("preferred_username")
                    if username:
                        return (username, False)
            except Exception:
                pass

        return (None, False)
    else:
        # Local mode - use debug username
        return (LOCAL_DEBUG_USERNAME, True)


def create_my_files_tab():
    """Create the My Files tab UI for managing uploaded slides.

    Returns:
        Dictionary of UI components for event handlers
    """
    with gr.Column():
        gr.Markdown("### My Uploaded Files")
        gr.Markdown(
            "View and manage your uploaded slides. Files are stored temporarily "
            "and cleared when the instance restarts."
        )

        # Storage usage indicator
        storage_usage_md = gr.Markdown("**Storage:** Loading...")

        # Files table
        files_table = gr.Dataframe(
            headers=["Slide ID", "Filename", "Size (MB)", "Upload Date", "# Analyses"],
            datatype=["str", "str", "number", "str", "number"],
            label="Uploaded Slides",
            interactive=False,
            column_widths=["3fr", "3fr", "1fr", "2fr", "1fr"],
        )

        # Action buttons
        with gr.Row():
            refresh_files_btn = gr.Button("Refresh", variant="secondary")
            selected_slide_id = gr.Textbox(
                label="Selected Slide ID",
                placeholder="Click a row in the table to select a slide",
            )

        with gr.Row():
            download_slide_btn = gr.Button("Download Original", variant="secondary")
            delete_slide_btn = gr.Button("Delete Slide", variant="stop")

        # Status messages
        file_action_status = gr.Textbox(
            label="Status",
            interactive=False,
            visible=False,
        )

        # Download output
        slide_download_file = gr.File(label="Download", visible=False)

        def load_files(request: gr.Request = None, profile: gr.OAuthProfile = None):
            """Load user's uploaded files."""
            username, is_local = _get_username(request, profile)

            if not username:
                return (
                    "**Storage:** Not logged in (login required on HF Spaces)",
                    [],
                    gr.Textbox(visible=False),
                    gr.File(visible=False),
                )

            try:
                # Add debug indicator for local mode
                mode_indicator = " 🔧 **[Local Debug Mode]**" if is_local else ""

                # Get storage usage
                usage = get_storage_usage(username)
                quota_gb = usage.quota_bytes / (1024**3)
                used_gb = usage.total_bytes / (1024**3)
                percent = (
                    (usage.total_bytes / usage.quota_bytes * 100)
                    if usage.quota_bytes > 0
                    else 0
                )

                storage_md = (
                    f"**Storage:** {used_gb:.2f} GB / {quota_gb:.0f} GB ({percent:.1f}%) | "
                    f"{usage.file_count} files{mode_indicator}"
                )

                if percent > 80:
                    storage_md += " ⚠️ **Nearly full!**"

                # Get files list
                slides = list_user_slides(username)

                if not slides:
                    return (
                        storage_md,
                        [],
                        gr.Textbox(visible=False),
                        gr.File(visible=False),
                    )

                # Get result counts
                from mosaic.user_results import list_user_results as list_results

                files_data = []
                for slide in slides:
                    # Count results for this slide
                    results = list_results(username, slide_id=slide.slide_id)

                    files_data.append(
                        [
                            Path(slide.original_filename).stem,
                            slide.original_filename,
                            round(slide.size_bytes / (1024**2), 2),  # Convert to MB
                            slide.upload_time[:19],  # Truncate to datetime
                            len(results),
                        ]
                    )

                return (
                    storage_md,
                    files_data,
                    gr.Textbox(visible=False),
                    gr.File(visible=False),
                )

            except Exception as e:
                logger.error(f"Failed to load user files: {e}")
                return (
                    "**Storage:** Error loading files",
                    [],
                    gr.Textbox(value=f"Error: {str(e)}", visible=True),
                    gr.File(visible=False),
                )

        def download_slide(
            slide_id_or_name,
            request: gr.Request = None,
            profile: gr.OAuthProfile = None,
        ):
            """Download the original slide file."""
            username, _ = _get_username(request, profile)

            if not username or not slide_id_or_name:
                return gr.File(visible=False), gr.Textbox(
                    value="Not logged in or no slide selected", visible=True
                )

            try:
                # Resolve name/stem to UUID if needed
                resolved_id = find_slide_by_name(username, slide_id_or_name)
                if resolved_id is None:
                    return gr.File(visible=False), gr.Textbox(
                        value="Slide not found", visible=True
                    )

                slide_path = get_slide_path(username, resolved_id)

                if slide_path is None:
                    return gr.File(visible=False), gr.Textbox(
                        value="Slide not found", visible=True
                    )

                return gr.File(value=str(slide_path), visible=True), gr.Textbox(
                    value="Download ready", visible=True
                )

            except Exception as e:
                logger.error(f"Failed to download slide {slide_id_or_name}: {e}")
                return gr.File(visible=False), gr.Textbox(
                    value=f"Error: {str(e)}", visible=True
                )

        def delete_slide(
            slide_id_or_name,
            request: gr.Request = None,
            profile: gr.OAuthProfile = None,
        ):
            """Delete a slide and all associated results."""
            username, _ = _get_username(request, profile)

            if not username or not slide_id_or_name:
                return gr.Textbox(
                    value="Not logged in or no slide selected", visible=True
                )

            try:
                # Resolve name/stem to UUID if needed
                resolved_id = find_slide_by_name(username, slide_id_or_name)
                if resolved_id is None:
                    return gr.Textbox(value="Slide not found", visible=True)

                success = delete_user_slide(username, resolved_id)

                if success:
                    return gr.Textbox(
                        value=f"Deleted slide '{slide_id_or_name}' and all associated results",
                        visible=True,
                    )
                else:
                    return gr.Textbox(value="Slide not found", visible=True)

            except Exception as e:
                logger.error(f"Failed to delete slide {slide_id_or_name}: {e}")
                return gr.Textbox(value=f"Error: {str(e)}", visible=True)

        # Wire up events
        refresh_files_btn.click(
            load_files,
            inputs=None,
            outputs=[
                storage_usage_md,
                files_table,
                file_action_status,
                slide_download_file,
            ],
        )

        download_slide_btn.click(
            download_slide,
            inputs=[selected_slide_id],
            outputs=[slide_download_file, file_action_status],
        )

        delete_slide_btn.click(
            delete_slide,
            inputs=[selected_slide_id],
            outputs=[file_action_status],
        ).then(
            load_files,  # Refresh after delete
            inputs=None,
            outputs=[
                storage_usage_md,
                files_table,
                file_action_status,
                slide_download_file,
            ],
        )

        def on_file_row_select(evt: gr.SelectData, table_data):
            """Auto-fill slide ID when a row is clicked."""
            if table_data is not None and len(table_data) > 0:
                row = evt.index[0]
                return str(table_data.iloc[row, 0])  # Slide ID column
            return ""

        files_table.select(
            on_file_row_select,
            inputs=[files_table],
            outputs=[selected_slide_id],
        )

    return {
        "storage_usage": storage_usage_md,
        "files_table": files_table,
        "refresh_btn": refresh_files_btn,
        "load_files": load_files,
        "file_action_status": file_action_status,
        "slide_download_file": slide_download_file,
    }


def create_my_results_tab():
    """Create the My Results tab UI for browsing analysis results.

    Returns:
        Dictionary of UI components for event handlers
    """
    with gr.Column():
        gr.Markdown("### My Analysis Results")
        gr.Markdown(
            "View and download results from previous analyses. "
            "Results are stored temporarily and cleared when the instance restarts."
        )

        # Results table
        results_table = gr.Dataframe(
            headers=["Analysis ID", "Slide Name", "Date", "Cancer Subtype", "Settings"],
            datatype=["str", "str", "str", "str", "str"],
            label="Analysis Results",
            interactive=False,
        )

        # Action buttons
        with gr.Row():
            refresh_results_btn = gr.Button("Refresh", variant="secondary")
            selected_analysis_id = gr.Textbox(
                label="Selected Analysis ID",
                placeholder="Click a row in the table to select a result",
            )

        with gr.Row():
            view_result_btn = gr.Button("View Details", variant="secondary")
            download_zip_btn = gr.Button("Download ZIP", variant="primary")
            delete_result_btn = gr.Button("Delete Result", variant="stop")

        # Result detail section (hidden by default)
        with gr.Accordion("Result Details", open=False) as result_details:
            result_metadata_md = gr.Markdown("No result selected")
            result_mask_img = gr.Image(label="Tissue Segmentation Mask", visible=False)
            result_aeon_df = gr.Dataframe(
                label="Cancer Subtype Predictions", visible=False
            )
            result_paladin_df = gr.Dataframe(
                label="Biomarker Predictions", visible=False
            )

        # Status and download
        result_action_status = gr.Textbox(
            label="Status", interactive=False, visible=False
        )
        result_download_file = gr.File(label="Download", visible=False)

        def load_results(request: gr.Request = None, profile: gr.OAuthProfile = None):
            """Load user's analysis results."""
            username, _ = _get_username(request, profile)

            if not username:
                return [], gr.Textbox(visible=False), gr.File(visible=False)

            try:
                results = list_user_results(username)

                if not results:
                    return [], gr.Textbox(visible=False), gr.File(visible=False)

                results_data = []
                for result in results:
                    # Format settings as short string
                    settings_str = f"{result.settings.get('site_type', 'N/A')} | {result.settings.get('sex', 'N/A')}"

                    results_data.append(
                        [
                            result.analysis_id,
                            result.slide_name,
                            result.timestamp[:19],  # Truncate to datetime
                            result.cancer_subtype or "Unknown",
                            settings_str,
                        ]
                    )

                return results_data, gr.Textbox(visible=False), gr.File(visible=False)

            except Exception as e:
                logger.error(f"Failed to load user results: {e}")
                return (
                    [],
                    gr.Textbox(value=f"Error: {str(e)}", visible=True),
                    gr.File(visible=False),
                )

        def view_result_details(
            analysis_id, request: gr.Request = None, profile: gr.OAuthProfile = None
        ):
            """Load and display result details."""
            username, _ = _get_username(request, profile)

            if not username or not analysis_id:
                return (
                    gr.Accordion(open=False),
                    "No result selected",
                    gr.Image(visible=False),
                    gr.Dataframe(visible=False),
                    gr.Dataframe(visible=False),
                    gr.Textbox(visible=False),
                )

            try:
                loaded = load_analysis_results(username, analysis_id)

                if loaded is None:
                    return (
                        gr.Accordion(open=False),
                        "Result not found",
                        gr.Image(visible=False),
                        gr.Dataframe(visible=False),
                        gr.Dataframe(visible=False),
                        gr.Textbox(value="Result not found", visible=True),
                    )

                metadata, mask, aeon_df, paladin_df = loaded

                # Format aeon results with human-readable cancer subtype names
                if aeon_df is not None and len(aeon_df) > 0:
                    from mosaic.ui.utils import get_oncotree_code_name

                    aeon_df = aeon_df.reset_index()
                    # Rename index column to "Cancer Subtype" if needed
                    if aeon_df.columns[0] != "Cancer Subtype":
                        aeon_df = aeon_df.rename(
                            columns={aeon_df.columns[0]: "Cancer Subtype"}
                        )
                    # Add human-readable names
                    aeon_df["Cancer Subtype"] = [
                        f"{get_oncotree_code_name(code)} ({code})"
                        for code in aeon_df["Cancer Subtype"]
                    ]
                    # Round confidence scores
                    for col in aeon_df.columns[1:]:
                        aeon_df[col] = pd.to_numeric(
                            aeon_df[col], errors="coerce"
                        ).round(3)

                # Round paladin scores
                if paladin_df is not None and "Score" in paladin_df.columns:
                    paladin_df["Score"] = pd.to_numeric(
                        paladin_df["Score"], errors="coerce"
                    ).round(3)

                # Format metadata
                settings_md = "### Analysis Settings\n\n"
                for key, value in metadata.settings.items():
                    settings_md += f"- **{key}**: {value}\n"

                # Format cancer subtype with readable name
                subtype_display = metadata.cancer_subtype or "Unknown"
                if subtype_display and subtype_display != "Unknown":
                    from mosaic.ui.utils import get_oncotree_code_name

                    name = get_oncotree_code_name(subtype_display)
                    if name and name != subtype_display:
                        subtype_display = f"{name} ({subtype_display})"

                metadata_text = (
                    f"### {metadata.slide_name}\n\n"
                    f"**Analysis ID:** {metadata.analysis_id}\n\n"
                    f"**Date:** {metadata.timestamp[:19]}\n\n"
                    f"**Predicted Subtype:** {subtype_display}\n\n"
                    f"{settings_md}"
                )

                return (
                    gr.Accordion(open=True),
                    metadata_text,
                    (
                        gr.Image(value=mask, visible=True)
                        if mask
                        else gr.Image(visible=False)
                    ),
                    (
                        gr.Dataframe(value=aeon_df, visible=True)
                        if aeon_df is not None
                        else gr.Dataframe(visible=False)
                    ),
                    (
                        gr.Dataframe(value=paladin_df, visible=True)
                        if paladin_df is not None
                        else gr.Dataframe(visible=False)
                    ),
                    gr.Textbox(visible=False),
                )

            except Exception as e:
                logger.error(f"Failed to load result details {analysis_id}: {e}")
                return (
                    gr.Accordion(open=False),
                    "Error loading result",
                    gr.Image(visible=False),
                    gr.Dataframe(visible=False),
                    gr.Dataframe(visible=False),
                    gr.Textbox(value=f"Error: {str(e)}", visible=True),
                )

        def download_result_zip(
            analysis_id, request: gr.Request = None, profile: gr.OAuthProfile = None
        ):
            """Create and download ZIP of result files."""
            username, _ = _get_username(request, profile)

            if not username or not analysis_id:
                return gr.File(visible=False), gr.Textbox(
                    value="Not logged in or no result selected", visible=True
                )

            try:
                zip_path = create_results_zip(username, analysis_id)

                if zip_path is None:
                    return gr.File(visible=False), gr.Textbox(
                        value="Result not found", visible=True
                    )

                return gr.File(value=str(zip_path), visible=True), gr.Textbox(
                    value="ZIP ready for download", visible=True
                )

            except Exception as e:
                logger.error(f"Failed to create ZIP for {analysis_id}: {e}")
                return gr.File(visible=False), gr.Textbox(
                    value=f"Error: {str(e)}", visible=True
                )

        def delete_result(
            analysis_id, request: gr.Request = None, profile: gr.OAuthProfile = None
        ):
            """Delete an analysis result."""
            username, _ = _get_username(request, profile)

            if not username or not analysis_id:
                return gr.Textbox(
                    value="Not logged in or no result selected", visible=True
                )

            try:
                success = delete_analysis_results(username, analysis_id)

                if success:
                    return gr.Textbox(
                        value=f"Deleted result {analysis_id}", visible=True
                    )
                else:
                    return gr.Textbox(value="Result not found", visible=True)

            except Exception as e:
                logger.error(f"Failed to delete result {analysis_id}: {e}")
                return gr.Textbox(value=f"Error: {str(e)}", visible=True)

        # Wire up events
        refresh_results_btn.click(
            load_results,
            inputs=None,
            outputs=[results_table, result_action_status, result_download_file],
        )

        view_result_btn.click(
            view_result_details,
            inputs=[selected_analysis_id],
            outputs=[
                result_details,
                result_metadata_md,
                result_mask_img,
                result_aeon_df,
                result_paladin_df,
                result_action_status,
            ],
        )

        download_zip_btn.click(
            download_result_zip,
            inputs=[selected_analysis_id],
            outputs=[result_download_file, result_action_status],
        )

        delete_result_btn.click(
            delete_result,
            inputs=[selected_analysis_id],
            outputs=[result_action_status],
        ).then(
            load_results,  # Refresh after delete
            inputs=None,
            outputs=[results_table, result_action_status, result_download_file],
        )

        def on_result_row_select(evt: gr.SelectData, table_data):
            """Auto-fill analysis ID when a row is clicked."""
            if table_data is not None and len(table_data) > 0:
                row = evt.index[0]
                return str(table_data.iloc[row, 0])  # Analysis ID column
            return ""

        results_table.select(
            on_result_row_select,
            inputs=[results_table],
            outputs=[selected_analysis_id],
        )

    return {
        "results_table": results_table,
        "refresh_btn": refresh_results_btn,
        "load_results": load_results,
        "result_action_status": result_action_status,
        "result_download_file": result_download_file,
    }

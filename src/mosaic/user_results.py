"""User analysis results management for HuggingFace Spaces.

This module provides ephemeral storage for analysis results, linked to uploaded
slides. Results are cleared on instance restart (no persistence).

Storage structure:
    /tmp/mosaic_user_data/{username}/
        results/
            {analysis_id}/
                metadata.json       # Result metadata
                slide_mask.png      # Tissue segmentation
                aeon_results.csv    # Cancer subtype predictions
                paladin_results.csv # Biomarker predictions
"""

import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from PIL import Image

from mosaic.user_storage import get_user_storage_dir


@dataclass
class ResultMetadata:
    """Metadata for an analysis result.

    Attributes:
        analysis_id: Unique identifier for this analysis
        slide_id: ID of the slide that was analyzed
        slide_name: Original slide filename
        timestamp: ISO timestamp of analysis
        settings: Dict of analysis settings (site_type, cancer_subtype, etc.)
        cancer_subtype: Predicted cancer subtype (for quick display)
    """

    analysis_id: str
    slide_id: str
    slide_name: str
    timestamp: str
    settings: dict
    cancer_subtype: Optional[str] = None


def _get_results_dir(username: str, analysis_id: str) -> Path:
    """Get the directory for a specific analysis result.

    Args:
        username: Username
        analysis_id: Analysis ID

    Returns:
        Path to results directory
    """
    user_dir = get_user_storage_dir(username)
    return user_dir / "results" / analysis_id


def save_analysis_results(
    username: str,
    analysis_id: str,
    slide_id: str,
    slide_name: str,
    settings: dict,
    aeon_results: Optional[pd.DataFrame],
    paladin_results: Optional[pd.DataFrame],
    slide_mask: Optional[Image.Image],
) -> bool:
    """Save analysis results for a user.

    Best-effort operation: logs errors but doesn't raise exceptions.

    Args:
        username: Username
        analysis_id: Unique analysis ID
        slide_id: ID of the analyzed slide
        slide_name: Original slide filename
        settings: Dict of analysis settings
        aeon_results: Aeon predictions DataFrame
        paladin_results: Paladin predictions DataFrame
        slide_mask: Tissue segmentation mask

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        results_dir = _get_results_dir(username, analysis_id)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Extract cancer subtype from aeon results (top prediction)
        cancer_subtype = None
        if aeon_results is not None and len(aeon_results) > 0:
            # Aeon results are sorted by confidence, first row is top prediction
            cancer_subtype = (
                aeon_results.index[0] if hasattr(aeon_results, "index") else None
            )

        # Save metadata
        metadata = ResultMetadata(
            analysis_id=analysis_id,
            slide_id=slide_id,
            slide_name=slide_name,
            timestamp=datetime.utcnow().isoformat(),
            settings=settings,
            cancer_subtype=cancer_subtype,
        )
        metadata_path = results_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Save aeon results (with index - "Cancer Subtype")
        if aeon_results is not None:
            aeon_path = results_dir / "aeon_results.csv"
            aeon_results.to_csv(aeon_path, index=True)

        # Save paladin results (no index)
        if paladin_results is not None:
            paladin_path = results_dir / "paladin_results.csv"
            paladin_results.to_csv(paladin_path, index=False)

        # Save slide mask
        if slide_mask is not None:
            mask_path = results_dir / "slide_mask.png"
            slide_mask.save(mask_path)

        logger.info(
            f"Saved analysis results for user {username}, "
            f"analysis {analysis_id}, slide {slide_id}"
        )
        return True

    except Exception as e:
        logger.error(
            f"Failed to save analysis results for user {username}, "
            f"analysis {analysis_id}: {e}"
        )
        return False


def load_analysis_results(
    username: str, analysis_id: str
) -> Optional[
    tuple[ResultMetadata, Image.Image | None, pd.DataFrame | None, pd.DataFrame | None]
]:
    """Load analysis results for a user.

    Args:
        username: Username
        analysis_id: Analysis ID

    Returns:
        Tuple of (metadata, slide_mask, aeon_results, paladin_results) or None if not found
    """
    try:
        results_dir = _get_results_dir(username, analysis_id)
        if not results_dir.exists():
            return None

        # Load metadata
        metadata_path = results_dir / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            metadata = ResultMetadata(**json.load(f))

        # Load mask
        mask_path = results_dir / "slide_mask.png"
        slide_mask = Image.open(mask_path) if mask_path.exists() else None

        # Load aeon results (restore index)
        aeon_path = results_dir / "aeon_results.csv"
        aeon_results = (
            pd.read_csv(aeon_path, index_col=0) if aeon_path.exists() else None
        )

        # Load paladin results
        paladin_path = results_dir / "paladin_results.csv"
        paladin_results = pd.read_csv(paladin_path) if paladin_path.exists() else None

        return metadata, slide_mask, aeon_results, paladin_results

    except Exception as e:
        logger.error(
            f"Failed to load analysis results for user {username}, "
            f"analysis {analysis_id}: {e}"
        )
        return None


def list_user_results(
    username: str, slide_id: Optional[str] = None
) -> list[ResultMetadata]:
    """List all analysis results for a user.

    Returns results sorted by timestamp (newest first).

    Args:
        username: Username
        slide_id: Optional slide ID to filter by

    Returns:
        List of ResultMetadata objects
    """
    user_dir = get_user_storage_dir(username)
    results_base = user_dir / "results"

    if not results_base.exists():
        return []

    results = []
    for result_dir in results_base.iterdir():
        if not result_dir.is_dir():
            continue

        metadata_path = result_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, "r") as f:
                metadata = ResultMetadata(**json.load(f))

            # Filter by slide_id if provided
            if slide_id is not None and metadata.slide_id != slide_id:
                continue

            results.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {result_dir}: {e}")
            continue

    # Sort by timestamp, newest first
    results.sort(key=lambda r: r.timestamp, reverse=True)
    return results


def delete_analysis_results(username: str, analysis_id: str) -> bool:
    """Delete analysis results.

    Args:
        username: Username
        analysis_id: Analysis ID

    Returns:
        True if deleted successfully, False if not found
    """
    try:
        results_dir = _get_results_dir(username, analysis_id)
        if not results_dir.exists():
            return False

        shutil.rmtree(results_dir)
        logger.info(f"Deleted analysis results {analysis_id} for user {username}")
        return True

    except Exception as e:
        logger.error(
            f"Failed to delete analysis results {analysis_id} for user {username}: {e}"
        )
        return False


def delete_results_for_slide(username: str, slide_id: str) -> int:
    """Delete all analysis results for a specific slide.

    Args:
        username: Username
        slide_id: Slide ID

    Returns:
        Number of results deleted
    """
    results = list_user_results(username, slide_id=slide_id)
    deleted_count = 0

    for result in results:
        if delete_analysis_results(username, result.analysis_id):
            deleted_count += 1

    if deleted_count > 0:
        logger.info(
            f"Deleted {deleted_count} analysis results for slide {slide_id}, "
            f"user {username}"
        )

    return deleted_count


def create_results_zip(username: str, analysis_id: str) -> Optional[Path]:
    """Create a ZIP file containing all result files.

    Args:
        username: Username
        analysis_id: Analysis ID

    Returns:
        Path to ZIP file, or None if failed
    """
    try:
        results_dir = _get_results_dir(username, analysis_id)
        if not results_dir.exists():
            return None

        # Create ZIP in parent directory
        zip_path = results_dir.parent / f"{analysis_id}_results.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add all files in results directory
            for file_path in results_dir.iterdir():
                if file_path.is_file():
                    zipf.write(file_path, arcname=file_path.name)

        logger.info(f"Created results ZIP for {analysis_id}, user {username}")
        return zip_path

    except Exception as e:
        logger.error(
            f"Failed to create results ZIP for {analysis_id}, user {username}: {e}"
        )
        return None

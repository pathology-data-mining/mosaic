"""Model downloading and initialization for Mosaic.

Downloads models and metadata from HuggingFace Hub and initialises the
global cancer-subtype mappings used by both the CLI and UI.
"""

import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download
from loguru import logger

from mosaic.data_directory import set_data_directory
from mosaic.ui.pipeline import set_cancer_subtype_maps
from mosaic.ui.utils import format_cancer_subtype


def build_cancer_subtype_maps(cancer_subtypes: list) -> tuple[dict, dict]:
    """Build the cancer-subtype display-name ↔ OncoTree-code maps.

    Args:
        cancer_subtypes: List of OncoTree code strings (e.g. ``["LUAD", "BRCA"]``).

    Returns:
        ``(cancer_subtype_name_map, reversed_cancer_subtype_name_map)`` where
        *cancer_subtype_name_map* maps display strings → codes and
        *reversed_cancer_subtype_name_map* is the inverse.
    """
    cancer_subtype_name_map = {"Unknown": "UNK"}
    cancer_subtype_name_map.update(
        {format_cancer_subtype(code): code for code in cancer_subtypes}
    )
    reversed_cancer_subtype_name_map = {
        value: key for key, value in cancer_subtype_name_map.items()
    }
    return cancer_subtype_name_map, reversed_cancer_subtype_name_map


def download_metadata():
    """Load metadata files from local data/ directory or download from HuggingFace.

    Fast loading of small files needed for UI initialization.
    Prioritizes local files in data/ directory (bundled with Docker image).

    Returns:
        tuple: (cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes)
    """
    # Check if metadata exists in local data/ directory (Docker deployment)
    local_data_dir = Path(__file__).parent.parent.parent / "data"
    local_model_map = local_data_dir / "paladin_model_map.csv"

    if local_model_map.exists():
        logger.info(f"Using local metadata from: {local_data_dir}")
        cache_dir = local_data_dir
    else:
        # Fall back to downloading from HuggingFace Hub
        logger.info("Downloading metadata from HuggingFace Hub...")
        cache_dir = Path(
            snapshot_download(
                repo_id="PDM-Group/paladin-aeon-models",
                allow_patterns=[
                    "*.csv",  # Model maps and metadata
                    "tissue_site_*",  # Tissue site mappings
                    "metadata/*",  # Metadata files (including target_dict.tsv)
                ],
            )
        )
        logger.info(f"Metadata downloaded to: {cache_dir}")

    # Set the data directory for other modules to use
    set_data_directory(str(cache_dir))

    model_map = pd.read_csv(Path(cache_dir) / "paladin_model_map.csv")
    cancer_subtypes = model_map["cancer_subtype"].unique().tolist()
    cancer_subtype_name_map, reversed_cancer_subtype_name_map = (
        build_cancer_subtype_maps(cancer_subtypes)
    )

    # Set the global maps in the UI module
    set_cancer_subtype_maps(
        cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
    )

    return cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes


def download_core_models():
    """Download core models: CTransPath, Optimus, Aeon, marker classifier.

    Priority models needed for basic analysis pipeline.
    Also downloads metadata files to ensure everything is in the same cache directory.
    """
    logger.info("Downloading core models (CTransPath, Optimus, Aeon)...")

    # Download CTransPath, Aeon, marker classifier, and metadata files
    # Include metadata to ensure all files are in same HF cache directory
    cache_dir = snapshot_download(
        repo_id="PDM-Group/paladin-aeon-models",
        allow_patterns=[
            "ctranspath.pth",
            "aeon_model.pkl",
            "marker_classifier.pkl",
            "*.csv",  # Model maps and metadata
            "tissue_site_*",  # Tissue site mappings
            "metadata/*",  # Metadata files (including target_dict.tsv)
        ],
    )
    # Update data directory to ensure models are found
    set_data_directory(str(cache_dir))
    logger.info(f"Core models downloaded to: {cache_dir}")

    # Download Optimus model
    logger.info("Downloading Optimus model (~1GB)...")
    from mussel.models import ModelType, get_model_factory

    optimus_factory = get_model_factory(ModelType.OPTIMUS)
    _ = optimus_factory.get_model(
        model_path="hf-hub:bioptimus/H-optimus-0",
        use_gpu=False,
        gpu_device_id=None,
    )
    logger.info("✓ Core models downloaded successfully")


def download_paladin_models():
    """Download all Paladin models.

    Lower priority models downloaded after core models.
    Excludes optimus.pkl since we download Optimus from bioptimus/H-optimus-0 instead.
    """
    logger.info("📦 Downloading all Paladin models...")
    cache_dir = snapshot_download(
        repo_id="PDM-Group/paladin-aeon-models",
        allow_patterns=[
            "paladin/*.pkl",  # All Paladin model files in paladin/ subdirectory
        ],
    )
    # Update data directory to ensure models are found
    set_data_directory(str(cache_dir))
    logger.info(f"✅ Paladin models downloaded successfully to: {cache_dir}")


def download_and_process_models():
    """Download all models and initialize cancer subtype mappings.

    Downloads metadata, core models, and Paladin models synchronously.

    Returns:
        tuple: (cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes)
    """
    result = download_metadata()
    download_core_models()
    download_paladin_models()
    return result

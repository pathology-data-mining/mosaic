"""Mosaic command-line interface and entry point.

This module provides the main CLI for the Mosaic application, handling:
- Model downloading and initialization
- Single slide processing
- Batch slide processing from CSV
- Launching the Gradio web interface
"""

from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download
from loguru import logger

from mosaic.data_directory import set_data_directory
from mosaic.ui import launch_gradio
from mosaic.ui.pipeline import set_cancer_subtype_maps
from mosaic.ui.utils import (
    get_oncotree_code_name,
    load_settings,
    validate_settings,
    IHC_SUBTYPES,
    SETTINGS_COLUMNS,
    SEX_OPTIONS,
)
from mosaic.analysis import (
    analyze_slide,
    extract_slide_features,
    run_inference_from_features,
)
from mosaic.model_manager import load_all_models


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
    cancer_subtype_name_map = {"Unknown": "UNK"}
    cancer_subtype_name_map.update(
        {f"{get_oncotree_code_name(code)} ({code})": code for code in cancer_subtypes}
    )
    reversed_cancer_subtype_name_map = {
        value: key for key, value in cancer_subtype_name_map.items()
    }

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


def main():
    """Main entry point for the Mosaic application.

    Parses command-line arguments and routes to the appropriate mode:
    - Single slide processing (--slide-path)
    - Batch processing (--slide-csv)
    - Web interface (default, no slide arguments)

    Command-line arguments control analysis parameters like site type,
    cancer subtype, segmentation configuration, and output directory.
    """
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Server name for Gradio app"
    )
    parser.add_argument(
        "--server-port", type=int, default=None, help="Server port for Gradio app"
    )
    parser.add_argument(
        "--share", action="store_true", help="Share Gradio app publicly"
    )
    parser.add_argument(
        "--slide-csv",
        type=str,
        help="CSV file with slide settings (for batch processing), see README for format",
    )
    parser.add_argument(
        "--slide-path",
        type=str,
        help="Path to a single slide (for single slide processing), not used if --slide-csv is provided",
    )
    parser.add_argument(
        "--site-type",
        type=str,
        choices=["Primary", "Metastatic"],
        default="Primary",
        help="Site type of the slide (for single slide processing)",
    )
    parser.add_argument(
        "--sex",
        type=str,
        choices=SEX_OPTIONS,
        default=None,
        help="Sex of the patient (required for single slide processing)",
    )
    parser.add_argument(
        "--tissue-site",
        type=str,
        default="Unknown",
        help="Tissue site of the slide (for single slide processing)",
    )
    parser.add_argument(
        "--cancer-subtype",
        type=str,
        default="Unknown",
        help="Cancer subtype of the slide (for single slide processing), use 'Unknown' to infer with Aeon",
    )
    parser.add_argument(
        "--ihc-subtype",
        type=str,
        choices=IHC_SUBTYPES,
        default="",
        help="IHC subtype if cancer subtype is breast (for single slide processing)",
    )
    parser.add_argument(
        "--segmentation-config",
        type=str,
        choices=["Biopsy", "Resection", "TCGA"],
        default="Biopsy",
        help="Segmentation configuration (for single slide processing)",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to save output results"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for feature extraction",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip downloading models from HuggingFace (assumes models are already cached)",
    )
    parser.add_argument(
        "--download-models-only",
        action="store_true",
        help="Download models from HuggingFace and exit without running analysis",
    )
    args = parser.parse_args()
    if args.debug:
        logger.add("debug.log", level="DEBUG")
        logger.debug("Debug logging enabled")

    # Handle model download options
    if getattr(args, "download_models_only", False):
        logger.info("Downloading models from HuggingFace...")
        download_and_process_models()
        logger.info("✓ Model download complete. Exiting.")
        return

    # Determine if we're launching the UI or doing CLI processing
    is_launching_ui = not args.slide_path and not args.slide_csv

    if not getattr(args, "skip_model_download", False):
        if is_launching_ui:
            # UI mode: download only metadata synchronously (fast, local in Docker)
            # All models download in background to start Gradio immediately
            (
                cancer_subtype_name_map,
                reversed_cancer_subtype_name_map,
                cancer_subtypes,
            ) = download_metadata()
            # Core and Paladin models will be downloaded in background by launch_gradio()
        else:
            # CLI mode: download everything synchronously
            (
                cancer_subtype_name_map,
                reversed_cancer_subtype_name_map,
                cancer_subtypes,
            ) = download_and_process_models()
    else:
        logger.info("Skipping model download, using cached models...")
        # Load cancer subtype mappings from cached data
        from mosaic.data_directory import get_data_directory

        cache_dir = get_data_directory()
        model_map = pd.read_csv(Path(cache_dir) / "paladin_model_map.csv")
        cancer_subtypes = model_map["cancer_subtype"].unique().tolist()
        cancer_subtype_name_map = {"Unknown": "UNK"}
        cancer_subtype_name_map.update(
            {
                f"{get_oncotree_code_name(code)} ({code})": code
                for code in cancer_subtypes
            }
        )
        reversed_cancer_subtype_name_map = {
            value: key for key, value in cancer_subtype_name_map.items()
        }
        # Set the global maps in the UI module
        set_cancer_subtype_maps(
            cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
        )

    if args.slide_path and not args.slide_csv:
        # Single slide processing mode
        if not args.output_dir:
            raise ValueError("Please provide --output-dir to save results")
        if not args.sex:
            raise ValueError(
                "Please provide --sex (Male or Female) for single slide processing"
            )
        settings_df = pd.DataFrame(
            [
                [
                    args.slide_path,
                    args.site_type,
                    args.sex,
                    args.tissue_site,
                    args.cancer_subtype,
                    # args.ihc_subtype,  # Not yet in use
                    args.segmentation_config,
                    Path(args.slide_path).stem,
                ]
            ],
            columns=SETTINGS_COLUMNS,
        )
        settings_df = validate_settings(
            settings_df,
            cancer_subtype_name_map,
            cancer_subtypes,
            reversed_cancer_subtype_name_map,
        )
        slide_mask, aeon_results, paladin_results = analyze_slide(
            args.slide_path,
            args.segmentation_config,
            args.site_type,
            args.sex,
            args.tissue_site,
            args.cancer_subtype,
            cancer_subtype_name_map,
            args.ihc_subtype,
            num_workers=args.num_workers,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        slide_name = Path(args.slide_path).stem
        if slide_mask is not None:
            mask_path = output_dir / f"{slide_name}_mask.png"
            slide_mask.save(mask_path)
            logger.info(f"Saved slide mask to {mask_path}")
        if aeon_results is not None:
            aeon_output_path = output_dir / f"{slide_name}_aeon_results.csv"
            aeon_results.reset_index().to_csv(aeon_output_path, index=False)
            logger.info(f"Saved Aeon results to {aeon_output_path}")
        if paladin_results is not None and len(paladin_results) > 0:
            paladin_output_path = output_dir / f"{slide_name}_paladin_results.csv"
            paladin_results.to_csv(paladin_output_path, index=False)
            logger.info(f"Saved Paladin results to {paladin_output_path}")
    elif args.slide_csv:
        if not args.output_dir:
            raise ValueError("Please provide --output-dir to save results")
        # Batch processing mode with optimized model loading

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and validate settings
        settings_df = load_settings(args.slide_csv)
        settings_df = validate_settings(
            settings_df,
            cancer_subtype_name_map,
            cancer_subtypes,
            reversed_cancer_subtype_name_map,
        )

        slides = settings_df["Slide"].tolist()

        # Build sample groups: sample_id → list of (idx, slide_path, row)
        sample_groups: dict = {}
        for idx in range(len(slides)):
            row = settings_df.iloc[idx]
            sid = str(row.get("Sample ID", "")) or Path(str(row["Slide"])).stem
            sample_groups.setdefault(sid, []).append((idx, slides[idx], row))

        is_multi_slide = any(len(v) > 1 for v in sample_groups.values())
        logger.info(
            f"Processing {len(slides)} slides → {len(sample_groups)} samples"
            f"{' (multi-slide mode)' if is_multi_slide else ''}"
        )

        # Load models once for the entire batch
        model_cache = load_all_models(use_gpu=True, aggressive_memory_mgmt=None)

        all_slide_masks = []
        all_aeon_results = []  # list of (sample_id, aeon_df)
        all_paladin_results = []

        try:
            for sample_id, slide_list in sample_groups.items():
                if len(slide_list) == 1:
                    # Single-slide sample: use existing analyze_slide path
                    idx, slide_path, row = slide_list[0]
                    slide_name = row["Slide"]
                    slide_stem = f"{idx:03d}_{Path(slide_name).stem}"
                    logger.info(f"[{sample_id}] 1 slide: {slide_name}")

                    slide_mask, aeon_results, paladin_results = analyze_slide(
                        slide_path=slide_path,
                        seg_config=row["Segmentation Config"],
                        site_type=row["Site Type"],
                        sex=row["Sex"],
                        tissue_site=row.get("Tissue Site", "Unknown"),
                        cancer_subtype=row["Cancer Subtype"],
                        cancer_subtype_name_map=cancer_subtype_name_map,
                        ihc_subtype=row.get("IHC Subtype", ""),
                        num_workers=args.num_workers,
                        progress=lambda frac=None, desc=None: None,
                        request=None,
                        model_cache=model_cache,
                    )

                    if slide_mask is not None:
                        all_slide_masks.append((slide_mask, slide_stem))
                    if aeon_results is not None:
                        all_aeon_results.append((sample_id, aeon_results))
                    if paladin_results is not None:
                        paladin_results.insert(
                            0, "Sample", pd.Series([sample_id] * len(paladin_results))
                        )
                        all_paladin_results.append(paladin_results)

                else:
                    # Multi-slide sample: 2-phase extraction + aggregation
                    from mussel.utils.feature_extract import aggregate_sample_features
                    from mosaic.inference.data import encode_sex, encode_tissue_site

                    logger.info(f"[{sample_id}] {len(slide_list)} slides — aggregating")

                    per_slide_features = []
                    per_slide_coords = []
                    per_sample_ids = []

                    for sub_idx, (idx, slide_path, row) in enumerate(slide_list):
                        slide_name = row["Slide"]
                        slide_stem = f"{idx:03d}_{Path(slide_name).stem}"
                        logger.info(
                            f"  [{sub_idx + 1}/{len(slide_list)}] Extracting: {slide_name}"
                        )

                        slide_mask, features, coords = extract_slide_features(
                            slide_path=slide_path,
                            seg_config=row["Segmentation Config"],
                            num_workers=args.num_workers,
                            progress=lambda frac=None, desc=None: None,
                            model_cache=model_cache,
                        )

                        if features is None:
                            logger.warning(f"No tissue in {slide_name}, skipping")
                            continue

                        per_slide_features.append(features)
                        per_slide_coords.append(coords)
                        per_sample_ids.append(sample_id)
                        if slide_mask is not None:
                            all_slide_masks.append((slide_mask, slide_stem))

                    if not per_slide_features:
                        logger.warning(f"No usable slides for {sample_id}, skipping")
                        continue

                    # Aggregate features
                    agg_dict = aggregate_sample_features(
                        per_slide_features, per_slide_coords, per_sample_ids
                    )
                    agg_features, _ = agg_dict[sample_id]
                    logger.info(f"  Aggregated: {agg_features.shape[0]} tiles")

                    # Run inference
                    first_row = slide_list[0][2]
                    sex_idx_val = (
                        encode_sex(first_row["Sex"]) if first_row.get("Sex") else None
                    )
                    ts_idx_val = encode_tissue_site(
                        first_row.get("Tissue Site", "Unknown")
                    )

                    aeon_results, paladin_results = run_inference_from_features(
                        features=agg_features,
                        site_type=first_row["Site Type"],
                        sex_idx=sex_idx_val,
                        tissue_site_idx=ts_idx_val,
                        cancer_subtype=first_row["Cancer Subtype"],
                        cancer_subtype_name_map=cancer_subtype_name_map,
                        model_cache=model_cache,
                        num_workers=args.num_workers,
                        progress=lambda frac=None, desc=None: None,
                    )

                    if aeon_results is not None:
                        all_aeon_results.append((sample_id, aeon_results))
                    if paladin_results is not None:
                        paladin_results.insert(
                            0, "Sample", pd.Series([sample_id] * len(paladin_results))
                        )
                        all_paladin_results.append(paladin_results)

        finally:
            logger.info("Cleaning up model cache")
            model_cache.cleanup()

        # Save slide masks
        for slide_mask, slide_stem in all_slide_masks:
            mask_path = output_dir / f"{slide_stem}_mask.png"
            slide_mask.save(mask_path)
            logger.info(f"Saved slide mask to {mask_path}")

        # Save per-sample Aeon results
        for sample_id, aeon_results in all_aeon_results:
            aeon_output_path = output_dir / f"{sample_id}_aeon_results.csv"
            aeon_results.reset_index().to_csv(aeon_output_path, index=False)
            logger.info(f"Saved Aeon results to {aeon_output_path}")

        # Save per-sample Paladin results
        if all_paladin_results:
            combined_paladin = pd.concat(all_paladin_results, ignore_index=True)
            for sid in combined_paladin["Sample"].unique():
                sample_paladin = combined_paladin[combined_paladin["Sample"] == sid]
                paladin_output_path = output_dir / f"{sid}_paladin_results.csv"
                sample_paladin.to_csv(paladin_output_path, index=False)
                logger.info(f"Saved Paladin results to {paladin_output_path}")

        # Save combined results
        if all_aeon_results:
            combined_aeon_results = pd.concat(
                [df for _, df in all_aeon_results], axis=1
            )
            combined_aeon_results.reset_index(inplace=True)
            cancer_subtype_names = [
                f"{get_oncotree_code_name(code)} ({code})"
                for code in combined_aeon_results["Cancer Subtype"]
            ]
            combined_aeon_results["Cancer Subtype"] = cancer_subtype_names
            combined_aeon_output_path = output_dir / "combined_aeon_results.csv"
            combined_aeon_results.to_csv(combined_aeon_output_path, index=False)
            logger.info(f"Saved combined Aeon results to {combined_aeon_output_path}")
        if all_paladin_results:
            combined_paladin_results = pd.concat(all_paladin_results, ignore_index=True)
            cancer_subtype_names = [
                f"{get_oncotree_code_name(code)} ({code})"
                for code in combined_paladin_results["Cancer Subtype"]
            ]
            combined_paladin_results["Cancer Subtype"] = cancer_subtype_names
            combined_paladin_output_path = output_dir / "combined_paladin_results.csv"
            combined_paladin_results.to_csv(combined_paladin_output_path, index=False)
            logger.info(
                f"Saved combined Paladin results to {combined_paladin_output_path}"
            )

    else:
        launch_gradio(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
        )


if __name__ == "__main__":
    main()

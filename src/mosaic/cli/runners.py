"""CLI execution routines for single-slide and batch slide processing."""

import pandas as pd
from argparse import Namespace
from pathlib import Path
from loguru import logger

from mosaic.settings import SETTINGS_COLUMNS, load_settings, normalize_settings
from mosaic.analysis import (
    analyze_slide,
    extract_slide_features,
    run_inference_from_features,
)
from mosaic.model_manager import load_all_models
from mosaic.ui.utils import get_oncotree_code_name


def _cancer_subtype_requires_aeon(cancer_subtype: str | None) -> bool:
    return cancer_subtype in ["Unknown", None, ""]


def _validate_sex_for_aeon(cancer_subtype: str | None, sex: str | None, scope: str) -> None:
    if _cancer_subtype_requires_aeon(cancer_subtype) and not sex:
        raise ValueError(
            f"Sex is required for {scope} when Cancer Subtype is Unknown "
            "(Aeon inference required). Please provide Male or Female."
        )


def run_single_slide(
    args: Namespace,
    cancer_subtype_name_map: dict,
    cancer_subtypes: list,
    reversed_cancer_subtype_name_map: dict,
) -> None:
    """Run analysis on a single slide specified via CLI arguments.

    Args:
        args: Parsed CLI arguments (must include slide_path, output_dir, sex, etc.).
        cancer_subtype_name_map: Mapping from display name to OncoTree code.
        cancer_subtypes: List of valid OncoTree codes.
        reversed_cancer_subtype_name_map: Mapping from OncoTree code to display name.

    Raises:
        ValueError: If --output-dir is not provided, or if Aeon inference is
            required but ``--sex`` is missing.
    """
    if not args.output_dir:
        raise ValueError("Please provide --output-dir to save results")
    _validate_sex_for_aeon(
        args.cancer_subtype, args.sex, "single-slide processing"
    )

    settings_df = pd.DataFrame(
        [
            [
                args.slide_path,
                args.site_type,
                args.sex,
                args.tissue_site,
                args.cancer_subtype,
                args.segmentation_config,
                Path(args.slide_path).stem,
            ]
        ],
        columns=SETTINGS_COLUMNS,
    )
    normalized, warnings = normalize_settings(
        settings_df,
        cancer_subtype_name_map,
        cancer_subtypes,
        reversed_cancer_subtype_name_map,
    )
    for warning in warnings:
        logger.warning(warning)

    normalized_row = normalized.iloc[0]

    slide_mask, aeon_results, paladin_results = analyze_slide(
        args.slide_path,
        args.segmentation_config,
        args.site_type,
        normalized_row["Sex"],
        normalized_row["Tissue Site"],
        normalized_row["Cancer Subtype"],
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


def run_batch(
    args: Namespace,
    cancer_subtype_name_map: dict,
    cancer_subtypes: list,
    reversed_cancer_subtype_name_map: dict,
) -> None:
    """Run batch analysis on slides specified in a CSV file.

    Loads models once and reuses them across all slides for efficiency.
    Supports multi-slide sample aggregation when multiple rows share a Sample ID.

    Args:
        args: Parsed CLI arguments (must include slide_csv, output_dir, num_workers).
        cancer_subtype_name_map: Mapping from display name to OncoTree code.
        cancer_subtypes: List of valid OncoTree codes.
        reversed_cancer_subtype_name_map: Mapping from OncoTree code to display name.

    Raises:
        ValueError: If --output-dir is not provided.
    """
    if not args.output_dir:
        raise ValueError("Please provide --output-dir to save results")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate settings
    settings_df = load_settings(args.slide_csv)
    normalized, warnings = normalize_settings(
        settings_df,
        cancer_subtype_name_map,
        cancer_subtypes,
        reversed_cancer_subtype_name_map,
    )
    for warning in warnings:
        logger.warning(warning)
    settings_df = normalized

    for _, row in settings_df.iterrows():
        _validate_sex_for_aeon(
            row["Cancer Subtype"], row.get("Sex"), f"slide '{row['Slide']}'"
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
                _process_single_slide_in_batch(
                    sample_id,
                    slide_list,
                    args,
                    cancer_subtype_name_map,
                    model_cache,
                    all_slide_masks,
                    all_aeon_results,
                    all_paladin_results,
                )
            else:
                _process_multi_slide_sample(
                    sample_id,
                    slide_list,
                    args,
                    cancer_subtype_name_map,
                    model_cache,
                    all_slide_masks,
                    all_aeon_results,
                    all_paladin_results,
                )
    finally:
        logger.info("Cleaning up model cache")
        model_cache.cleanup()

    _save_batch_outputs(
        output_dir,
        all_slide_masks,
        all_aeon_results,
        all_paladin_results,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _process_single_slide_in_batch(
    sample_id,
    slide_list,
    args,
    cancer_subtype_name_map,
    model_cache,
    all_slide_masks,
    all_aeon_results,
    all_paladin_results,
):
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


def _process_multi_slide_sample(
    sample_id,
    slide_list,
    args,
    cancer_subtype_name_map,
    model_cache,
    all_slide_masks,
    all_aeon_results,
    all_paladin_results,
):
    from mussel.utils.feature_extract import aggregate_sample_features
    from mosaic.inference.data import encode_sex, encode_tissue_site

    logger.info(f"[{sample_id}] {len(slide_list)} slides — aggregating")

    per_slide_features = []
    per_slide_coords = []
    per_sample_ids = []

    for sub_idx, (idx, slide_path, row) in enumerate(slide_list):
        slide_name = row["Slide"]
        slide_stem = f"{idx:03d}_{Path(slide_name).stem}"
        logger.info(f"  [{sub_idx + 1}/{len(slide_list)}] Extracting: {slide_name}")

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
        return

    # Aggregate features across slides
    agg_dict = aggregate_sample_features(
        per_slide_features, per_slide_coords, per_sample_ids
    )
    agg_features, _ = agg_dict[sample_id]
    logger.info(f"  Aggregated: {agg_features.shape[0]} tiles")

    # Run inference on aggregated features
    first_row = slide_list[0][2]
    sex_idx_val = encode_sex(first_row["Sex"]) if first_row.get("Sex") else None
    ts_idx_val = encode_tissue_site(first_row.get("Tissue Site", "Unknown"))

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


def _save_batch_outputs(
    output_dir, all_slide_masks, all_aeon_results, all_paladin_results
):
    """Persist all batch outputs to disk."""
    for slide_mask, slide_stem in all_slide_masks:
        mask_path = output_dir / f"{slide_stem}_mask.png"
        slide_mask.save(mask_path)
        logger.info(f"Saved slide mask to {mask_path}")

    for sample_id, aeon_results in all_aeon_results:
        aeon_output_path = output_dir / f"{sample_id}_aeon_results.csv"
        aeon_results.reset_index().to_csv(aeon_output_path, index=False)
        logger.info(f"Saved Aeon results to {aeon_output_path}")

    if all_paladin_results:
        combined_paladin = pd.concat(all_paladin_results, ignore_index=True)
        for sid in combined_paladin["Sample"].unique():
            sample_paladin = combined_paladin[combined_paladin["Sample"] == sid]
            paladin_output_path = output_dir / f"{sid}_paladin_results.csv"
            sample_paladin.to_csv(paladin_output_path, index=False)
            logger.info(f"Saved Paladin results to {paladin_output_path}")

    if all_aeon_results:
        combined_aeon_results = pd.concat([df for _, df in all_aeon_results], axis=1)
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
        logger.info(f"Saved combined Paladin results to {combined_paladin_output_path}")

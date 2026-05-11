"""CLI execution routines for single-slide and batch slide processing."""

import pandas as pd
from argparse import Namespace
from pathlib import Path
from loguru import logger

from mosaic.settings import (
    SETTINGS_COLUMNS,
    COL_SLIDE,
    COL_SAMPLE_ID,
    COL_SEG_CONFIG,
    load_settings,
    normalize_settings,
    row_to_slide_kwargs,
)
from mosaic.analysis import (
    analyze_slide,
    extract_slide_features,
    run_inference_from_features,
)
from mosaic.model_manager import load_all_models
from mosaic.ui.utils import expand_cancer_subtype_column


_NOOP_PROGRESS = lambda frac=None, desc=None: None  # noqa: E731


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
        ValueError: If --output-dir or --sex are not provided.
    """
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

    slides = settings_df[COL_SLIDE].tolist()

    # Build sample groups: sample_id → list of (idx, slide_path, row)
    sample_groups: dict = {}
    for idx in range(len(slides)):
        row = settings_df.iloc[idx]
        sid = str(row.get(COL_SAMPLE_ID, "")) or Path(str(row[COL_SLIDE])).stem
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


def _accumulate_results(
    sample_id,
    aeon_results,
    paladin_results,
    all_aeon_results,
    all_paladin_results,
) -> None:
    """Append non-None inference results to the shared batch accumulators."""
    if aeon_results is not None:
        all_aeon_results.append((sample_id, aeon_results))
    if paladin_results is not None:
        paladin_results.insert(
            0, "Sample", pd.Series([sample_id] * len(paladin_results))
        )
        all_paladin_results.append(paladin_results)


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
    slide_name = row[COL_SLIDE]
    slide_stem = f"{idx:03d}_{Path(slide_name).stem}"
    logger.info(f"[{sample_id}] 1 slide: {slide_name}")

    slide_mask, aeon_results, paladin_results = analyze_slide(
        slide_path=slide_path,
        **row_to_slide_kwargs(row),
        cancer_subtype_name_map=cancer_subtype_name_map,
        num_workers=args.num_workers,
        progress=_NOOP_PROGRESS,
        request=None,
        model_cache=model_cache,
    )

    if slide_mask is not None:
        all_slide_masks.append((slide_mask, slide_stem))
    _accumulate_results(
        sample_id, aeon_results, paladin_results, all_aeon_results, all_paladin_results
    )


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
        slide_name = row[COL_SLIDE]
        slide_stem = f"{idx:03d}_{Path(slide_name).stem}"
        logger.info(f"  [{sub_idx + 1}/{len(slide_list)}] Extracting: {slide_name}")

        slide_mask, features, coords = extract_slide_features(
            slide_path=slide_path,
            seg_config=row[COL_SEG_CONFIG],
            num_workers=args.num_workers,
            progress=_NOOP_PROGRESS,
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
    kw = row_to_slide_kwargs(first_row)
    sex_idx_val = encode_sex(kw["sex"]) if kw["sex"] else None
    ts_idx_val = encode_tissue_site(kw["tissue_site"])

    aeon_results, paladin_results = run_inference_from_features(
        features=agg_features,
        site_type=kw["site_type"],
        sex_idx=sex_idx_val,
        tissue_site_idx=ts_idx_val,
        cancer_subtype=kw["cancer_subtype"],
        cancer_subtype_name_map=cancer_subtype_name_map,
        model_cache=model_cache,
        num_workers=args.num_workers,
        progress=_NOOP_PROGRESS,
    )

    _accumulate_results(
        sample_id, aeon_results, paladin_results, all_aeon_results, all_paladin_results
    )


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
        expand_cancer_subtype_column(combined_aeon_results)
        combined_aeon_output_path = output_dir / "combined_aeon_results.csv"
        combined_aeon_results.to_csv(combined_aeon_output_path, index=False)
        logger.info(f"Saved combined Aeon results to {combined_aeon_output_path}")

    if all_paladin_results:
        combined_paladin_results = pd.concat(all_paladin_results, ignore_index=True)
        expand_cancer_subtype_column(combined_paladin_results)
        combined_paladin_output_path = output_dir / "combined_paladin_results.csv"
        combined_paladin_results.to_csv(combined_paladin_output_path, index=False)
        logger.info(f"Saved combined Paladin results to {combined_paladin_output_path}")

"""Performance benchmark for batch processing optimization.

This script compares the performance of:
1. Sequential single-slide processing (old method)
2. Batch processing with model caching (new method)

Usage:
    python tests/benchmark_batch_performance.py --slides slide1.svs slide2.svs slide3.svs
    python tests/benchmark_batch_performance.py --slide-csv test_slides.csv
"""

import argparse
import time
import pandas as pd
from pathlib import Path
import torch
from loguru import logger

from mosaic.analysis import analyze_slide
from mosaic.batch_analysis import analyze_slides_batch
from mosaic.ui.utils import load_settings, validate_settings


def benchmark_sequential_processing(
    slides, settings_df, cancer_subtype_name_map, num_workers
):
    """Benchmark traditional sequential processing (models loaded per slide)."""
    logger.info("=" * 80)
    logger.info("BENCHMARKING: Sequential Processing (OLD METHOD)")
    logger.info("=" * 80)

    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    results = []
    for idx, (slide_path, (_, row)) in enumerate(zip(slides, settings_df.iterrows())):
        logger.info(f"Processing slide {idx + 1}/{len(slides)}: {slide_path}")

        slide_start = time.time()

        slide_mask, aeon_results, paladin_results = analyze_slide(
            slide_path=slide_path,
            seg_config=row["Segmentation Config"],
            site_type=row["Site Type"],
            sex=row.get("Sex", "Unknown"),
            tissue_site=row.get("Tissue Site", "Unknown"),
            cancer_subtype=row["Cancer Subtype"],
            cancer_subtype_name_map=cancer_subtype_name_map,
            ihc_subtype=row.get("IHC Subtype", ""),
            num_workers=num_workers,
        )

        slide_time = time.time() - slide_start
        logger.info(f"Slide {idx + 1} completed in {slide_time:.2f}s")

        results.append(
            {
                "slide": slide_path,
                "time": slide_time,
                "has_mask": slide_mask is not None,
                "has_aeon": aeon_results is not None,
                "has_paladin": paladin_results is not None,
            }
        )

    total_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    logger.info("=" * 80)
    logger.info(f"Sequential processing completed in {total_time:.2f}s")
    logger.info(f"Average time per slide: {total_time / len(slides):.2f}s")
    if torch.cuda.is_available():
        logger.info(f"Peak GPU memory: {peak_memory / (1024**3):.2f} GB")
    logger.info("=" * 80)

    return {
        "method": "sequential",
        "total_time": total_time,
        "num_slides": len(slides),
        "avg_time_per_slide": total_time / len(slides),
        "peak_memory_gb": peak_memory / (1024**3) if torch.cuda.is_available() else 0,
        "per_slide_results": results,
    }


def benchmark_batch_processing(
    slides, settings_df, cancer_subtype_name_map, num_workers
):
    """Benchmark optimized batch processing (models loaded once)."""
    logger.info("=" * 80)
    logger.info("BENCHMARKING: Batch Processing (NEW METHOD)")
    logger.info("=" * 80)

    start_time = time.time()

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    all_slide_masks, all_aeon_results, all_paladin_results = analyze_slides_batch(
        slides=slides,
        settings_df=settings_df,
        cancer_subtype_name_map=cancer_subtype_name_map,
        num_workers=num_workers,
        aggressive_memory_mgmt=None,  # Auto-detect
        progress=None,
    )

    total_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    logger.info("=" * 80)
    logger.info(f"Batch processing completed in {total_time:.2f}s")
    logger.info(f"Average time per slide: {total_time / len(slides):.2f}s")
    if torch.cuda.is_available():
        logger.info(f"Peak GPU memory: {peak_memory / (1024**3):.2f} GB")
    logger.info("=" * 80)

    return {
        "method": "batch",
        "total_time": total_time,
        "num_slides": len(slides),
        "avg_time_per_slide": total_time / len(slides),
        "peak_memory_gb": peak_memory / (1024**3) if torch.cuda.is_available() else 0,
        "num_successful": len(all_slide_masks),
    }


def compare_results(sequential_stats, batch_stats):
    """Compare and report performance differences."""
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)

    speedup = sequential_stats["total_time"] / batch_stats["total_time"]
    time_saved = sequential_stats["total_time"] - batch_stats["total_time"]
    percent_faster = (
        1 - (batch_stats["total_time"] / sequential_stats["total_time"])
    ) * 100

    logger.info(f"Number of slides: {sequential_stats['num_slides']}")
    logger.info(f"")
    logger.info(f"Sequential processing: {sequential_stats['total_time']:.2f}s")
    logger.info(f"Batch processing:      {batch_stats['total_time']:.2f}s")
    logger.info(f"")
    logger.info(f"Time saved:  {time_saved:.2f}s")
    logger.info(f"Speedup:     {speedup:.2f}x")
    logger.info(f"Improvement: {percent_faster:.1f}% faster")

    if torch.cuda.is_available():
        logger.info(f"")
        logger.info(
            f"Sequential peak memory: {sequential_stats['peak_memory_gb']:.2f} GB"
        )
        logger.info(f"Batch peak memory:      {batch_stats['peak_memory_gb']:.2f} GB")
        memory_diff = batch_stats["peak_memory_gb"] - sequential_stats["peak_memory_gb"]
        logger.info(f"Memory difference:      {memory_diff:+.2f} GB")

    logger.info("=" * 80)

    return {
        "speedup": speedup,
        "time_saved_seconds": time_saved,
        "percent_faster": percent_faster,
        "sequential_stats": sequential_stats,
        "batch_stats": batch_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark batch processing performance"
    )
    parser.add_argument("--slides", nargs="+", help="List of slide paths to process")
    parser.add_argument(
        "--slide-csv", type=str, help="CSV file with slide paths and settings"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--skip-sequential",
        action="store_true",
        help="Skip sequential benchmark (faster, only test batch mode)",
    )
    parser.add_argument(
        "--output", type=str, help="Save benchmark results to JSON file"
    )

    args = parser.parse_args()

    if not args.slides and not args.slide_csv:
        parser.error("Must provide either --slides or --slide-csv")

    # Load cancer subtype mappings
    from mosaic.gradio_app import download_and_process_models

    cancer_subtype_name_map, cancer_subtypes, reversed_cancer_subtype_name_map = (
        download_and_process_models()
    )

    # Prepare slides and settings
    if args.slide_csv:
        settings_df = load_settings(args.slide_csv)
        settings_df = validate_settings(
            settings_df,
            cancer_subtype_name_map,
            cancer_subtypes,
            reversed_cancer_subtype_name_map,
        )
        slides = settings_df["Slide"].tolist()
    else:
        slides = args.slides
        # Create default settings
        settings_df = pd.DataFrame(
            {
                "Slide": slides,
                "Site Type": ["Primary"] * len(slides),
                "Sex": ["Unknown"] * len(slides),
                "Tissue Site": ["Unknown"] * len(slides),
                "Cancer Subtype": ["Unknown"] * len(slides),
                "IHC Subtype": [""] * len(slides),
                "Segmentation Config": ["Biopsy"] * len(slides),
            }
        )

    logger.info(f"Benchmarking with {len(slides)} slides")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run benchmarks
    if not args.skip_sequential:
        sequential_stats = benchmark_sequential_processing(
            slides, settings_df, cancer_subtype_name_map, args.num_workers
        )

    batch_stats = benchmark_batch_processing(
        slides, settings_df, cancer_subtype_name_map, args.num_workers
    )

    # Compare results
    if not args.skip_sequential:
        comparison = compare_results(sequential_stats, batch_stats)

        # Save results if requested
        if args.output:
            import json

            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(comparison, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Verify Aeon test results against expected ground truth.

This script reads the test results and compares them against the ground truth
values in test_samples.json to validate the Aeon model predictions.

Usage:
    python verify_aeon_results.py \
        --test-samples test_slides/test_samples.json \
        --results-dir test_slides/results
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple


def load_test_samples(test_samples_file: Path) -> List[Dict]:
    """Load test samples from JSON file.

    Args:
        test_samples_file: Path to test_samples.json

    Returns:
        List of test sample dictionaries
    """
    with open(test_samples_file) as f:
        return json.load(f)


def load_aeon_results(slide_id: str, results_dir: Path) -> Tuple[str, float]:
    """Load Aeon prediction results for a slide.

    Args:
        slide_id: Slide identifier
        results_dir: Directory containing results

    Returns:
        Tuple of (predicted_subtype, confidence)
    """
    results_file = results_dir / slide_id / f"{slide_id}_aeon_results.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)

    if df.empty:
        raise ValueError(f"Empty results file: {results_file}")

    # Get top prediction
    top_prediction = df.iloc[0]
    return top_prediction["Cancer Subtype"], top_prediction["Confidence"]


def verify_results(test_samples: List[Dict], results_dir: Path) -> Dict:
    """Verify all test results against ground truth.

    Args:
        test_samples: List of test sample dictionaries
        results_dir: Directory containing results

    Returns:
        Dictionary with verification statistics
    """
    total = len(test_samples)
    passed = 0
    failed = 0
    results = []

    print("=" * 80)
    print("Aeon Model Verification Report")
    print("=" * 80)
    print()

    for sample in test_samples:
        slide_id = sample.get("slide_id") or sample.get("image_id")
        ground_truth = sample.get("cancer_subtype") or sample.get("cancer_type")
        site_type = sample["site_type"]
        sex = sample["sex"]
        tissue_site = sample["tissue_site"]

        print(f"Slide: {slide_id}")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Site Type: {site_type}")
        print(f"  Sex: {sex}")
        print(f"  Tissue Site: {tissue_site}")

        try:
            predicted, confidence = load_aeon_results(slide_id, results_dir)

            print(f"  Predicted: {predicted}")
            print(f"  Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

            # Check if prediction matches
            if predicted == ground_truth:
                print("  Status: ✓ PASS")
                passed += 1
                status = "PASS"
            else:
                print(f"  Status: ✗ FAIL (expected {ground_truth}, got {predicted})")
                failed += 1
                status = "FAIL"

            results.append(
                {
                    "slide_id": slide_id,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "confidence": confidence,
                    "site_type": site_type,
                    "sex": sex,
                    "tissue_site": tissue_site,
                    "status": status,
                }
            )

        except Exception as e:
            print(f"  Status: ✗ ERROR - {e}")
            failed += 1
            results.append(
                {
                    "slide_id": slide_id,
                    "ground_truth": ground_truth,
                    "predicted": None,
                    "confidence": None,
                    "site_type": site_type,
                    "sex": sex,
                    "tissue_site": tissue_site,
                    "status": "ERROR",
                    "error": str(e),
                }
            )

        print()

    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total slides: {total}")
    print(f"Passed: {passed} ({passed / total * 100:.1f}%)")
    print(f"Failed: {failed} ({failed / total * 100:.1f}%)")
    print()

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")

    # Calculate statistics for passed tests
    if passed > 0:
        confidences = [r["confidence"] for r in results if r["status"] == "PASS"]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        print()
        print("Confidence Statistics (for passed tests):")
        print(f"  Average: {avg_confidence:.4f} ({avg_confidence * 100:.2f}%)")
        print(f"  Minimum: {min_confidence:.4f} ({min_confidence * 100:.2f}%)")
        print(f"  Maximum: {max_confidence:.4f} ({max_confidence * 100:.2f}%)")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": passed / total if total > 0 else 0,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify Aeon test results against ground truth"
    )
    parser.add_argument(
        "--test-samples",
        type=Path,
        default=Path("test_slides/test_samples.json"),
        help="Path to test_samples.json (default: test_slides/test_samples.json)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("test_slides/results"),
        help="Directory containing results (default: test_slides/results)",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path to save verification report as JSON"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.test_samples.exists():
        raise FileNotFoundError(f"Test samples file not found: {args.test_samples}")

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    # Load test samples
    test_samples = load_test_samples(args.test_samples)

    # Verify results
    verification_report = verify_results(test_samples, args.results_dir)

    # Save report if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(verification_report, f, indent=2)
        print()
        print(f"Verification report saved to: {args.output}")

    # Exit with appropriate code
    if verification_report["failed"] > 0:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()

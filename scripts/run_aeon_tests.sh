#!/bin/bash
# Aeon Model Test Script
# This script runs the Aeon cancer subtype prediction model on test slides
# for reproducibility and validation.

set -e  # Exit on error

# Configuration
TEST_SAMPLES_FILE="test_slides/test_samples.json"
RESULTS_DIR="test_slides/results"
LOG_DIR="test_slides/logs"
SEGMENTATION_CONFIG="Biopsy"
NUM_WORKERS=4

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================="
echo "Aeon Model Test Suite"
echo "========================================="
echo ""

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

# Check if test samples file exists
if [ ! -f "${TEST_SAMPLES_FILE}" ]; then
    echo -e "${RED}Error: Test samples file not found: ${TEST_SAMPLES_FILE}${NC}"
    exit 1
fi

# Read test samples
echo "Reading test samples from ${TEST_SAMPLES_FILE}..."
SLIDE_IDS=$(python3 -c "
import json
with open('${TEST_SAMPLES_FILE}') as f:
    samples = json.load(f)
    for sample in samples:
        slide_id = sample.get('slide_id') or sample.get('image_id')
        print(slide_id)
")

# Count slides
NUM_SLIDES=$(echo "${SLIDE_IDS}" | wc -l)
echo -e "${GREEN}Found ${NUM_SLIDES} test slides${NC}"
echo ""

# Process each slide
CURRENT=0
PASSED=0
FAILED=0

for SLIDE_ID in ${SLIDE_IDS}; do
    CURRENT=$((CURRENT + 1))

    echo "========================================="
    echo -e "${YELLOW}Processing slide ${CURRENT}/${NUM_SLIDES}: ${SLIDE_ID}${NC}"
    echo "========================================="

    # Get slide metadata
    METADATA=$(python3 -c "
import json
with open('${TEST_SAMPLES_FILE}') as f:
    samples = json.load(f)
    for sample in samples:
        slide_id = sample.get('slide_id') or sample.get('image_id')
        if slide_id == '${SLIDE_ID}':
            cancer_subtype = sample.get('cancer_subtype') or sample.get('cancer_type')
            print(f\"{cancer_subtype}|{sample['site_type']}|{sample['sex']}|{sample['tissue_site']}\")
            break
")

    IFS='|' read -r CANCER_SUBTYPE SITE_TYPE SEX TISSUE_SITE <<< "${METADATA}"

    echo "Ground Truth:"
    echo "  Cancer Subtype: ${CANCER_SUBTYPE}"
    echo "  Site Type: ${SITE_TYPE}"
    echo "  Sex: ${SEX}"
    echo "  Tissue Site: ${TISSUE_SITE}"
    echo ""

    # Find slide file
    SLIDE_FILE=$(find test_slides -name "${SLIDE_ID}.svs" -o -name "${SLIDE_ID}.tiff" -o -name "${SLIDE_ID}.ndpi" 2>/dev/null | head -1)

    if [ -z "${SLIDE_FILE}" ]; then
        echo -e "${RED}Error: Slide file not found for ${SLIDE_ID}${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "Slide file: ${SLIDE_FILE}"
    echo ""

    # Run Mosaic pipeline with Aeon inference
    LOG_FILE="${LOG_DIR}/${SLIDE_ID}_aeon_test.log"

    echo "Running Mosaic pipeline..."
    if uv run python -m mosaic.cli \
        --input-slide "${SLIDE_FILE}" \
        --output-dir "${RESULTS_DIR}/${SLIDE_ID}" \
        --cancer-subtype "Unknown" \
        --site-type "${SITE_TYPE}" \
        --sex "${SEX}" \
        --tissue-site "${TISSUE_SITE}" \
        --segmentation-config "${SEGMENTATION_CONFIG}" \
        --num-workers "${NUM_WORKERS}" \
        > "${LOG_FILE}" 2>&1; then

        # Check if results exist
        AEON_RESULTS="${RESULTS_DIR}/${SLIDE_ID}/${SLIDE_ID}_aeon_results.csv"

        if [ -f "${AEON_RESULTS}" ]; then
            # Extract prediction
            PREDICTION=$(python3 -c "
import pandas as pd
df = pd.read_csv('${AEON_RESULTS}')
if not df.empty:
    print(f\"{df.iloc[0]['Cancer Subtype']}|{df.iloc[0]['Confidence']:.4f}\")
")

            IFS='|' read -r PRED_SUBTYPE CONFIDENCE <<< "${PREDICTION}"

            echo ""
            echo "Aeon Prediction:"
            echo "  Predicted: ${PRED_SUBTYPE}"
            echo "  Confidence: ${CONFIDENCE}"
            echo ""

            # Check if prediction matches ground truth
            if [ "${PRED_SUBTYPE}" == "${CANCER_SUBTYPE}" ]; then
                echo -e "${GREEN}✓ PASS: Prediction matches ground truth${NC}"
                PASSED=$((PASSED + 1))
            else
                echo -e "${RED}✗ FAIL: Prediction does not match ground truth${NC}"
                echo "  Expected: ${CANCER_SUBTYPE}"
                echo "  Got: ${PRED_SUBTYPE}"
                FAILED=$((FAILED + 1))
            fi
        else
            echo -e "${RED}✗ FAIL: Aeon results file not found${NC}"
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e "${RED}✗ FAIL: Mosaic pipeline failed${NC}"
        echo "Check log file: ${LOG_FILE}"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total slides: ${NUM_SLIDES}"
echo -e "${GREEN}Passed: ${PASSED}${NC}"
if [ ${FAILED} -gt 0 ]; then
    echo -e "${RED}Failed: ${FAILED}${NC}"
else
    echo "Failed: ${FAILED}"
fi
echo ""

if [ ${FAILED} -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check logs in ${LOG_DIR}${NC}"
    exit 1
fi

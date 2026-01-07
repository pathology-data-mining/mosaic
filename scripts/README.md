# Mosaic Scripts

This directory contains utility scripts for working with the Mosaic pipeline, particularly for Aeon model testing and deployment.

## Aeon Model Scripts

### 1. export_aeon_checkpoint.py

Export PyTorch Lightning checkpoint to pickle format for inference.

**Usage:**
```bash
python scripts/export_aeon_checkpoint.py \
    --checkpoint data/checkpoint.ckpt \
    --output data/aeon_model.pkl \
    --metadata-dir data/metadata
```

**Arguments:**
- `--checkpoint`: Path to PyTorch Lightning checkpoint (.ckpt file)
- `--output`: Path to save exported model (.pkl file)
- `--metadata-dir`: Directory containing metadata files (default: data/metadata)

**Requirements:**
- paladin package from git repo (must have AeonLightningModule)
- PyTorch Lightning
- Metadata files: n_classes.txt, ontology_embedding_dim.txt, target_dict.tsv

**Example:**
```bash
# Export the checkpoint
uv run python scripts/export_aeon_checkpoint.py \
    --checkpoint data/checkpoint.ckpt \
    --output data/aeon_model.pkl

# Output:
# Loading metadata from data/metadata...
# Loading checkpoint from data/checkpoint.ckpt...
# Saving model to data/aeon_model.pkl...
# ✓ Successfully exported checkpoint to data/aeon_model.pkl
#   Model size: 118.0 MB
#   Model class: AeonLateAggregator
#   Number of classes: 160
#   Ontology embedding dim: 20
#   Number of histologies: 160
```

### 2. run_aeon_tests.sh

Run the Aeon model on test slides and validate predictions.

**Usage:**
```bash
./scripts/run_aeon_tests.sh
```

**Configuration:**
The script reads test samples from `test_slides/test_samples.json` and processes each slide through the full Mosaic pipeline with:
- Cancer subtype: Unknown (triggers Aeon inference)
- Segmentation config: Biopsy
- Number of workers: 4

**Output:**
- Results saved to `test_slides/results/{slide_id}/`
- Logs saved to `test_slides/logs/`
- Summary showing passed/failed tests

**Example Output:**
```
=========================================
Aeon Model Test Suite
=========================================

Found 3 test slides

=========================================
Processing slide 1/3: 881837
=========================================
Ground Truth:
  Cancer Subtype: BLCA
  Site Type: Primary
  Sex: Male
  Tissue Site: Bladder

Running Mosaic pipeline...

Aeon Prediction:
  Predicted: BLCA
  Confidence: 0.9819

✓ PASS: Prediction matches ground truth

[... continues for all slides ...]

=========================================
Test Summary
=========================================
Total slides: 3
Passed: 3
Failed: 0

All tests passed!
```

### 3. verify_aeon_results.py

Verify Aeon test results against expected ground truth.

**Usage:**
```bash
python scripts/verify_aeon_results.py \
    --test-samples test_slides/test_samples.json \
    --results-dir test_slides/results \
    --output test_slides/verification_report.json
```

**Arguments:**
- `--test-samples`: Path to test samples JSON file (default: test_slides/test_samples.json)
- `--results-dir`: Directory containing results (default: test_slides/results)
- `--output`: Optional path to save verification report as JSON

**Example:**
```bash
# Verify results and save report
uv run python scripts/verify_aeon_results.py \
    --output test_slides/verification_report.json

# Output:
# ================================================================================
# Aeon Model Verification Report
# ================================================================================
#
# Slide: 881837
#   Ground Truth: BLCA
#   Site Type: Primary
#   Sex: Male
#   Tissue Site: Bladder
#   Predicted: BLCA
#   Confidence: 0.9819 (98.19%)
#   Status: ✓ PASS
#
# [... continues for all slides ...]
#
# ================================================================================
# Summary
# ================================================================================
# Total slides: 3
# Passed: 3 (100.0%)
# Failed: 0 (0.0%)
#
# ✓ All tests passed!
#
# Confidence Statistics (for passed tests):
#   Average: 0.9910 (99.10%)
#   Minimum: 0.9819 (98.19%)
#   Maximum: 0.9961 (99.61%)
```

## Workflow

### Complete Testing Workflow

1. **Export checkpoint** (if needed):
   ```bash
   uv run python scripts/export_aeon_checkpoint.py \
       --checkpoint data/checkpoint.ckpt \
       --output data/aeon_model.pkl
   ```

2. **Run tests**:
   ```bash
   ./scripts/run_aeon_tests.sh
   ```

3. **Verify results**:
   ```bash
   uv run python scripts/verify_aeon_results.py \
       --output test_slides/verification_report.json
   ```

### Quick Verification

If you already have test results and just want to verify them:

```bash
uv run python scripts/verify_aeon_results.py
```

## Test Samples Format

The test samples JSON file should have this format:

```json
[
  {
    "slide_id": "881837",
    "cancer_subtype": "BLCA",
    "site_type": "Primary",
    "sex": "Male",
    "tissue_site": "Bladder"
  },
  {
    "slide_id": "744547",
    "cancer_subtype": "HCC",
    "site_type": "Metastatic",
    "sex": "Male",
    "tissue_site": "Liver"
  }
]
```

## Dependencies

All scripts require:
- Python 3.10+
- uv package manager
- Mosaic package with dependencies

Additional requirements for checkpoint export:
- paladin from git repository (dev branch)
- PyTorch Lightning

## Exit Codes

- `0`: Success (all tests passed)
- `1`: Failure (one or more tests failed)

## Troubleshooting

### "AeonLightningModule not found"
```bash
uv sync --upgrade-package paladin
```

### "Metadata files not found"
Make sure you have:
- `data/metadata/n_classes.txt`
- `data/metadata/ontology_embedding_dim.txt`
- `data/metadata/target_dict.tsv`

### "Test slides not found"
Place your test slides in `test_slides/` directory and update `test_samples.json` with correct paths.

## See Also

- [AEON_TEST_SUMMARY.md](../test_slides/AEON_TEST_SUMMARY.md) - Detailed test results and validation
- [README.md](../README.md) - Main Mosaic documentation

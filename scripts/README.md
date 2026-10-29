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

## Paladin Model Sync

### sync_paladin_models.py

Idempotent script for syncing paladin model checkpoints to HuggingFace.
Works for any project name (`paladin-redo`, or future variants).

**Prerequisites:**
- HF token in `~/.hf_cred.env` (loaded automatically)
- For `gen-configs`: checkpoints at `/gpfs/mskmind_ess/boehmk/paladin/storage/{project}/`
- For SLURM jobs: paladin venv at `/gpfs/mskmind_ess/limr/repos/paladin/venv`

**Phases:**

#### Phase 1: Generate SLURM configs

```bash
python scripts/sync_paladin_models.py --project paladin-redo --phase gen-configs
# Or:
make gen-pkl-configs PROJECT=paladin-redo
```

Reads `aggregated_auroc.freeze.corrected.csv`, picks the best-auroc run per
`(histology, target_name)` combo, writes Hydra pkl configs to
`/gpfs/mskmind_ess/limr/repos/paladin/conf/train/pkl_{run_id}.yaml`, and
generates `scripts/submit_paladin_redo_pickles.sh`.

#### SLURM pkl export (user action)

After `gen-configs`, run the SLURM jobs to export pkl files:

```bash
cd /gpfs/mskmind_ess/limr/repos/paladin
bash /path/to/mosaic/scripts/submit_paladin_redo_pickles.sh
```

Jobs use the paladin venv and output `{run_id}.epoch=E-step=S.model.pkl` alongside
the checkpoint. Wait for all jobs to complete before proceeding.

#### Phase 2: Upload to HuggingFace

```bash
python scripts/sync_paladin_models.py --project paladin-redo --phase upload
# Or:
make upload-paladin-models PROJECT=paladin-redo
```

Uploads new pkl files to `PDM-Group/paladin-aeon-models` HF repo under
`paladin_redo/`. Skips files already present (idempotent). Exits non-zero
if any expected pkls are missing.

#### Phase 3: Update model map

```bash
python scripts/sync_paladin_models.py --project paladin-redo --phase update-map
# Or:
make update-model-map PROJECT=paladin-redo
```

Updates `data/paladin_model_map.csv` (replacing existing entries or adding new ones)
and pushes the CSV to HF. Then commit locally:

```bash
git add data/paladin_model_map.csv
git commit -m "feat: add paladin-redo models to paladin_model_map.csv"
git push
```

#### Run upload + update-map together

```bash
make sync-paladin-models PROJECT=paladin-redo
```

**For future paladin variants**, just pass the new project name:

```bash
python scripts/sync_paladin_models.py --project paladin-v2 --phase gen-configs
# ... run SLURM jobs ...
make sync-paladin-models PROJECT=paladin-v2
```

---

## Aeon Model Scripts

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

## HF Space Health Check

### hf_space_e2e.py

End-to-end check of the deployed Space (https://raylim-mosaic.hf.space). It drives
the real UI with Playwright and runs a full analysis on a public TCGA-OV slide
(`TCGA-57-1582-01Z-00-DX1`, ~90 MB, downloaded from GDC). It does not import
`mosaic`, so it only needs Playwright:

```bash
pip install playwright && playwright install --with-deps chromium
python scripts/hf_space_e2e.py --out-dir hf_e2e_out/   # or: make e2e-hf-space
```

| Stage | Checks |
|-------|--------|
| `runtime` | HF runtime API reports `RUNNING` (wakes a sleeping Space, waits up to 25 min) |
| `ui_load` | Page loads, Analyze button visible |
| `tcga_fetch` | "Fetch from TCGA" downloads the slide on the Space |
| `download` | Slide downloaded locally from GDC (cached in `~/.cache/mosaic-hf-e2e`) |
| `analysis` | Slide uploaded, analysis completes |
| `results` | Aeon top 3 contains HGSOC, Paladin probabilities in [0, 1], no browser freeze |

The analysis uploads the slide rather than using the TCGA path because
TCGA results are cached by settings, so later runs would skip inference.

Output: `report.json` (per-stage status, timings, failure category) plus a
screenshot for each failed browser stage. Exit code 0 means all stages passed.
`--restart-on-infra-error` factory-restarts the Space once (using `HF_TOKEN`)
when it is down with an HF host error such as the containerd/NVIDIA mount failure.

### hf_space_e2e_agent.sh

Scheduled wrapper. It runs the check with `--restart-on-infra-error`, then a
`claude -p` agent (`hf_space_e2e_agent_prompt.md`) reads the report, diagnoses
any failure and files or updates a Jira issue labelled `mosaic-hf-e2e`. The
agent can only read files, query the HF API and git log, and use four Jira
tools. Schedule it from cron by submitting it with `sbatch` (the script header
has an example entry). Because cron doesn't read `~/.bashrc`, put `PATH`,
`JIRA_PROJECT`, optional `JIRA_ASSIGNEE`, `HF_TOKEN` (write access to `raylim/mosaic`) and any `JIRA_*`
credentials in `~/.config/mosaic-hf-e2e.env`, which the script sources.

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

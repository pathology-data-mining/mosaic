# Aeon Model Test Summary

## Overview

This document summarizes the Aeon cancer subtype prediction model testing performed on January 7, 2026.

## Model Information

- **Model File**: `aeon_model.pkl` (118MB)
- **Source**: Exported from `checkpoint.ckpt` (469MB, Nov 29, 2024)
- **Architecture**: AeonLateAggregator with late fusion
- **Output Classes**: 160 cancer subtypes
- **Input Features**:
  - Tile embeddings from Optimus model
  - Sex (one-hot encoded, 3 classes)
  - Tissue site (one-hot encoded, 57 classes)
  - Site type (Primary/Metastatic)

## Test Slides

Three test slides were processed using the full Mosaic pipeline with Aeon inference:

### Slide 1: 881837
- **File**: `881837.svs`
- **Ground Truth**: BLCA (Bladder Urothelial Carcinoma)
- **Site Type**: Primary
- **Sex**: Male
- **Tissue Site**: Bladder
- **Prediction**: BLCA
- **Confidence**: 98.19%
- **Status**: ✓ PASS

### Slide 2: 744547
- **File**: `744547.svs`
- **Ground Truth**: HCC (Hepatocellular Carcinoma)
- **Site Type**: Metastatic
- **Sex**: Male
- **Tissue Site**: Liver
- **Prediction**: HCC
- **Confidence**: 99.49%
- **Status**: ✓ PASS

### Slide 3: 755246
- **File**: `755246.svs`
- **Ground Truth**: HCC (Hepatocellular Carcinoma)
- **Site Type**: Primary
- **Sex**: Male
- **Tissue Site**: Liver
- **Prediction**: HCC
- **Confidence**: 99.61%
- **Status**: ✓ PASS

## Test Results

| Slide ID | Ground Truth | Prediction | Confidence | Next Highest | Status |
|----------|--------------|------------|------------|--------------|--------|
| 881837   | BLCA         | BLCA       | 98.19%     | UTUC (0.87%) | ✓ PASS |
| 744547   | HCC          | HCC        | 99.49%     | IHCH (0.18%) | ✓ PASS |
| 755246   | HCC          | HCC        | 99.61%     | IHCH (0.29%) | ✓ PASS |

**Overall Accuracy**: 3/3 (100%)

## Pipeline Configuration

### Segmentation
- **Config**: Biopsy (`SegmentationConfig.BIOPSY`)
- **Tissue Detection**: Automated segmentation of tissue regions
- **Tile Size**: 224x224 pixels at 20x magnification

### Feature Extraction
- **CTransPath**: Pretrained histopathology foundation model
- **Optimus**: Multi-task feature aggregator
- **Marker Classifier**: Tissue marker prediction

### Aeon Inference
- **Model Path**: `data/aeon_model.pkl`
- **Batch Size**: 8
- **Workers**: 4
- **Sex Encoding**: Male=0 (one-hot: [1,0,0])
- **Tissue Site Encoding**:
  - Bladder=11 (one-hot vector, 57 dims)
  - Liver=26 (one-hot vector, 57 dims)

## Key Implementation Details

### Cancer Type Mapping
- Mappings loaded from `data/metadata/target_dict.tsv`
- 160 histologies supported
- 5 cancer types excluded from predictions: UDMN, ADNOS, CUP, CUPNOS, NOT

### Model Architecture
```python
AeonLateAggregator(
    tile_emb_dim=768,
    num_targets=160,
    sex_embedding_dim=mini_latent_dim (latent_dim // 4),
    tissue_site_embedding_dim=mini_latent_dim,
    site_embedding_dim=mini_latent_dim
)
```

### Encoding Functions
- **Sex**: `encode_sex(sex_str)` → index (0-2) → one-hot (3 classes)
- **Tissue Site**: `encode_tissue_site(tissue_site_str)` → index (0-56) → one-hot (57 classes)

## Critical Fixes Applied

### Issue 1: Model-Metadata Mismatch
- **Problem**: Model outputs 160 classes but code used 183-entry mapping
- **Solution**: Load mappings from `metadata/target_dict.tsv` instead of global constants
- **Files Modified**: `src/mosaic/inference/aeon.py` (lines 87-102, 127, 130-147)

### Issue 2: Checkpoint Format
- **Problem**: Inference code expects `.pkl` files, not PyTorch Lightning `.ckpt` files
- **Solution**: Exported checkpoint using paladin's `AeonLightningModule.load_from_checkpoint()`
- **Export Command**: See `scripts/export_aeon_checkpoint.py`

### Issue 3: Missing AeonLateAggregator
- **Problem**: PyPI paladin package had `AeonAggregator`, not `AeonLateAggregator`
- **Solution**: Installed paladin from git repo dev branch
- **Command**: `uv sync --upgrade-package paladin`

## Dependencies

### Critical Packages
- `paladin` (from git: ssh://git@github.com/pathology-data-mining/paladin.git@dev)
- `torch>=2.0`
- `pytorch-lightning`
- `pandas`
- `numpy`

### Model Files Required
- `aeon_model.pkl` (118MB)
- `metadata/target_dict.tsv`
- `metadata/n_classes.txt`
- `metadata/ontology_embedding_dim.txt`
- `sex_original_to_idx.csv`
- `tissue_site_original_to_idx.csv`

## Reproducibility

All test results are fully reproducible using:
1. The test samples defined in `test_samples.json`
2. The run script: `scripts/run_aeon_tests.sh`
3. The model and metadata uploaded to `PDM-Group/paladin-aeon-models` on Hugging Face

## Output Files

For each slide, the following files are generated in `test_slides/results/{slide_id}/`:
- `{slide_id}_aeon_results.csv` - Full confidence scores for all 160 cancer subtypes
- `{slide_id}_paladin_results.csv` - Biomarker predictions
- `{slide_id}_mask.png` - Tissue segmentation mask
- `{slide_id}_features.h5` - Extracted tile features

## Validation Metrics

- **Prediction Accuracy**: 100% (3/3)
- **Average Confidence**: 99.10%
- **Minimum Confidence**: 98.19%
- **Maximum Confidence**: 99.61%

## Hugging Face Repository

All model files and metadata have been uploaded to:
- **Repository**: `PDM-Group/paladin-aeon-models`
- **URL**: https://huggingface.co/PDM-Group/paladin-aeon-models

### Uploaded Files (Jan 7, 2026)
- `aeon_model.pkl` (118MB)
- `metadata/` (5 files)
- `sex_original_to_idx.csv`
- `tissue_site_original_to_idx.csv`

## Test Date

- **Date**: January 7, 2026
- **Git Commit**: 49fbf68 (Complete implementation of sex and tissue site parameters)
- **Tester**: Ray Lim

# Add TCGA results caching with HuggingFace Dataset sync

## Summary

Implements comprehensive caching system for TCGA slide metadata and analysis results, with optional sync to HuggingFace Dataset for persistence across sessions and environments.

### Key Benefits

- **Instant metadata display**: Cached GDC metadata loads in milliseconds vs 2-5s API call
- **Instant results display**: Cached analysis results display immediately vs 30-120s re-inference
- **HF Spaces persistence**: Results survive Space restarts via HF Dataset sync
- **Works everywhere**: Local dev, HF Spaces, Docker containers

### Implementation Details

**Part 1: Persistent Storage on HF Spaces**
- Modified `get_tcga_cache_directory()` to use `/data/mosaic_tcga_slides` on HF Spaces for persistent storage across restarts

**Part 2: Metadata Caching**
- Added `save_metadata_cache()` and `load_metadata_cache()` functions
- Modified `get_case_metadata()` to check cache first, falling back to GDC API on miss
- Cache stored as `metadata.json` with version field for forward compatibility

**Part 3: Analysis Results Caching**
- Added `compute_settings_hash()` to generate deterministic 12-char hash from analysis settings
- Added `save_analysis_results()` and `load_analysis_results()` functions
- Cache keyed by settings hash: different analysis settings create separate cache entries

**Part 4: UI Integration**
- Modified `analyze_slides()` to check for cached results before analysis
- Automatically saves results to cache after analysis for TCGA slides
- Updated `fetch_tcga_slide()` progress messages to indicate cached vs fresh metadata

**Part 5: HF Dataset Sync**
- Added `upload_results_to_hf()` and `download_results_from_hf()` functions
- Uses `MOSAIC_TCGA_CACHE_HF_REPO` env var (e.g., `PDM-Group/mosaic-tcga-cache`)
- On-demand download in `fetch_tcga_slide()` before metadata lookup
- Automatic upload in `analyze_slides()` after saving results

### Cache Structure

```
{cache_dir}/{uuid}/
├── metadata.json              # GDC metadata (version-aware)
├── {filename}.svs             # Slide file
└── results/{settings_hash}/   # Analysis results by settings
    ├── aeon_results.csv       # Cancer subtype predictions
    ├── paladin_results.csv    # Biomarker predictions
    └── slide_mask.png         # Tissue segmentation mask
```

### Environment Variables

- `MOSAIC_TCGA_CACHE_DIR`: Override default cache location (`~/.cache/mosaic/tcga_slides`)
- `MOSAIC_TCGA_CACHE_HF_REPO`: Enable HF Dataset sync (e.g., `PDM-Group/mosaic-tcga-cache`)
  - When set, automatically syncs cached results to/from HF Dataset
  - Works on both local installations and HF Spaces

### Testing

Added 18 comprehensive tests:
- **TestMetadataCache** (7 tests): Roundtrip, corruption handling, version mismatch, None values, permission errors
- **TestAnalysisResultsCache** (6 tests): Roundtrip, missing files, hash determinism, index restoration
- **TestHFDatasetSync** (3 tests): Upload/download with mocked HF API, env var handling
- **TestGetCaseMetadataWithCache** (2 tests): Cache hit/miss behavior

All 321 tests passing ✅

### Design Decisions

1. **Best-effort caching**: All caching operations are non-fatal and fall back gracefully on errors
2. **Settings-aware**: Different analysis settings (segmentation, site type, etc.) create separate cache entries via SHA-256 hash
3. **Backward compatible**: Works for both TCGA and non-TCGA slides; no breaking changes
4. **Opt-in HF sync**: Only syncs when `MOSAIC_TCGA_CACHE_HF_REPO` is set
5. **Version-aware**: Metadata cache includes version field for future schema changes

### Verification

- ✅ `make format` - All files properly formatted
- ✅ `make test-fast` - 321 tests passed, 2 skipped
- ✅ Manual testing (local caching, HF Spaces persistence)

### Files Modified

- `src/mosaic/data_directory.py` - HF Spaces /data detection
- `src/mosaic/tcga.py` - Cache functions + HF Dataset sync (~400 lines added)
- `src/mosaic/ui/app.py` - Integrate caching in fetch/analyze workflows
- `tests/test_tcga.py` - 18 new comprehensive tests (~550 lines added)

### Statistics

- **6 files changed**: 1,091 insertions, 37 deletions
- **New functions**: 8 (save/load metadata, save/load results, compute hash, upload/download HF)
- **Test coverage**: 18 new tests covering all new functionality

🤖 Generated with [Claude Code](https://claude.com/claude-code)

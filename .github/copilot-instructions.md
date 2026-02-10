# Copilot Instructions for Mosaic

## Quick Start

### Build, Test, and Lint

```bash
# Setup
make install-dev              # Install all dependencies including dev tools
source .venv/bin/activate     # Activate virtual environment

# Testing
make test                     # Run all tests with coverage
make test-fast                # Run tests without coverage (2-3x faster)
make test-specific TEST=tests/test_cli.py::test_single_slide_mode  # Run single test
make test-ui                  # UI component tests only
make test-cli                 # CLI routing tests only

# Code Quality
make format                   # Auto-format with black (run before commits)
make format-check             # Verify formatting (CI check)
make lint                     # Pylint on src/mosaic/
make quality                  # format-check + lint

# Running
make run-ui                   # Launch Gradio web interface
make run-single SLIDE=path.svs OUTPUT=dir/  # Single slide CLI
make run-batch CSV=settings.csv OUTPUT=dir/ # Batch processing CLI
```

### Dependencies Installation

This project has **private dependencies** that require SSH key access:
- **paladin** (private): `ssh://git@github.com/pathology-data-mining/paladin.git@dev`
- **mussel** (public): `https://github.com/pathology-data-mining/Mussel.git@mosaic-dev`

Ensure SSH keys are configured before running `uv sync` or Docker builds.

## Architecture

### Module Hierarchy

```
mosaic.gradio_app:main()         # CLI entry point
  ├── ModelCache                 # Pre-load models for batch processing
  ├── analyze_slide()            # Single slide (original, backward compatible)
  ├── analyze_slide_with_models() # Batch-aware (uses pre-loaded models)
  └── launch_gradio()            # Web UI
        ├── analyze_slides()     # Auto-detects batch vs single
        └── set_cancer_subtype_maps()  # Global cancer subtype mapping
```

### Core Pipeline (src/mosaic/analysis.py)

```
WSI Input
  → Tissue Segmentation (Mussel)
  → CTransPath Feature Extraction
  → Marker Classification (filter features)
  → Optimus Feature Extraction
  → Aeon Inference (100+ cancer subtypes)
  → Paladin Inference (cancer subtype-specific biomarkers)
```

### Key Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `mosaic.hardware` | GPU detection & config | `IS_ZEROGPU`, `IS_T4_GPU`, `DEFAULT_BATCH_SIZE` |
| `mosaic.model_manager` | Model caching for batch | `ModelCache.load_all_models()`, `cleanup()` |
| `mosaic.analysis` | Core slide analysis | `analyze_slide()`, `analyze_slide_with_models()` |
| `mosaic.ui.app` | Gradio web interface | `launch_gradio()`, `analyze_slides()` |
| `mosaic.ui.utils` | UI utilities | `validate_settings()`, `get_oncotree_code_name()` |
| `mosaic.inference.aeon` | Cancer subtype prediction | `run()`, `run_with_model()` |
| `mosaic.inference.paladin` | Biomarker prediction | `run()`, `run_with_models()` |

## Key Conventions

### 1. Batch Processing Pattern

**Backward Compatibility First**: Original functions are NEVER modified. New batch-aware functions are added in parallel.

```python
# Original function - UNCHANGED (backward compatible)
def analyze_slide(slide_path, settings):
    """Single slide analysis."""
    # ... original implementation ...

# New function - PARALLEL (batch-aware)
def analyze_slide_with_models(slide_path, model_cache, settings):
    """Single slide using pre-loaded models."""
    # ... uses model_cache instead of loading models ...
```

**Automatic Batch Detection** in UI:
```python
if len(slides) > 1:
    # Use batch mode with ModelCache
    cache = ModelCache()
    cache.load_all_models()
    try:
        results = [analyze_slide_with_models(s, cache, settings) for s in slides]
    finally:
        cache.cleanup()
else:
    # Use original single slide mode
    result = analyze_slide(slide, settings)
```

### 2. Hardware-Aware GPU Management

**Always use hardware detection** from `mosaic.hardware`:

```python
from mosaic.hardware import IS_T4_GPU, IS_ZEROGPU, DEFAULT_BATCH_SIZE

if IS_ZEROGPU:
    batch_size = 128
    num_workers = 0
elif IS_T4_GPU:
    batch_size = 64
    num_workers = 4
else:
    batch_size = 64
    num_workers = 8
```

**Memory Management by GPU Type**:
- **T4 GPU**: Aggressive cleanup (delete Paladin models immediately after inference)
- **A100/H100**: Caching strategy (keep Paladin models loaded for reuse)

**ZeroGPU Decorator** for HuggingFace Spaces:
```python
from mosaic.hardware import spaces

@spaces.GPU(duration=120)  # Allocate GPU for 120 seconds
def gpu_intensive_operation(data):
    return process_on_gpu(data)
```
The decorator is a no-op when not on HuggingFace Spaces.

### 3. Testing Patterns

**All tests run WITHOUT GPU or real models** using mocks from `tests/conftest.py`:

```python
# conftest.py provides mocks for:
# - mussel (tissue segmentation & feature extraction)
# - huggingface_hub (model downloading)
# - gradio (UI components with visibility tracking)
# - model inference (Aeon, Paladin)

def test_analysis(mock_aeon_model, mock_paladin_model):
    """Tests use mocked models, never real GPU inference."""
    result = analyze_slide(slide_path, settings)
    assert result is not None
```

**Test Categories**:
- `test_*.py` - Unit/integration tests
- `test_regression_*.py` - Backward compatibility tests
- `benchmark_*.py` - Performance benchmarks (not run in CI)

### 4. Gradio Component Visibility

Use `gr.update()` for dynamic UI state changes:

```python
import gradio as gr

def on_upload(files):
    if len(files) > 1:
        return gr.update(visible=True)  # Show batch options
    else:
        return gr.update(visible=False)

with gr.Column(visible=False) as batch_column:
    batch_options = gr.Checkbox(label="Batch Options")

upload.upload(on_upload, inputs=files, outputs=batch_column)
```

### 5. Logging (NOT print statements)

**Always use loguru** for logging:

```python
from loguru import logger

logger.info(f"Starting operation with {len(slides)} slides")
logger.success("Operation completed successfully")
logger.error(f"Validation failed: {error}")
logger.exception(f"Unexpected error: {e}")  # Includes traceback
```

**Never use print() in production code** (only in tests with `-s` flag).

### 6. Configuration via Environment Variables

```python
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface")
DATA_DIR = os.environ.get("MOSAIC_DATA_DIR", "./data")
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
```

## CI/CD

### GitHub Actions Workflows

| Workflow | Trigger | Matrix | Key Steps |
|----------|---------|--------|-----------|
| `tests.yml` | Push to main/dev, PRs | Python 3.10, 3.11 | Setup SSH, install uv, run `make test-coverage` |
| `code-quality.yml` | Push, PRs | N/A | `format-check` (blocking), `lint` (informational) |
| `docker.yml` | Push to main/dev, tags, PRs | N/A | Multi-registry push (GHCR, Docker Hub) |

**Required Secrets**:
- `SSH_PRIVATE_KEY` - Access to private paladin repo
- `HF_TOKEN` - HuggingFace model access
- `CODECOV_TOKEN` - Coverage reporting (optional)

### Pre-commit Hooks

```bash
make pre-commit-install    # Install hooks (format-check + test-fast)
```

Runs on every commit:
1. `make format-check` - Verify black formatting
2. `make test-fast` - Quick validation

## Docker

### Building

```bash
make docker-build          # Uses SSH agent forwarding
# OR
GH_TOKEN=<token> docker build --secret id=GH_TOKEN .
```

**Important**: Docker builds need either SSH key or `GH_TOKEN` for private paladin dependency.

### Running

```bash
make docker-run                              # Web UI
make docker-run-single SLIDE=example.svs     # Single slide CLI
make docker-shell                            # Interactive debugging
```

## Common Gotchas

1. **Model Loading**: When adding new model inference, add BOTH original function (backward compat) AND batch-aware variant (`*_with_model`).

2. **GPU Detection**: Never hardcode batch sizes or worker counts. Always use `mosaic.hardware` constants.

3. **Tests Without GPU**: All tests must work without GPU. Use fixtures from `conftest.py` for mocking.

4. **Private Dependencies**: When changing dependencies in `pyproject.toml`, ensure SSH keys are configured before `uv sync`.

5. **Gradio State**: For multi-component state updates, return dictionary with component IDs as keys or use `gr.update()`.

6. **Memory Management**: On T4 GPUs, Paladin models are NOT cached. Always call `cache.cleanup()` in finally blocks.

## Data Files

| File | Purpose |
|------|---------|
| `data/paladin_model_map.csv` | Cancer subtype → Paladin model mapping |
| `data/sex_original_to_idx.csv` | Sex encoding (Male=0, Female=1) |
| `data/tissue_site_original_to_idx.csv` | Tissue site location encoding |
| `data/metadata/target_dict.tsv` | Model output class mappings |

## Test Slides & Validation

### test_slides/ Directory

Contains configuration for end-to-end validation with real whole slide images (WSIs):
- **test_samples.json**: Ground truth labels (3 slides, not in repo - must be downloaded)
- **results/**: Output directory for test runs
- **AEON_TEST_SUMMARY.md**: Detailed validation results

**Note**: The actual WSI files (~600MB-1.1GB each) are NOT in the repository and must be downloaded separately.

**Ground Truth Format** (test_samples.json):
```json
[
  {
    "image_id": "881837",
    "cancer_type": "BLCA",
    "site_type": "Primary",
    "sex": "MALE",
    "tissue_site": "Bladder",
    "confidence": 0.9412
  }
]
```

### Validation Workflow (scripts/verify_aeon_results.py)

```bash
# After downloading test slides to test_slides/:

# 1. Run Aeon model on test slides
./scripts/run_aeon_tests.sh

# 2. Verify predictions against ground truth
python scripts/verify_aeon_results.py \
    --test-samples test_slides/test_samples.json \
    --results-dir test_slides/results \
    --output test_slides/verification_report.json

# Quick verification (uses defaults)
python scripts/verify_aeon_results.py
```

**What verify_aeon_results.py does**:
- Loads ground truth from `test_samples.json`
- Reads Aeon predictions from `results/{slide_id}/{slide_id}_aeon_results.csv`
- Compares predicted cancer subtype vs expected ground truth
- Reports accuracy, confidence statistics, pass/fail per slide
- Exits with code 1 if any predictions fail
- Optionally saves detailed JSON report

**When to Use**:
- After model updates or checkpoint exports
- Validating deployment in new environments
- Regression testing for backward compatibility
- Benchmarking model performance changes

See `scripts/README.md` for detailed documentation on all validation scripts.

## Development Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow existing patterns (UI in `ui/`, analysis in `analysis.py`)
3. **Format**: `make format`
4. **Test**: `make test-fast` (quick), then `make test-coverage` (before PR)
5. **Lint**: `make lint`
6. **Commit**: Use conventional commits (`feat:`, `fix:`, `test:`, etc.)
7. **Push & PR**: CI checks must pass (Python 3.10, 3.11, formatting, Docker)

## Performance Optimization

### Batch Processing Speedup

- **Original**: Load models for each slide (50 model loads for 10 slides)
- **Optimized**: Load models once (5 model loads for 10 slides)
- **Result**: 1.25x-1.45x overall speedup, ~90% reduction in model loading overhead

See `BATCH_PROCESSING_IMPLEMENTATION.md` for implementation details.

### Profiling

```bash
make profile SLIDE=test_slides/example.svs  # Profile single slide
make benchmark                              # Run performance benchmarks
```

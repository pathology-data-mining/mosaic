# CLAUDE.md - AI Assistant Guide for Mosaic

**Last Updated**: 2026-01-21
**Repository**: Mosaic - H&E Whole Slide Image Cancer Subtype and Biomarker Inference

This document provides AI assistants with comprehensive information about the Mosaic codebase structure, development workflows, and key conventions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Key Technologies](#key-technologies)
- [Development Workflows](#development-workflows)
- [Code Organization](#code-organization)
- [Testing Conventions](#testing-conventions)
- [CI/CD Pipelines](#cicd-pipelines)
- [Common Tasks](#common-tasks)
- [Important Patterns](#important-patterns)
- [Environment Setup](#environment-setup)
- [Git Workflow](#git-workflow)
- [Critical Files Reference](#critical-files-reference)

---

## Project Overview

**Mosaic** is a production-grade deep learning pipeline for analyzing H&E whole slide images (WSIs) to predict:

1. **Cancer Subtypes** using the Aeon model (100+ cancer types)
2. **Biomarkers** using the Paladin model suite (cancer subtype-specific)

### Key Features

- **Multi-interface**: Web UI (Gradio), CLI (single/batch), programmatic API
- **Hardware-aware**: Auto-detects ZeroGPU (H100), T4, or standard GPUs
- **Batch optimization**: Models loaded once and reused across slides (1.25x-1.45x speedup)
- **Production-ready**: Docker deployment, HuggingFace Spaces support, comprehensive testing

### Technology Stack

- **Language**: Python 3.10-3.11
- **Package Manager**: `uv` (Astral)
- **Web Framework**: Gradio 6.0+
- **Deep Learning**: PyTorch 2.0+, Lightning 2.6+
- **WSI Processing**: OpenSlide, Mussel (tissue segmentation & feature extraction)
- **Models**: CTransPath, Optimus (feature extraction), Aeon (subtyping), Paladin (biomarkers)

---

## Repository Structure

```
mosaic/
├── src/mosaic/                     # Main application code
│   ├── gradio_app.py              # CLI entry point (230 lines)
│   ├── analysis.py                # Core slide analysis pipeline (500+ lines)
│   ├── model_manager.py           # Model caching & memory management (286 lines)
│   ├── hardware.py                # GPU detection & optimization (85 lines)
│   ├── data_directory.py          # Data path management
│   ├── inference/                 # Model inference modules
│   │   ├── aeon.py               # Cancer subtype prediction
│   │   ├── paladin.py            # Biomarker prediction
│   │   └── data.py               # Inference data structures
│   └── ui/                        # Web interface modules
│       ├── app.py                # Gradio interface (474 lines)
│       └── utils.py              # UI utilities (OncoTree, CSV validation)
│
├── tests/                          # Test suite (pytest)
│   ├── conftest.py                # Pytest configuration & mocks
│   ├── test_cli.py                # CLI argument parsing tests
│   ├── test_gradio_app.py         # Gradio interface tests
│   ├── test_model_manager.py      # Model caching tests
│   ├── test_ui_*.py               # UI component tests
│   ├── test_regression_*.py       # End-to-end regression tests
│   ├── benchmark_*.py             # Performance benchmarks
│   └── inference/                 # Inference module tests
│
├── scripts/                        # Utility scripts
│   ├── export_aeon_checkpoint.py
│   └── verify_aeon_results.py
│
├── data/                          # Model metadata & mappings
│   ├── paladin_model_map.csv     # Cancer subtype → model mapping
│   ├── sex_original_to_idx.csv
│   ├── tissue_site_original_to_idx.csv
│   └── metadata/                  # Model-specific metadata
│
├── .github/workflows/             # CI/CD automation
│   ├── tests.yml                 # Test pipeline (Python 3.10, 3.11)
│   ├── code-quality.yml          # Black & Pylint checks
│   └── docker.yml                # Docker build & push to GHCR/Docker Hub
│
├── pyproject.toml                 # Project configuration (uv/pip)
├── Makefile                       # Development commands (264 lines, 40+ targets)
├── Dockerfile                     # Container definition (multi-stage build)
├── app.py                         # HuggingFace Spaces entry point
├── README.md                      # User documentation (430 lines)
├── ARCHITECTURE.md                # Code organization guide
├── CONTRIBUTING.md                # Developer guide
└── BATCH_PROCESSING_IMPLEMENTATION.md  # Batch optimization details
```

### Source Code Organization

**Module Hierarchy:**
```
mosaic.gradio_app:main()
  ├── download_and_process_models()
  ├── analyze_slide() [from analysis]
  │     ├── segment_tissue() [from mussel]
  │     ├── get_features() [CTransPath, Optimus]
  │     ├── filter_features() [Marker Classifier]
  │     ├── run_aeon() [from inference]
  │     └── run_paladin() [from inference]
  └── launch_gradio() [from ui]
        ├── analyze_slides() [batch coordinator]
        └── validate_settings() [from ui.utils]
```

---

## Key Technologies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `gradio` | >=6.0.0 | Web UI framework |
| `torch` | >=2.0.0 | Deep learning backend |
| `lightning` | >=2.6.0 | PyTorch trainer |
| `mussel[torch-gpu]` | git | Tissue segmentation & feature extraction |
| `paladin` | git (private) | Biomarker prediction models |
| `openslide-python` | - | Whole slide image reading |
| `huggingface-hub` | - | Model downloading |
| `loguru` | >=0.7.3 | Enhanced logging |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| `black` | Code formatting (PEP 8) |
| `pylint` | Linting |
| `pytest` | Test framework |
| `pytest-cov` | Coverage reporting |
| `pytest-mock` | Test mocking |

### Private Dependencies

**Important**: This project depends on two private repositories:

1. **Paladin** (`ssh://git@github.com/pathology-data-mining/paladin.git@dev`)
   - Biomarker prediction models
   - Requires SSH key or GH_TOKEN

2. **Mussel** (`https://github.com/pathology-data-mining/Mussel.git@mosaic-dev`)
   - Tissue segmentation and feature extraction
   - Public repository

**Installation Methods:**
```bash
# Local dev (SSH)
uv sync

# Docker build (token-based)
GH_TOKEN=<token> docker build --secret id=GH_TOKEN .
```

---

## Development Workflows

### Makefile Targets (40+ commands)

The Makefile is the primary interface for development tasks. Key targets:

#### Installation & Setup
```bash
make install            # Production dependencies only
make install-dev        # Production + development dependencies
```

#### Testing (Most Common)
```bash
make test               # Run all tests with coverage
make test-fast          # Without coverage (2-3x faster)
make test-coverage      # Detailed coverage + HTML report
make test-ui            # UI component tests only
make test-cli           # CLI routing tests only
make test-verbose       # With print statements (-s flag)
make test-specific TEST=tests/test_file.py::test_name  # Single test
```

#### Code Quality
```bash
make format             # Auto-format with black
make format-check       # Check formatting (CI-safe)
make lint               # Pylint on src/mosaic/
make lint-strict        # Pylint on src/ + tests/
make quality            # format-check + lint
```

#### Running the Application
```bash
make run-ui                                    # Launch web interface
make run-ui-public                             # Web UI with public sharing
make run-single SLIDE=path.svs OUTPUT=dir/     # Single slide CLI
make run-batch CSV=settings.csv OUTPUT=dir/    # Batch processing CLI
```

#### Docker Workflows
```bash
make docker-build       # Build with SSH forwarding
make docker-run         # Run web UI container
make docker-shell       # Interactive shell in container
make docker-push        # Push to registries
make docker-prune       # Clean build cache
```

#### Development Utilities
```bash
make shell              # Python REPL
make ipython            # IPython REPL
make notebook           # Jupyter notebook
make check-deps         # List outdated dependencies
make profile SLIDE=path.svs  # Profile single slide analysis
make benchmark          # Performance benchmarks
```

#### Git Hooks
```bash
make pre-commit-install    # Install hooks (format-check + test-fast)
make pre-commit-uninstall  # Remove hooks
```

### Quick Reference

**Most Common Development Commands:**
1. `make install-dev` - Set up environment
2. `make format` - Format code before committing
3. `make test-fast` - Quick validation
4. `make test-coverage` - Before PR submission
5. `make run-ui` - Test web interface locally

---

## Code Organization

### Architecture Principles

1. **Separation of Concerns**: UI, analysis, and CLI logic are in separate modules
2. **Hardware Abstraction**: `hardware.py` provides centralized GPU detection
3. **Model Caching**: `model_manager.py` implements load-once-reuse-many pattern
4. **Batch Optimization**: Automatic batch mode for multi-slide processing
5. **Backward Compatibility**: Original functions unchanged, batch functions parallel

### Key Modules

#### `src/mosaic/hardware.py` - Hardware Detection
**Purpose**: Centralized GPU detection and configuration

**Exports:**
- `IS_ZEROGPU` - True if running on HuggingFace ZeroGPU (H100)
- `IS_T4_GPU` - True if running on NVIDIA T4
- `GPU_TYPE` - Human-readable GPU name
- `DEFAULT_BATCH_SIZE` - Hardware-appropriate batch size
- `DEFAULT_NUM_WORKERS` - Hardware-appropriate worker count
- `spaces` - Decorator for ZeroGPU allocation (no-op if not available)

**Pattern:**
```python
from mosaic.hardware import IS_T4_GPU, DEFAULT_BATCH_SIZE, spaces

if IS_T4_GPU:
    # Use aggressive memory management
    batch_size = DEFAULT_BATCH_SIZE
```

#### `src/mosaic/model_manager.py` - Model Lifecycle
**Purpose**: Pre-load and cache models for batch processing

**Key Class: `ModelCache`**
```python
class ModelCache:
    def load_all_models(self) -> None:
        """Load core models once (CTransPath, Optimus, Aeon, Marker Classifier)"""

    def load_paladin_model_for_inference(self, cancer_subtype: str):
        """Lazy-load Paladin models (with T4-aware cleanup)"""

    def cleanup(self) -> None:
        """Release all GPU memory"""
```

**Memory Management:**
- **T4 GPU**: Aggressive cleanup (Paladin models loaded/deleted per inference)
- **A100 GPU**: Caching strategy (Paladin models cached for reuse)

#### `src/mosaic/analysis.py` - Core Pipeline
**Purpose**: Orchestrate slide analysis workflow

**Main Functions:**
- `analyze_slide()` - Single slide (original, backward compatible)
- `analyze_slide_with_models()` - Batch-aware (uses pre-loaded models)

**Data Flow:**
```
WSI → Tissue Segmentation (Mussel) → CTransPath Features →
Marker Classification → Optimus Features → Aeon (Subtypes) → Paladin (Biomarkers)
```

#### `src/mosaic/ui/app.py` - Web Interface
**Purpose**: Gradio web UI

**Key Functions:**
- `launch_gradio()` - Main entry point
- `analyze_slides()` - Event handler (auto-detects batch vs single)
- `set_cancer_subtype_maps()` - Global cancer subtype mapping management

**Automatic Batch Detection:**
```python
if len(slides) > 1:
    # Use batch mode (model caching)
else:
    # Use single slide mode (original pipeline)
```

#### `src/mosaic/ui/utils.py` - UI Utilities
**Key Functions:**
- `validate_settings()` - CSV settings validation
- `load_settings()` - Parse CSV configuration
- `get_oncotree_code_name()` - OncoTree API integration (with caching)
- `create_user_directory()` - Session directory management

#### `src/mosaic/inference/` - Model Inference

**`aeon.py` - Cancer Subtype Prediction:**
- `run()` - Original function (loads model each time)
- `run_with_model()` - Batch-aware (uses pre-loaded model)

**`paladin.py` - Biomarker Prediction:**
- `run()` - Original function
- `run_with_models()` - Batch-aware function

**`data.py` - Data Structures:**
- `SiteType` enum (Primary/Metastatic)
- `TileFeatureTensorDataset` (PyTorch DataLoader)
- `CANCER_TYPE_TO_INT_MAP` (100+ cancer types)

---

## Testing Conventions

### Test Organization

**Framework**: pytest with coverage reporting

**Configuration** (`pyproject.toml`):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src/mosaic --cov-report=term-missing"
```

### Test Categories

| Category | Files | Purpose |
|----------|-------|---------|
| **Unit Tests** | `test_model_manager.py`, `test_data.py` | Component isolation |
| **Integration Tests** | `test_cli.py`, `test_ui_*.py` | Multi-component workflows |
| **Regression Tests** | `test_regression_single_slide.py` | Backward compatibility |
| **Benchmarks** | `benchmark_batch_performance.py` | Performance validation |

### Key Testing Patterns

#### Mocking Heavy Dependencies

**`tests/conftest.py`** provides mocks for:
- `mussel` - Tissue segmentation & feature extraction
- `huggingface_hub` - Model downloading
- `gradio` - UI components (with visibility tracking)

**Example:**
```python
# Tests run WITHOUT downloading models or processing real slides
@pytest.fixture
def mock_mussel_segment(monkeypatch):
    """Mock tissue segmentation to return fake coordinates"""
    def mock_segment(*args, **kwargs):
        return (fake_coords, fake_attrs)
    monkeypatch.setattr("mosaic.analysis.segment_tissue", mock_segment)
```

#### Testing Without GPU

All tests are designed to run **without GPU access**:
```python
# Good: Mock model loading
def test_analysis(mock_aeon_model, mock_paladin_model):
    result = analyze_slide(slide_path, settings)
    assert result is not None

# Bad: Requires actual GPU and models
def test_analysis():
    result = analyze_slide(slide_path, settings)  # Will fail in CI
```

### Running Tests

```bash
# All tests with coverage
make test

# Fast (no coverage)
make test-fast

# Specific test
make test-specific TEST=tests/test_cli.py::test_single_slide_mode

# With print statements
make test-verbose

# Coverage report with HTML
make test-coverage
# Open htmlcov/index.html to view
```

### Writing New Tests

**Template:**
```python
"""Test module for [component name]."""
import pytest
from mosaic.module import function_to_test


def test_basic_functionality(mock_dependency):
    """Test [describe what is being tested]."""
    # Arrange
    input_data = ...

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected_output


def test_error_handling():
    """Test error handling for invalid input."""
    with pytest.raises(ValueError, match="expected error message"):
        function_to_test(invalid_input)
```

---

## CI/CD Pipelines

### GitHub Actions Workflows

#### 1. **Tests** (`.github/workflows/tests.yml`)
**Triggers**: Push to main/dev, PRs, manual dispatch

**Matrix**: Python 3.10, 3.11

**Steps:**
1. Checkout with full history
2. Set up SSH key for private repos
3. Install Python & uv
4. Install dependencies (`uv sync`)
5. Run `make test-coverage`
6. Upload coverage to Codecov
7. Upload HTML coverage reports (artifacts)

**Required Secrets:**
- `SSH_PRIVATE_KEY` - Access to private paladin repo
- `HF_TOKEN` - HuggingFace API access
- `CODECOV_TOKEN` - Coverage reporting (optional)

#### 2. **Code Quality** (`.github/workflows/code-quality.yml`)
**Checks:**
- `format-check` - Black formatting validation (blocking)
- `lint` - Pylint checks (informational, non-blocking)

#### 3. **Docker Build & Push** (`.github/workflows/docker.yml`)
**Triggers**: Push to main/dev, tags, PRs

**Registries:**
- **GitHub Container Registry** (ghcr.io) - All branches
- **Docker Hub** (docker.io/mskmind/mosaic) - Main branch only

**Tags:**
- `main` - Latest stable
- `dev` - Development branch
- `pr-123` - Pull request builds
- `sha-abc1234` - Git commit SHA
- `v1.2.3` - Semantic version tags

**Required Secrets:**
- `SSH_PRIVATE_KEY` or `GH_TOKEN` - Private repo access
- `DOCKER_HUB_USERNAME` - Docker Hub login (optional)
- `DOCKER_HUB_TOKEN` - Docker Hub token (optional)

**Build Process:**
```yaml
# Uses BuildKit with SSH secret mounting
docker build --ssh default --secret id=GH_TOKEN -t mosaic .
```

### Pre-commit Hooks

Install local pre-commit hooks:
```bash
make pre-commit-install
```

**Runs on commit:**
1. `make format-check` - Verify formatting
2. `make test-fast` - Quick validation

---

## Common Tasks

### Task 1: Add a New Feature

**Steps:**
1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement changes**
   - Follow existing code organization (UI in `ui/`, analysis in `analysis.py`, etc.)
   - Add type hints where appropriate
   - Use `loguru` logger for logging

3. **Write tests**
   ```bash
   # Create test file
   touch tests/test_your_feature.py

   # Follow existing test patterns
   # Mock heavy dependencies (models, mussel, etc.)
   ```

4. **Format and validate**
   ```bash
   make format        # Auto-format
   make test-fast     # Quick validation
   make lint          # Check code quality
   ```

5. **Run full test suite**
   ```bash
   make test-coverage
   ```

6. **Commit and push**
   ```bash
   git add .
   git commit -m "Add [feature description]"
   git push origin feature/your-feature-name
   ```

### Task 2: Fix a Bug

**Steps:**
1. **Reproduce the bug**
   ```bash
   make run-ui  # Or make run-single SLIDE=... OUTPUT=...
   ```

2. **Write regression test first** (TDD approach)
   ```python
   def test_bug_fix():
       """Test that [bug] is fixed."""
       result = buggy_function(problematic_input)
       assert result == expected_correct_output
   ```

3. **Fix the bug**
   - Modify relevant module
   - Ensure test passes

4. **Validate**
   ```bash
   make test-specific TEST=tests/test_bug_fix.py::test_bug_fix
   make test-fast  # Ensure no regressions
   ```

### Task 3: Update Dependencies

**Check for outdated packages:**
```bash
make check-deps
```

**Update specific package:**
```bash
# Edit pyproject.toml to change version constraint
# Then regenerate lock file
make lock
```

**Update all dependencies:**
```bash
make update-deps  # CAUTION: May break compatibility
make test          # Validate everything still works
```

### Task 4: Profile Performance

**Profile single slide:**
```bash
make profile SLIDE=test_slides/example.svs
```

**Run benchmarks:**
```bash
make benchmark
```

**Custom profiling:**
```bash
python -m cProfile -o profile.stats -m mosaic.gradio_app --slide-path slide.svs --output-dir output/
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(30)"
```

### Task 5: Build and Test Docker Image

**Build locally:**
```bash
make docker-build
```

**Run container (web UI):**
```bash
make docker-run
```

**Run container (single slide):**
```bash
make docker-run-single SLIDE=example.svs
```

**Interactive debugging:**
```bash
make docker-shell
# Inside container:
python -m mosaic.gradio_app --slide-path /app/data/slide.svs --output-dir /app/output
```

### Task 6: Download Models

**Download all models:**
```bash
make download-models
```

**Or via CLI:**
```bash
mosaic --download-models-only
```

**Models are cached to:**
- `$HF_HOME/hub/` (if HF_HOME set)
- `~/.cache/huggingface/hub/` (default)

---

## Important Patterns

### Pattern 1: Hardware-Aware Batch Sizing

**Always use hardware detection for GPU operations:**

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

**Never hardcode batch sizes or worker counts.**

### Pattern 2: Model Caching for Batch Processing

**Original (single slide):**
```python
def analyze_slide(slide_path, settings):
    # Load models each time
    aeon_model = load_aeon_model()
    # ... process ...
    return results
```

**Batch-optimized:**
```python
def analyze_slides_batch(slides, settings):
    # Load models ONCE
    cache = ModelCache()
    cache.load_all_models()

    try:
        for slide in slides:
            # Reuse pre-loaded models
            result = analyze_slide_with_models(slide, cache, settings)
    finally:
        cache.cleanup()  # Always cleanup
```

### Pattern 3: Gradio Component Visibility

**Use update() for dynamic UI:**
```python
import gradio as gr

def on_upload(files):
    if len(files) > 1:
        return gr.update(visible=True)  # Show batch options
    else:
        return gr.update(visible=False)  # Hide batch options

# In interface definition
with gr.Column(visible=False) as batch_column:
    batch_options = gr.Checkbox(label="Batch Options")

upload_button.upload(on_upload, inputs=files, outputs=batch_column)
```

### Pattern 4: Error Handling with Logging

**Use loguru for structured logging:**
```python
from loguru import logger

def risky_operation(input_data):
    try:
        logger.info(f"Starting operation with {len(input_data)} items")
        result = process(input_data)
        logger.success(f"Operation completed successfully")
        return result
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during operation: {e}")
        raise
```

**Never use print() for logging in production code.**

### Pattern 5: Backward Compatibility

**When adding batch optimization:**

```python
# Original function - UNCHANGED
def analyze_slide(slide_path, settings):
    """Single slide analysis (original, backward compatible)."""
    # ... original implementation ...

# New function - PARALLEL
def analyze_slide_with_models(slide_path, model_cache, settings):
    """Single slide analysis using pre-loaded models (batch-aware)."""
    # ... new implementation using model_cache ...

# Batch coordinator
def analyze_slides_batch(slides, settings):
    """Batch analysis with model caching."""
    cache = ModelCache()
    cache.load_all_models()

    try:
        results = []
        for slide in slides:
            result = analyze_slide_with_models(slide, cache, settings)
            results.append(result)
        return results
    finally:
        cache.cleanup()
```

**Key: Original function unchanged, new function added in parallel.**

### Pattern 6: Configuration via Environment Variables

**Respect environment variables:**
```python
import os

# HuggingFace token
HF_TOKEN = os.environ.get("HF_TOKEN")

# HuggingFace cache directory
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Data directory
DATA_DIR = os.environ.get("MOSAIC_DATA_DIR", "./data")

# Gradio server configuration
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
```

### Pattern 7: ZeroGPU Decorator

**For functions that need GPU on HuggingFace Spaces:**
```python
from mosaic.hardware import spaces

@spaces.GPU(duration=120)  # Allocate GPU for 120 seconds
def gpu_intensive_operation(data):
    # This runs on GPU when available
    return process_on_gpu(data)
```

**The decorator is a no-op when not on HuggingFace Spaces.**

---

## Environment Setup

### Prerequisites

- **Python**: 3.10 or 3.11 (3.12 not yet supported)
- **uv**: Package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **GPU**: NVIDIA CUDA-capable GPU (optional but recommended)
- **SSH Key**: Access to private paladin repository

### Setup Steps

#### 1. Clone Repository
```bash
git clone https://github.com/pathology-data-mining/mosaic.git
cd mosaic
```

#### 2. Configure SSH Key

**For private repository access:**
```bash
# Generate SSH key (if needed)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub account
cat ~/.ssh/id_ed25519.pub
# Copy and add to GitHub: Settings → SSH and GPG keys
```

#### 3. Install Dependencies
```bash
# Development installation (recommended)
make install-dev

# Or production only
make install

# Activate virtual environment
source .venv/bin/activate
```

#### 4. Set Environment Variables
```bash
# Required for model access
export HF_TOKEN="your_huggingface_token_here"

# Optional: Set cache directory
export HF_HOME="/path/to/huggingface/cache"

# Optional: Set data directory
export MOSAIC_DATA_DIR="/path/to/mosaic/data"

# Optional: Set Gradio server port
export GRADIO_SERVER_PORT="7860"
```

**Add to `~/.bashrc` or `~/.zshrc` for persistence.**

#### 5. Verify Installation
```bash
# Check version
python -c "import mosaic; print('Mosaic installed successfully')"

# Run tests
make test-fast

# Download models (requires HF_TOKEN)
make download-models
```

### Docker Setup

**Build Docker image:**
```bash
# With SSH key
make docker-build

# Or manually
./build.sh
```

**Run Docker container:**
```bash
# Web UI
make docker-run

# Single slide
docker run -v $(PWD)/data:/app/data mskmind/mosaic \
  --slide-path /app/data/slide.svs --output-dir /app/output
```

### HuggingFace Spaces Deployment

**Requirements:**
1. Added to PDM Group on HuggingFace
2. HuggingFace access token with read permissions

**Deployment:**
1. Create new Space (Gradio SDK, ZeroGPU hardware)
2. Push code to Space repository
3. Add `HF_TOKEN` secret in Space settings
4. App will auto-start and download models

---

## Git Workflow

### Branch Strategy

- **`main`** - Production-ready code
- **`dev`** - Development branch
- **`feature/name`** - Feature branches
- **`fix/name`** - Bug fix branches

### Commit Message Convention

**Format:**
```
<type>: <short description>

<optional detailed explanation>

<optional footer>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code restructuring (no functional changes)
- `test:` - Add or update tests
- `docs:` - Documentation changes
- `style:` - Code formatting (black, etc.)
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

**Examples:**
```
feat: add batch processing optimization for multi-slide analysis

Implemented ModelCache class to load models once and reuse across
slides in a batch. This provides 1.25x-1.45x speedup for batches.

Closes #42

---

fix: resolve T4 GPU out-of-memory errors in Paladin inference

Added aggressive memory management for T4 GPUs that deletes Paladin
models immediately after inference instead of caching.

---

test: add regression tests for single slide analysis

Ensures backward compatibility with original analyze_slide() function.
```

### Pull Request Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: add your feature"
   ```

3. **Run quality checks**
   ```bash
   make format
   make test-coverage
   make lint
   ```

4. **Push to remote**
   ```bash
   git push origin feature/your-feature
   ```

5. **Create Pull Request**
   - Go to GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in PR template:
     - **Description**: What does this PR do?
     - **Testing**: How was it tested?
     - **Related Issues**: Closes #123

6. **Wait for CI checks**
   - Tests must pass (Python 3.10, 3.11)
   - Code quality checks must pass
   - Docker build must succeed

7. **Address review feedback**
   ```bash
   # Make changes
   git add .
   git commit -m "fix: address review feedback"
   git push origin feature/your-feature
   ```

8. **Merge** (after approval)
   - Squash and merge (recommended for feature branches)
   - Regular merge (for larger changes)

---

## Critical Files Reference

### Configuration Files

| File | Purpose | Key Contents |
|------|---------|--------------|
| `pyproject.toml` | Project configuration | Dependencies, scripts, pytest config, pylint settings |
| `uv.lock` | Dependency lock file | Exact versions of all packages |
| `Makefile` | Development commands | 40+ targets for testing, building, running |
| `Dockerfile` | Container definition | Multi-stage build, SSH secret mounting |
| `.github/workflows/*.yml` | CI/CD pipelines | Tests, code quality, Docker builds |

### Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | User documentation | End users |
| `ARCHITECTURE.md` | Code organization | Developers |
| `CONTRIBUTING.md` | Contribution guide | Contributors |
| `BATCH_PROCESSING_IMPLEMENTATION.md` | Batch optimization details | Developers |
| `CLAUDE.md` (this file) | AI assistant guide | AI assistants |

### Data Files

| File | Purpose |
|------|---------|
| `data/paladin_model_map.csv` | Cancer subtype → Paladin model mapping |
| `data/sex_original_to_idx.csv` | Sex encoding (Male=0, Female=1) |
| `data/tissue_site_original_to_idx.csv` | Tissue site location encoding |
| `data/metadata/target_dict.tsv` | Model output class mappings |
| `data/metadata/int_to_name_class_mapping.tsv` | Cancer type name mappings |

### Entry Points

| File | CLI Command | Purpose |
|------|-------------|---------|
| `mosaic` script | `mosaic` | Main CLI entry point |
| `src/mosaic/gradio_app.py:main` | `mosaic` | CLI implementation |
| `src/mosaic/inference/aeon.py:main` | `aeon_inference` | Standalone Aeon inference |
| `src/mosaic/inference/paladin.py:main` | `paladin_inference` | Standalone Paladin inference |
| `app.py` | (HF Spaces) | HuggingFace Spaces entry point |

---

## Quick Reference Card

### Most Common Commands

```bash
# Setup
make install-dev
export HF_TOKEN="your_token"

# Development
make format           # Before committing
make test-fast        # Quick validation
make test-coverage    # Before PR

# Running
make run-ui                                  # Web interface
make run-single SLIDE=x.svs OUTPUT=out/      # CLI single
make run-batch CSV=settings.csv OUTPUT=out/  # CLI batch

# Docker
make docker-build     # Build image
make docker-run       # Run container
make docker-shell     # Debug container

# Quality
make format-check     # CI formatting check
make lint             # Code quality
make quality          # All checks
```

### File Location Quick Lookup

| What | Where |
|------|-------|
| CLI entry point | `src/mosaic/gradio_app.py` |
| Core analysis | `src/mosaic/analysis.py` |
| Web UI | `src/mosaic/ui/app.py` |
| GPU detection | `src/mosaic/hardware.py` |
| Model caching | `src/mosaic/model_manager.py` |
| Cancer subtype inference | `src/mosaic/inference/aeon.py` |
| Biomarker inference | `src/mosaic/inference/paladin.py` |
| Test mocks | `tests/conftest.py` |
| Main tests | `tests/test_*.py` |

### Common Error Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'paladin'` | Ensure SSH key configured, run `uv sync` |
| `HuggingFace authentication failed` | Set `HF_TOKEN` environment variable |
| `CUDA out of memory` | Reduce `--num-workers`, use T4-aware settings |
| `Port 7860 already in use` | Set `GRADIO_SERVER_PORT=7861` |
| `Tests fail with import errors` | Run `make install-dev` to install test dependencies |
| `Docker build fails on paladin` | Ensure `GH_TOKEN` secret or SSH key is configured |

---

## Change Log

| Date | Changes |
|------|---------|
| 2026-01-21 | Initial CLAUDE.md creation |

---

**End of AI Assistant Guide**

For questions or updates to this guide, please submit a pull request or create an issue.

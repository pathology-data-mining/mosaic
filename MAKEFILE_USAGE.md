# Makefile Usage Guide

This document provides detailed information about the Makefile targets available in the Mosaic project.

## Quick Start

```bash
# See all available commands
make help

# Setup development environment
make install-dev

# Run tests
make test

# Launch web interface
make run-ui
```

## Development Setup

### `make install`
Install production dependencies only (no dev tools).
```bash
make install
```

### `make install-dev`
Install all dependencies including development tools (pytest, ruff, etc.).
```bash
make install-dev
```

## Testing

### `make test`
Run full test suite with coverage reporting.
```bash
make test
```

### `make test-fast`
Run tests without coverage (faster execution).
```bash
make test-fast
```

### `make test-coverage`
Run tests with detailed coverage report (terminal + HTML).
```bash
make test-coverage
# View HTML report at: htmlcov/index.html
```

### `make test-ui`
Run only UI-related tests.
```bash
make test-ui
```

### `make test-cli`
Run only CLI-related tests.
```bash
make test-cli
```

### `make test-verbose`
Run tests with verbose output and show print statements.
```bash
make test-verbose
```

### `make test-specific`
Run a specific test file, class, or method.
```bash
# Run specific test file
make test-specific TEST=tests/test_cli.py

# Run specific test class
make test-specific TEST=tests/test_cli.py::TestArgumentParsing

# Run specific test method
make test-specific TEST=tests/test_cli.py::TestArgumentParsing::test_no_arguments_launches_web_interface
```

## Code Quality

### `make lint`
Check code for linting issues using pylint (src only for speed).
```bash
make lint
```

### `make lint-strict`
Run pylint on both src and tests (slower but comprehensive).
```bash
make lint-strict
```

### `make format`
Format code using black formatter.
```bash
make format
```

### `make format-check`
Check if code is properly formatted without making changes.
```bash
make format-check
```

### `make quality`
Run all code quality checks (format-check + lint).
```bash
make quality
```

## Running the Application

### `make run-ui`
Launch the Gradio web interface locally.
```bash
make run-ui
# Open browser to http://localhost:7860
```

### `make run-ui-public`
Launch Gradio web interface with public sharing enabled.
```bash
make run-ui-public
# Returns a public gradio.app URL for sharing
```

### `make run-single`
Process a single slide from the command line.
```bash
make run-single SLIDE=data/my_slide.svs OUTPUT=output/
```

### `make run-batch`
Process multiple slides from a CSV file.
```bash
make run-batch CSV=data/settings.csv OUTPUT=output/
```

## Docker

### `make docker-build`
Build Docker image for Mosaic.
```bash
make docker-build

# Build with custom tag
make docker-build DOCKER_TAG=v1.0.0

# Build with custom image name
make docker-build DOCKER_IMAGE_NAME=my-mosaic DOCKER_TAG=latest
```

### `make docker-build-no-cache`
Build Docker image without using cache (useful for clean builds).
```bash
make docker-build-no-cache
```

### `make docker-run`
Run Docker container in web UI mode.
```bash
make docker-run
# Access at http://localhost:7860
```

### `make docker-run-single`
Run Docker container to process a single slide.
```bash
# Place your slide in ./data directory first
make docker-run-single SLIDE=my_slide.svs
# Results will be in ./output directory
```

### `make docker-run-batch`
Run Docker container for batch processing.
```bash
# Place CSV and slides in ./data directory
make docker-run-batch CSV=settings.csv
# Results will be in ./output directory
```

### `make docker-shell`
Open an interactive shell inside the Docker container.
```bash
make docker-shell
```

### `make docker-tag`
Tag Docker image for pushing to a registry.
```bash
make docker-tag DOCKER_REGISTRY=docker.io/myusername
```

### `make docker-push`
Push Docker image to registry.
```bash
# Set your registry first
make docker-push DOCKER_REGISTRY=docker.io/myusername DOCKER_TAG=latest
```

### `make docker-clean`
Remove local Docker image.
```bash
make docker-clean
```

### `make docker-prune`
Clean up Docker build cache to free space.
```bash
make docker-prune
```

## Cleanup

### `make clean`
Remove Python cache files and build artifacts.
```bash
make clean
```

### `make clean-outputs`
Remove generated output files (masks, CSVs).
```bash
make clean-outputs
```

### `make clean-all`
Remove all artifacts, cache, and Docker images.
```bash
make clean-all
```

## Model Management

### `make download-models`
Explicitly download required models from HuggingFace.
```bash
make download-models
# Note: Models are automatically downloaded on first run
```

## CI/CD

### `make ci-test`
Run complete CI test suite (install deps, test with coverage, lint).
```bash
make ci-test
```

### `make ci-docker`
Build Docker image for CI pipeline.
```bash
make ci-docker
```

## Development Utilities

### `make shell`
Open Python shell with project in path.
```bash
make shell
```

### `make ipython`
Open IPython shell with project in path.
```bash
make ipython
```

### `make notebook`
Start Jupyter notebook server.
```bash
make notebook
```

### `make check-deps`
Check for outdated dependencies.
```bash
make check-deps
```

### `make update-deps`
Update all dependencies (use with caution).
```bash
make update-deps
```

### `make lock`
Update uv.lock file.
```bash
make lock
```

## Git Hooks

### `make pre-commit-install`
Install pre-commit hooks that run lint, format-check, and test-fast before each commit.
```bash
make pre-commit-install
```

### `make pre-commit-uninstall`
Remove pre-commit hooks.
```bash
make pre-commit-uninstall
```

## Information

### `make info`
Display project information and key commands.
```bash
make info
```

### `make version`
Show version information.
```bash
make version
```

### `make tree`
Show project directory structure (requires `tree` command).
```bash
make tree
```

## Performance

### `make profile`
Profile single slide analysis to identify performance bottlenecks.
```bash
make profile SLIDE=tests/testdata/948176.svs
# Creates profile.stats file with profiling data
```

### `make benchmark`
Run performance benchmarks on test slide.
```bash
make benchmark
# Times full analysis pipeline
```

## Common Workflows

### Setting up for development
```bash
# 1. Install dependencies
make install-dev

# 2. Run tests to ensure everything works
make test

# 3. Install pre-commit hooks
make pre-commit-install
```

### Before committing changes
```bash
# Run quality checks
make quality

# Run tests
make test

# Clean up
make clean
```

### Preparing a release
```bash
# Run full CI suite
make ci-test

# Build Docker image
make docker-build DOCKER_TAG=v1.0.0

# Test Docker image
make docker-run DOCKER_TAG=v1.0.0

# Push to registry
make docker-push DOCKER_REGISTRY=your-registry DOCKER_TAG=v1.0.0
```

### Processing slides
```bash
# Web UI (recommended for exploration)
make run-ui

# Single slide (CLI)
make run-single SLIDE=data/sample.svs OUTPUT=results/

# Batch processing (CLI)
make run-batch CSV=data/batch_settings.csv OUTPUT=results/

# Using Docker
make docker-build
make docker-run-batch CSV=batch_settings.csv
```

## Customization

You can customize Makefile behavior by setting environment variables or editing the Makefile:

```bash
# Custom Docker registry
export DOCKER_REGISTRY=my-registry.com/username

# Custom image name
export DOCKER_IMAGE_NAME=my-custom-mosaic

# Then use make commands as normal
make docker-build
make docker-push
```

## Troubleshooting

### Tests fail
```bash
# Run with verbose output
make test-verbose

# Run specific failing test
make test-specific TEST=tests/test_file.py::test_name
```

### Docker build fails
```bash
# Build without cache
make docker-build-no-cache

# Check Docker logs
docker logs <container-id>
```

### Permission errors
```bash
# Clean and rebuild
make clean-all
make install-dev
```

### Out of disk space
```bash
# Clean Docker cache
make docker-prune

# Clean project artifacts
make clean
```

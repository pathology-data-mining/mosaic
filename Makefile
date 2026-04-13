.PHONY: help install install-dev test test-coverage test-verbose lint format clean docker-build docker-run docker-run-cli docker-push docker-clean run-ui run-cli test-e2e test-e2e-headed

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_IMAGE_NAME := mskmind/mosaic
DOCKER_TAG := latest
DOCKER_REGISTRY := docker.io
PYTHON := uv run python
PYTEST := uv run pytest
BLACK := uv run black
PYLINT := uv run pylint

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development Setup

install: ## Install production dependencies using uv
	uv sync --no-dev

install-dev: ## Install development dependencies using uv
	uv sync

##@ Testing

test: ## Run all tests (unit + integration)
	$(PYTEST) tests/ -v && $(PYTEST) integration/ -v --no-cov -s

test-fast: ## Run unit tests without coverage (faster)
	$(PYTEST) tests/ -v --no-cov

test-coverage: ## Run tests with detailed coverage report
	$(PYTEST) tests/ -v --cov=src/mosaic --cov-report=term-missing --cov-report=html

test-integration: ## Run Gradio UI integration tests (starts real server, no GPU needed)
	$(PYTEST) integration/ -v --no-cov -s

test-ui: ## Run only UI tests
	$(PYTEST) tests/test_ui_components.py tests/test_ui_events.py -v

test-cli: ## Run only CLI tests
	$(PYTEST) tests/test_cli.py -v

test-verbose: ## Run tests with verbose output and show print statements
	$(PYTEST) tests/ -vv -s

test-specific: ## Run specific test (usage: make test-specific TEST=tests/test_cli.py::TestClass::test_method)
	$(PYTEST) $(TEST) -v

test-e2e: ## Run end-to-end browser tests (requires GPU, models, and test slide)
	$(PYTEST) tests/e2e/ --e2e -v --no-cov

test-e2e-headed: ## Run end-to-end browser tests with visible browser window
	$(PYTEST) tests/e2e/ --e2e -v --no-cov --headed

test-watch: ## Run tests in watch mode (requires pytest-watch)
	$(PYTEST) tests/ --watch

##@ Code Quality

lint: ## Run linting checks with pylint
	$(PYLINT) src/mosaic/

lint-strict: ## Run pylint on both src and tests
	$(PYLINT) src/mosaic/ tests/

format: ## Format code with black
	$(BLACK) src/ tests/

format-check: ## Check code formatting without making changes
	$(BLACK) --check src/ tests/

quality: format-check lint ## Run all code quality checks

##@ Application

run-ui: ## Launch Gradio web interface
	$(PYTHON) -m mosaic.gradio_app

run-ui-public: ## Launch Gradio web interface with public sharing
	$(PYTHON) -m mosaic.gradio_app --share

run-single: ## Run single slide analysis (usage: make run-single SLIDE=path/to/slide.svs OUTPUT=output_dir [ARGS="--extra-args"])
	$(PYTHON) -m mosaic.gradio_app --slide-path $(SLIDE) --output-dir $(OUTPUT) $(ARGS)

run-batch: ## Run batch analysis from CSV (usage: make run-batch CSV=settings.csv OUTPUT=output_dir [ARGS="--extra-args"])
	$(PYTHON) -m mosaic.gradio_app --slide-csv $(CSV) --output-dir $(OUTPUT) $(ARGS)

##@ Docker

# Docker run options matching the mosaic entrypoint script
DOCKER_GPU_ARGS := --gpus=all --runtime=nvidia
HF_CACHE_DIR := $(HOME)/.cache/huggingface
DOCKER_COMMON_ARGS := --shm-size=3G --env HF_TOKEN="$(HF_TOKEN)" --env HF_HOME=/mnt/hf_cache --env TRANSFORMERS_CACHE=/mnt/hf_cache --user $(shell id -u):$(shell id -g) -v $(HF_CACHE_DIR):/mnt/hf_cache

docker-build: ## Build Docker image
	@echo "Building Docker image..."
	@./build.sh

docker-build-no-cache: ## Build Docker image without cache
	@echo "Building Docker image (no cache)..."
	@DOCKER_BUILDKIT=1 docker build --no-cache -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .

docker-run: ## Run Docker container (web UI mode)
	@test -n "$(HF_TOKEN)" || (echo "Error: HF_TOKEN environment variable is not set" && exit 1)
	docker run -it --rm \
		$(DOCKER_GPU_ARGS) \
		$(DOCKER_COMMON_ARGS) \
		-p 7860:7860 \
		-v $(PWD)/data:/mnt/data:ro \
		-v $(PWD)/output:/mnt/output \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

docker-run-cli: ## Run Docker container with mosaic CLI (usage: make docker-run-cli ARGS="--help")
	@test -n "$(HF_TOKEN)" || (echo "Error: HF_TOKEN environment variable is not set" && exit 1)
	docker run -it --rm \
		$(DOCKER_GPU_ARGS) \
		$(DOCKER_COMMON_ARGS) \
		-v $(PWD)/data:/mnt/data:ro \
		-v $(PWD)/output:/mnt/output \
		--entrypoint mosaic \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		$(ARGS)

docker-run-single: ## Run Docker container (single slide mode, usage: make docker-run-single SLIDE=path/to/slide.svs [ARGS="--extra-args"])
	@test -n "$(HF_TOKEN)" || (echo "Error: HF_TOKEN environment variable is not set" && exit 1)
	@test -n "$(SLIDE)" || (echo "Error: SLIDE variable is required" && exit 1)
	@mkdir -p $(PWD)/output
	docker run -it --rm \
		$(DOCKER_GPU_ARGS) \
		$(DOCKER_COMMON_ARGS) \
		-v $(dir $(abspath $(SLIDE))):/mnt/slides:ro \
		-v $(PWD)/output:/mnt/output \
		--entrypoint mosaic \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		--slide-path /mnt/slides/$(notdir $(SLIDE)) \
		--output-dir /mnt/output \
		$(ARGS)

docker-run-batch: ## Run Docker container (batch mode, usage: make docker-run-batch CSV=path/to/slides.csv [ARGS="--extra-args"])
	@test -n "$(HF_TOKEN)" || (echo "Error: HF_TOKEN environment variable is not set" && exit 1)
	@test -n "$(CSV)" || (echo "Error: CSV variable is required" && exit 1)
	@mkdir -p $(PWD)/output
	docker run -it --rm \
		$(DOCKER_GPU_ARGS) \
		$(DOCKER_COMMON_ARGS) \
		-v $(dir $(abspath $(CSV))):/mnt/data:ro \
		-v $(PWD)/output:/mnt/output \
		--entrypoint mosaic \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		--slide-csv /mnt/data/$(notdir $(CSV)) \
		--output-dir /mnt/output \
		$(ARGS)

docker-shell: ## Open shell in Docker container
	@test -n "$(HF_TOKEN)" || (echo "Error: HF_TOKEN environment variable is not set" && exit 1)
	docker run -it --rm \
		$(DOCKER_GPU_ARGS) \
		$(DOCKER_COMMON_ARGS) \
		-v $(PWD)/data:/mnt/data:ro \
		-v $(PWD)/output:/mnt/output \
		--entrypoint /bin/bash \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

docker-tag: ## Tag Docker image for registry
	docker tag $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

docker-push: docker-tag ## Push Docker image to registry
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

docker-clean: ## Remove Docker image
	docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) || true

docker-prune: ## Clean up Docker build cache
	docker system prune -f
	docker builder prune -f

##@ Cleanup

clean: ## Remove build artifacts and cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

clean-outputs: ## Remove output files (masks, results CSVs)
	rm -rf output/*
	@echo "Output directory cleaned"

clean-all: clean docker-clean ## Remove all build artifacts, cache, and Docker images

##@ Model Management

download-models: ## Download required models from HuggingFace
	@echo "Downloading models from HuggingFace Hub..."
	$(PYTHON) -m mosaic.gradio_app --download-models-only

##@ Paladin Model Sync

gen-pkl-configs: ## Generate Hydra pkl configs for SLURM export (usage: make gen-pkl-configs PROJECT=paladin-redo)
	$(PYTHON) scripts/sync_paladin_models.py --project $(PROJECT) --phase gen-configs

upload-paladin-models: ## Upload pkl files to HF repo (usage: make upload-paladin-models PROJECT=paladin-redo)
	$(PYTHON) scripts/sync_paladin_models.py --project $(PROJECT) --phase upload

update-model-map: ## Update paladin_model_map.csv and push to HF (usage: make update-model-map PROJECT=paladin-redo)
	$(PYTHON) scripts/sync_paladin_models.py --project $(PROJECT) --phase update-map

sync-paladin-models: ## Run upload + update-map phases (usage: make sync-paladin-models PROJECT=paladin-redo)
	$(PYTHON) scripts/sync_paladin_models.py --project $(PROJECT)

##@ Documentation

docs-requirements: ## Show what needs to be documented
	@echo "Documentation TODO:"
	@echo "  - API documentation"
	@echo "  - Model architecture details"
	@echo "  - CLI usage examples"
	@echo "  - Docker deployment guide"

##@ CI/CD

ci-test: install-dev test-coverage format-check ## Run all CI checks (no lint to save time)
	@echo "All CI checks passed!"

ci-test-strict: install-dev test-coverage format-check lint ## Run all CI checks including pylint
	@echo "All strict CI checks passed!"

ci-docker: docker-build ## Build Docker image for CI
	@echo "Docker image built successfully"

##@ Development Utilities

shell: ## Open Python shell with project in path
	$(PYTHON)

ipython: ## Open IPython shell with project in path
	uv run ipython

notebook: ## Start Jupyter notebook server
	uv run jupyter notebook

check-deps: ## Check for outdated dependencies
	uv pip list --outdated

update-deps: ## Update dependencies (be careful!)
	uv sync --upgrade

lock: ## Update lock file
	uv lock

##@ Git Hooks

pre-commit-install: ## Install pre-commit hooks
	@echo "Setting up pre-commit hooks..."
	@echo "#!/bin/sh" > .git/hooks/pre-commit
	@echo "make format-check test-fast" >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hooks installed (format-check + test-fast)"

pre-commit-uninstall: ## Uninstall pre-commit hooks
	rm -f .git/hooks/pre-commit
	@echo "Pre-commit hooks uninstalled"

##@ Information

info: ## Display project information
	@echo "Mosaic - H&E Whole Slide Image Analysis"
	@echo "========================================"
	@echo ""
	@echo "Python version:"
	@$(PYTHON) --version
	@echo ""
	@echo "UV version:"
	@uv --version
	@echo ""
	@echo "Project structure:"
	@echo "  src/mosaic/          - Main application code"
	@echo "  tests/               - Test suite"
	@echo "  data/                - Input data directory"
	@echo "  output/              - Analysis results"
	@echo ""
	@echo "Key commands:"
	@echo "  make install-dev     - Setup development environment"
	@echo "  make test            - Run test suite"
	@echo "  make run-ui          - Launch web interface"
	@echo "  make docker-build    - Build Docker image"

version: ## Show version information
	@$(PYTHON) -c "import mosaic; print(f'Mosaic version: {mosaic.__version__}')" 2>/dev/null || echo "Version info not available"

tree: ## Show project directory tree (requires tree command)
	@tree -L 3 -I '__pycache__|*.pyc|*.egg-info|.pytest_cache|.ruff_cache|htmlcov|.venv' . || echo "tree command not found. Install with: apt-get install tree"

##@ Performance

profile: ## Profile a single slide analysis (usage: make profile SLIDE=path/to/slide.svs)
	$(PYTHON) -m cProfile -o profile.stats -m mosaic.gradio_app --slide-path $(SLIDE) --output-dir profile_output
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

benchmark: ## Run performance benchmarks
	@echo "Running benchmark suite..."
	@echo "This will process the test slide and measure performance"
	time $(PYTHON) -m mosaic.gradio_app --slide-path tests/testdata/948176.svs --output-dir benchmark_output

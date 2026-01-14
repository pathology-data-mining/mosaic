.PHONY: help install install-dev test test-coverage test-verbose lint format clean docker-build docker-run docker-push docker-clean run-ui run-cli

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

test: ## Run all tests
	$(PYTEST) tests/ -v

test-fast: ## Run tests without coverage (faster)
	$(PYTEST) tests/ -v --no-cov

test-coverage: ## Run tests with detailed coverage report
	$(PYTEST) tests/ -v --cov=src/mosaic --cov-report=term-missing --cov-report=html

test-ui: ## Run only UI tests
	$(PYTEST) tests/test_ui_components.py tests/test_ui_events.py -v

test-cli: ## Run only CLI tests
	$(PYTEST) tests/test_cli.py -v

test-verbose: ## Run tests with verbose output and show print statements
	$(PYTEST) tests/ -vv -s

test-specific: ## Run specific test (usage: make test-specific TEST=tests/test_cli.py::TestClass::test_method)
	$(PYTEST) $(TEST) -v

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

run-single: ## Run single slide analysis (usage: make run-single SLIDE=path/to/slide.svs OUTPUT=output_dir)
	$(PYTHON) -m mosaic.gradio_app --slide-path $(SLIDE) --output-dir $(OUTPUT)

run-batch: ## Run batch analysis from CSV (usage: make run-batch CSV=settings.csv OUTPUT=output_dir)
	$(PYTHON) -m mosaic.gradio_app --slide-csv $(CSV) --output-dir $(OUTPUT)

##@ Docker

docker-build: ## Build Docker image with SSH forwarding
	@echo "Building Docker image with SSH authentication..."
	@./build.sh

docker-build-no-cache: ## Build Docker image without cache
	@echo "Building Docker image with SSH authentication (no cache)..."
	@eval "$$(ssh-agent -s)" && \
	ssh-add ~/.ssh/id_ed25519 && \
	export DOCKER_BUILDKIT=1 && \
	docker build --no-cache --ssh default=$$SSH_AUTH_SOCK -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) . && \
	eval "$$(ssh-agent -k)"

docker-run: ## Run Docker container (web UI mode)
	docker run -it --rm \
		--gpus all \
		-p 7860:7860 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/output:/app/output \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

docker-run-single: ## Run Docker container (single slide mode)
	docker run -it --rm \
		--gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/output:/app/output \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		--slide-path /app/data/$(SLIDE) \
		--output-dir /app/output

docker-run-batch: ## Run Docker container (batch mode)
	docker run -it --rm \
		--gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/output:/app/output \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		--slide-csv /app/data/$(CSV) \
		--output-dir /app/output

docker-shell: ## Open shell in Docker container
	docker run -it --rm \
		--gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/output:/app/output \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		/bin/bash

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

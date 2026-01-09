# Makefile Quick Reference

## Most Common Commands

```bash
# Setup
make install-dev              # Install dev dependencies
make help                     # Show all available commands

# Testing
make test                     # Run tests with coverage
make test-fast                # Run tests quickly (no coverage)
make test-ui                  # Test UI components only
make test-cli                 # Test CLI only

# Code Quality
make format                   # Format code with black
make format-check             # Check formatting
make quality                  # Run all quality checks

# Running
make run-ui                   # Launch web interface
make run-single SLIDE=x.svs OUTPUT=out/  # Process single slide
make run-batch CSV=s.csv OUTPUT=out/     # Process batch

# Docker
make docker-build             # Build image
make docker-run               # Run web UI in container
make docker-shell             # Shell into container

# Cleanup
make clean                    # Remove cache files
make clean-all                # Remove everything
```

## Development Workflow

```bash
# 1. Initial setup
make install-dev

# 2. Make changes to code
# ... edit files ...

# 3. Format and test
make format
make test

# 4. Before committing
make quality
make test-coverage

# 5. Optional: Install pre-commit hooks
make pre-commit-install
```

## Docker Workflow

```bash
# Build and test locally
make docker-build
make docker-run

# Process slides with Docker
make docker-run-single SLIDE=my_slide.svs
make docker-run-batch CSV=settings.csv

# Push to registry
make docker-tag DOCKER_REGISTRY=myregistry.com/user
make docker-push DOCKER_REGISTRY=myregistry.com/user
```

## CI/CD

```bash
# Run all CI checks
make ci-test                  # Tests + format check (fast)
make ci-test-strict           # Tests + format check + pylint (slow)
make ci-docker                # Build Docker for CI
```

## Tips

- Use `make help` to see all available commands
- Use `make test-specific TEST=path/to/test` for debugging
- Use `make test-verbose` to see print statements
- Use `make info` to see project information
- Set environment variables to customize Docker:
  ```bash
  export DOCKER_REGISTRY=myregistry.com/user
  export DOCKER_TAG=v1.0.0
  make docker-build
  ```

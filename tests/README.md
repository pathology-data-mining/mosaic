# Mosaic Unit Tests

This directory contains comprehensive unit tests for the Mosaic package.

## Running Tests

To run all tests:

```bash
pytest tests/
```

To run tests with coverage report:

```bash
pytest tests/ --cov=src/mosaic --cov-report=term-missing
```

To run a specific test file:

```bash
pytest tests/inference/test_data.py -v
```

## Test Structure

- `tests/inference/` - Tests for inference modules
  - `test_data.py` - Tests for data structures and datasets
  - `test_aeon.py` - Tests for Aeon inference module
  - `test_paladin.py` - Tests for Paladin inference module
- `tests/test_gradio_app.py` - Tests for Gradio application utilities
- `tests/conftest.py` - Pytest configuration and fixtures

## Test Coverage

Current test coverage includes:

### Data Module (`mosaic.inference.data`)
- ✅ Cancer type mapping constants
- ✅ SiteType enum
- ✅ TileFeatureTensorDataset class
  - Dataset initialization
  - Feature padding and truncation
  - Data type conversions
  - Site type handling

### Aeon Module (`mosaic.inference.aeon`)
- ✅ Module constants
- ✅ Cancer type indices

### Paladin Module (`mosaic.inference.paladin`)
- ✅ Model map loading
- ✅ Aeon score loading
- ✅ Cancer subtype selection
- ✅ Logits to point estimates conversion
- ✅ UsageError exception

### Gradio App Module (`mosaic.gradio_app`)
- ✅ Constants (IHC subtypes, settings columns)
- ✅ Settings loading and validation
- ✅ Oncotree code name retrieval
- ✅ CSV export functionality

## Notes

- Tests use mocking for external dependencies (e.g., mussel models, network requests)
- Some tests require pytest-mock for mocking functionality
- Tests are designed to run without requiring full model downloads or GPU access

# Batch Processing Tests

This directory contains comprehensive tests for the batch processing optimization feature.

## Test Files

### Unit Tests

**`test_model_manager.py`** - Tests for model loading and caching
- ModelCache class initialization
- Model loading (Aeon, Paladin, CTransPath, Optimus)
- GPU type detection (T4 vs A100)
- Aggressive memory management vs caching
- Model cleanup functionality
- Paladin lazy-loading and caching

### Integration Tests

**`test_batch_analysis.py`** - Tests for batch processing coordinator
- End-to-end batch analysis workflow
- Batch processing with multiple slides
- Error handling (individual slide failures)
- Cleanup on errors
- Progress tracking
- Multi-slide result aggregation

### Regression Tests

**`test_regression_single_slide.py`** - Ensures single-slide mode is unchanged
- Single-slide analysis behavior
- Gradio UI single-slide path
- API backward compatibility
- Function signatures unchanged
- Return types unchanged

### Performance Benchmarks

**`benchmark_batch_performance.py`** - Performance comparison tool
- Sequential processing (old method) benchmark
- Batch processing (new method) benchmark
- Performance comparison and reporting
- Memory usage tracking

## Running Tests

### Run All Tests

```bash
# From repository root
pytest tests/test_model_manager.py tests/test_batch_analysis.py tests/test_regression_single_slide.py -v
```

### Run Specific Test Files

```bash
# Unit tests only
pytest tests/test_model_manager.py -v

# Integration tests only
pytest tests/test_batch_analysis.py -v

# Regression tests only
pytest tests/test_regression_single_slide.py -v
```

### Run Specific Test Classes or Functions

```bash
# Test specific class
pytest tests/test_model_manager.py::TestModelCache -v

# Test specific function
pytest tests/test_model_manager.py::TestModelCache::test_model_cache_initialization -v
```

### Run with Coverage

```bash
pytest tests/ --cov=mosaic.model_manager --cov=mosaic.batch_analysis --cov-report=html
```

## Running Performance Benchmarks

### Basic Benchmark (3 slides with default settings)

```bash
python tests/benchmark_batch_performance.py --slides slide1.svs slide2.svs slide3.svs
```

### Benchmark with CSV Settings

```bash
python tests/benchmark_batch_performance.py --slide-csv test_slides.csv
```

### Benchmark Batch Mode Only (Skip Sequential)

Useful for quick testing when you don't need comparison:

```bash
python tests/benchmark_batch_performance.py --slides slide1.svs slide2.svs --skip-sequential
```

### Save Benchmark Results

```bash
python tests/benchmark_batch_performance.py \
    --slide-csv test_slides.csv \
    --output benchmark_results.json
```

### Benchmark Options

- `--slides`: List of slide paths (e.g., `slide1.svs slide2.svs`)
- `--slide-csv`: Path to CSV with slide settings
- `--num-workers`: Number of CPU workers for data loading (default: 4)
- `--skip-sequential`: Skip sequential benchmark (faster)
- `--output`: Save results to JSON file

## Expected Test Results

### Unit Tests
- **test_model_manager.py**: Should pass all tests
- Tests model loading, caching, cleanup
- Tests GPU detection and adaptive memory management

### Integration Tests
- **test_batch_analysis.py**: Should pass all tests
- Tests end-to-end batch workflow
- Tests error handling and recovery

### Regression Tests
- **test_regression_single_slide.py**: Should pass all tests
- Ensures backward compatibility
- Single-slide behavior unchanged

### Performance Benchmarks

Expected performance improvements:
- **Speedup**: 1.25x - 1.45x (25-45% faster)
- **Time saved**: Depends on batch size and model loading overhead
- **Memory**: Similar peak memory to single-slide (~9-15GB on typical slides)

Example output:
```
PERFORMANCE COMPARISON
================================================================================
Number of slides: 10

Sequential processing: 450.23s
Batch processing:      300.45s

Time saved:  149.78s
Speedup:     1.50x
Improvement: 33.3% faster

Sequential peak memory: 12.45 GB
Batch peak memory:      13.12 GB
Memory difference:      +0.67 GB
================================================================================
```

## Test Coverage Goals

- **Model Manager**: >90% coverage
- **Batch Analysis**: >85% coverage
- **Regression Tests**: 100% of critical paths
- **Integration Tests**: All major workflows

## Troubleshooting

### Tests Fail Due to Missing Models

If tests fail with "model not found" errors:
```bash
# Download models first
python -m mosaic.gradio_app --help
# This will trigger model download
```

### CUDA Out of Memory Errors

If benchmarks fail with OOM:
- Reduce number of slides in benchmark
- Use `--skip-sequential` to reduce memory pressure
- Test on T4 GPU will use aggressive memory management automatically

### Import Errors

Ensure mosaic package is installed:
```bash
pip install -e .
```

## Contributing

When adding new features to batch processing:
1. Add unit tests to `test_model_manager.py` or `test_batch_analysis.py`
2. Add regression tests if modifying existing functions
3. Run benchmarks to verify performance improvements
4. Update this README with new test information

## CI/CD Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run Batch Processing Tests
  run: |
    pytest tests/test_model_manager.py tests/test_batch_analysis.py tests/test_regression_single_slide.py -v --cov
```

For performance regression detection:
```yaml
- name: Performance Benchmark
  run: |
    python tests/benchmark_batch_performance.py --slide-csv ci_test_slides.csv --output benchmark.json
    python scripts/check_performance_regression.py benchmark.json
```

#!/bin/bash
# Test runner script for batch processing tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Batch Processing Test Suite"
echo "========================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Install with: pip install pytest pytest-cov"
    exit 1
fi

# Default: run all tests
TEST_SUITE="${1:-all}"

case "$TEST_SUITE" in
    "unit")
        echo -e "${YELLOW}Running Unit Tests...${NC}"
        pytest tests/test_model_manager.py -v
        ;;
    "integration")
        echo -e "${YELLOW}Running Integration Tests...${NC}"
        pytest tests/test_batch_analysis.py -v
        ;;
    "regression")
        echo -e "${YELLOW}Running Regression Tests...${NC}"
        pytest tests/test_regression_single_slide.py -v
        ;;
    "all")
        echo -e "${YELLOW}Running All Tests...${NC}"
        pytest tests/test_model_manager.py \
               tests/test_batch_analysis.py \
               tests/test_regression_single_slide.py \
               -v
        ;;
    "coverage")
        echo -e "${YELLOW}Running Tests with Coverage...${NC}"
        pytest tests/test_model_manager.py \
               tests/test_batch_analysis.py \
               tests/test_regression_single_slide.py \
               --cov=mosaic.model_manager \
               --cov=mosaic.batch_analysis \
               --cov=mosaic.analysis \
               --cov-report=term-missing \
               --cov-report=html \
               -v
        echo ""
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    "quick")
        echo -e "${YELLOW}Running Quick Test (no mocks needed)...${NC}"
        pytest tests/test_model_manager.py::TestModelCache -v
        ;;
    *)
        echo -e "${RED}Unknown test suite: $TEST_SUITE${NC}"
        echo ""
        echo "Usage: $0 [unit|integration|regression|all|coverage|quick]"
        echo ""
        echo "  unit         - Run unit tests (test_model_manager.py)"
        echo "  integration  - Run integration tests (test_batch_analysis.py)"
        echo "  regression   - Run regression tests (test_regression_single_slide.py)"
        echo "  all          - Run all tests (default)"
        echo "  coverage     - Run all tests with coverage report"
        echo "  quick        - Run quick sanity test"
        exit 1
        ;;
esac

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

exit $EXIT_CODE

"""Pytest configuration for end-to-end browser tests.

All tests in this directory are skipped unless --e2e is passed on the command line.
This prevents them from running in CI.

Usage:
    pytest tests/e2e/ --e2e -v
    pytest tests/e2e/ --e2e -v --headed   # show browser window
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run end-to-end browser tests (requires GPU, models, and test slide)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--e2e"):
        skip = pytest.mark.skip(reason="Pass --e2e to run end-to-end browser tests")
        for item in items:
            if item.fspath.dirpath().basename == "e2e":
                item.add_marker(skip)

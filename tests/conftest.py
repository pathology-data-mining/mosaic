"""Pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before any imports
# This is necessary to allow tests to run without full environment setup
mock_modules = [
    'mussel.models',
    'mussel.utils',
    'mussel.utils.segment',
    'mussel.cli.tessellate',
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

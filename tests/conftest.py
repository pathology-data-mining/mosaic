"""Pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock

# Create mock for gradio with Error class
class GradioMock(MagicMock):
    """Mock for gradio that supports Error and Warning classes."""
    Error = Exception
    Warning = lambda msg: None
    Request = MagicMock
    Progress = MagicMock
    
# Mock heavy dependencies before any imports
# This is necessary to allow tests to run without full environment setup
sys.modules['mussel'] = MagicMock()
sys.modules['mussel.models'] = MagicMock()
sys.modules['mussel.utils'] = MagicMock()
sys.modules['mussel.utils.segment'] = MagicMock()
sys.modules['mussel.cli'] = MagicMock()
sys.modules['mussel.cli.tessellate'] = MagicMock()
sys.modules['gradio'] = GradioMock()
sys.modules['huggingface_hub'] = MagicMock()
sys.modules['loguru'] = MagicMock()

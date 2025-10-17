"""Pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock

# Create mock for torch with sub-modules
class TorchMock(MagicMock):
    """Mock for torch that supports nested imports."""
    utils = MagicMock()
    nn = MagicMock()
    cuda = MagicMock()

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
sys.modules['torch'] = TorchMock()
sys.modules['torch.utils'] = TorchMock.utils
sys.modules['torch.utils.data'] = TorchMock.utils.data
sys.modules['torch.nn'] = TorchMock.nn
sys.modules['torch.cuda'] = TorchMock.cuda
sys.modules['huggingface_hub'] = MagicMock()
sys.modules['loguru'] = MagicMock()

"""Pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock


# Create mock component that tracks visible attribute
class MockComponent(MagicMock):
    """Mock Gradio component that tracks visible attribute."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visible = kwargs.get("visible", True)


# Create mock for gradio with Error class
class GradioMock(MagicMock):
    """Mock for gradio that supports Error and Warning classes."""

    Error = Exception

    @staticmethod
    def Warning(msg):
        """Mock Warning that accepts a message."""
        return None

    @staticmethod
    def DownloadButton(*args, **kwargs):
        """Mock DownloadButton that tracks visible attribute."""
        return MockComponent(*args, **kwargs)

    @staticmethod
    def DataFrame(*args, **kwargs):
        """Mock DataFrame that tracks visible attribute."""
        return MockComponent(*args, **kwargs)

    @staticmethod
    def Label(*args, **kwargs):
        """Mock Label that tracks visible attribute."""
        return MockComponent(*args, **kwargs)

    @staticmethod
    def Textbox(*args, **kwargs):
        """Mock Textbox that tracks visible attribute."""
        return MockComponent(*args, **kwargs)

    @staticmethod
    def Button(*args, **kwargs):
        """Mock Button that tracks interactive attribute."""
        button = MockComponent(*args, **kwargs)
        button.interactive = kwargs.get("interactive", True)
        return button

    Request = MagicMock
    Progress = MagicMock


# Mock heavy dependencies before any imports
# This is necessary to allow tests to run without full environment setup
sys.modules["mussel"] = MagicMock()
sys.modules["mussel.models"] = MagicMock()
sys.modules["mussel.utils"] = MagicMock()
sys.modules["mussel.utils.segment"] = MagicMock()
sys.modules["mussel.cli"] = MagicMock()
sys.modules["mussel.cli.tessellate"] = MagicMock()
sys.modules["gradio"] = GradioMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["loguru"] = MagicMock()

# Import fixtures from test_fixtures.py to make them available to all tests
pytest_plugins = ["tests.test_fixtures"]

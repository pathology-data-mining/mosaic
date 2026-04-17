"""Pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock


# Create mock component that tracks visible attribute
class MockComponent(MagicMock):
    """Mock Gradio component that tracks visible attribute."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visible = kwargs.get("visible", True)
        # Capture datatype explicitly so tests can assert it is a list (not the
        # Gradio default string "str").  Without an explicit list Gradio 6 creates
        # a circular Svelte reactive dependency that freezes the browser for ~420 s.
        self.datatype = kwargs.get("datatype", "str")


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
    def Dataframe(*args, **kwargs):
        """Mock Dataframe (lowercase-f alias) that tracks visible attribute."""
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

    @staticmethod
    def LoginButton(*args, **kwargs):
        """Mock LoginButton that tracks visible attribute."""
        return MockComponent(*args, **kwargs)

    @staticmethod
    def update(**kwargs):
        """Mock update() that returns a dict like real Gradio."""
        return kwargs

    Request = MagicMock
    Progress = MagicMock


# Mock for gradio.helpers with TrackedIterable
class GradioHelpersMock(MagicMock):
    """Mock for gradio.helpers module."""

    TrackedIterable = type("TrackedIterable", (), {"__init__": lambda self, *a: None})


# Mock heavy dependencies before any imports
# This is necessary to allow tests to run without full environment setup
sys.modules["mussel"] = MagicMock()
sys.modules["mussel.models"] = MagicMock()
sys.modules["mussel.utils"] = MagicMock()
sys.modules["mussel.utils.segment"] = MagicMock()
sys.modules["mussel.utils.feature_extract"] = MagicMock()
sys.modules["mussel.cli"] = MagicMock()
sys.modules["mussel.cli.tessellate"] = MagicMock()
sys.modules["gradio"] = GradioMock()
sys.modules["gradio.helpers"] = GradioHelpersMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["loguru"] = MagicMock()

# Import fixtures from test_fixtures.py to make them available to all tests
pytest_plugins = ["tests.test_fixtures"]

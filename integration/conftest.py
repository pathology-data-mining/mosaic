"""Conftest for Gradio integration tests.

Unlike tests/conftest.py, this file does NOT mock gradio or loguru.
It mocks only the heavy ML/model dependencies so the real Gradio server
can start and accept HTTP connections.
"""

import socket
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy ML deps (not installed in CI without GPU, or require large downloads).
# huggingface_hub is intentionally NOT mocked — gradio_client depends on it.
sys.modules["mussel"] = MagicMock()
sys.modules["mussel.models"] = MagicMock()
sys.modules["mussel.utils"] = MagicMock()
sys.modules["mussel.utils.segment"] = MagicMock()
sys.modules["mussel.cli"] = MagicMock()
sys.modules["mussel.cli.tessellate"] = MagicMock()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    """Poll until the TCP port accepts connections or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


@pytest.fixture(scope="session")
def gradio_server():
    """Start the real Gradio app once for the whole integration test session.

    Yields a dict with ``url`` (str) and ``port`` (int).
    """
    port = _find_free_port()
    host = "127.0.0.1"

    mock_telemetry = MagicMock()

    with (
        patch(
            "mosaic.telemetry.TelemetryTracker.get_instance",
            return_value=mock_telemetry,
        ),
        patch("mosaic.ui.pipeline.start_background_model_download"),
    ):
        from mosaic.ui.app import launch_gradio
        from mosaic.ui.pipeline import set_cancer_subtype_maps

        # Populate cancer subtype maps so dropdown choices are non-empty
        sample_map = {
            "LUAD": "Lung Adenocarcinoma",
            "BLCA": "Bladder Urothelial Carcinoma",
            "Unknown": "Unknown",
        }
        reversed_map = {v: k for k, v in sample_map.items()}
        cancer_subtypes = list(sample_map.keys())
        set_cancer_subtype_maps(sample_map, reversed_map, cancer_subtypes)

        thread = threading.Thread(
            target=launch_gradio,
            kwargs={"server_name": host, "server_port": port, "share": False},
            daemon=True,
        )
        thread.start()

        assert _wait_for_port(
            host, port, timeout=20
        ), f"Gradio server did not come up on port {port} within 20 s"

    yield {"url": f"http://{host}:{port}", "port": port, "host": host}
    # Daemon thread exits automatically when the test process finishes.

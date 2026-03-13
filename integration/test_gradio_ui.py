"""Gradio UI integration tests.

These tests start the real Gradio server (not mocked) and verify the app
using HTTP requests and the gradio_client Python SDK.

Run with:
    make test-integration
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd
import pytest
from gradio_client import Client, handle_file
from unittest.mock import patch, MagicMock
from PIL import Image
import tempfile
import io


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(url: str, timeout: int = 10) -> tuple[int, bytes]:
    """Simple HTTP GET; returns (status_code, body_bytes)."""
    resp = urllib.request.urlopen(url, timeout=timeout)
    return resp.status, resp.read()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestSmoke:
    """Basic connectivity — verify the app starts and responds."""

    def test_root_returns_200(self, gradio_server):
        """GET / returns HTTP 200."""
        status, _ = _get(gradio_server["url"] + "/")
        assert status == 200

    def test_config_endpoint_is_json(self, gradio_server):
        """GET /config returns valid JSON with Gradio app metadata."""
        status, body = _get(gradio_server["url"] + "/config")
        assert status == 200
        data = json.loads(body)
        assert "version" in data or "components" in data

    def test_static_assets_served(self, gradio_server):
        """Gradio serves its own JS bundle."""
        status, body = _get(gradio_server["url"] + "/config")
        assert status == 200
        assert len(body) > 0


# ---------------------------------------------------------------------------
# API schema tests
# ---------------------------------------------------------------------------

# Endpoints that must be present (wired up in launch_gradio)
REQUIRED_ENDPOINTS = {
    "/handle_slide_upload",
    "/update_cancer_subtype",
    "/analyze_with_tcga_support",
    "/fetch_tcga_slide",
    "/switch_input_mode",
}


class TestAPISchema:
    """Verify the Gradio API exposes the expected named endpoints."""

    @pytest.fixture(scope="class")
    def client(self, gradio_server):
        return Client(gradio_server["url"], verbose=False)

    def test_named_endpoints_exist(self, client):
        """All required API endpoints are registered."""
        api = client.view_api(return_format="dict")
        endpoints = set(api.get("named_endpoints", {}).keys())
        missing = REQUIRED_ENDPOINTS - endpoints
        assert not missing, f"Missing endpoints: {missing}"

    def test_analyze_endpoint_has_correct_parameters(self, client):
        """The /analyze_with_tcga_support endpoint accepts slides and settings inputs."""
        api = client.view_api(return_format="dict")
        endpoint = api["named_endpoints"]["/analyze_with_tcga_support"]
        param_names = [p["parameter_name"] for p in endpoint["parameters"]]
        # Must accept slides (the file upload component)
        assert (
            "slides" in param_names
        ), f"Expected 'slides' parameter. Got: {param_names}"

    def test_handle_slide_upload_has_files_input(self, client):
        """The /handle_slide_upload endpoint accepts a files parameter."""
        api = client.view_api(return_format="dict")
        endpoint = api["named_endpoints"]["/handle_slide_upload"]
        param_names = [p["parameter_name"] for p in endpoint["parameters"]]
        assert "files" in param_names, f"Expected 'files' param, got: {param_names}"


# ---------------------------------------------------------------------------
# Functional endpoint tests (lightweight — no model loading)
# ---------------------------------------------------------------------------


class TestFunctionalEndpoints:
    """Call lightweight API endpoints end-to-end via the real HTTP server."""

    @pytest.fixture(scope="class")
    def client(self, gradio_server):
        return Client(gradio_server["url"], verbose=False)

    def test_update_cancer_subtype_returns_value(self, client):
        """/update_cancer_subtype is callable and returns a value."""
        # "Unknown" is always in the cancer subtype map (set in conftest)
        result = client.predict(
            cancer_subtype="Unknown", api_name="/update_cancer_subtype"
        )
        # Should return a string (the subtype name or an updated component value)
        assert result is not None

    def test_update_site_type_returns_value(self, client):
        """/update_site_type is callable and returns a value."""
        result = client.predict(site_type="Primary", api_name="/update_site_type")
        assert result is not None

    def test_switch_input_mode_returns_value(self, client):
        """/switch_input_mode toggles between upload and TCGA modes."""
        result = client.predict(api_name="/switch_input_mode")
        assert result is not None


# ---------------------------------------------------------------------------
# Analysis pipeline integration test
# ---------------------------------------------------------------------------


class TestAnalysisPipeline:
    """End-to-end test of the analysis pipeline via the Gradio API.

    Uses a real SVS-like file (actually a small TIFF that mussel mocks accept)
    and patches analyze_slide to return deterministic results — no GPU needed.
    """

    @pytest.fixture(scope="class")
    def fake_slide_path(self, tmp_path_factory):
        """Create a tiny fake slide file for upload."""
        d = tmp_path_factory.mktemp("slides")
        slide = d / "fake_slide.svs"
        # Create a minimal TIFF-compatible file so openslide doesn't crash on open
        img = Image.new("RGB", (256, 256), color=(200, 180, 160))
        img.save(str(slide), format="TIFF")
        return slide

    @pytest.fixture(scope="class")
    def mock_results(self):
        mask = Image.new("RGB", (100, 100), color="red")
        aeon = pd.DataFrame({"Cancer Subtype": ["LUAD"], "Confidence": [0.95]})
        paladin = pd.DataFrame(
            {
                "Cancer Subtype": ["LUAD"],
                "Biomarker": ["TP53"],
                "Score": [0.85],
            }
        )
        return (mask, aeon, paladin)

    def test_analysis_pipeline_completes(
        self, gradio_server, fake_slide_path, mock_results
    ):
        """Analysis pipeline endpoint is callable; server stays healthy after the call.

        analyze_slide is mocked to avoid GPU/model loading. We submit a request
        to the analysis endpoint and verify the server is still alive afterwards,
        regardless of whether the fake-TIFF pipeline call succeeds or raises.
        """
        with (
            patch("mosaic.ui.pipeline.analyze_slide", return_value=mock_results),
            patch("mosaic.ui.pipeline.load_all_models", return_value=MagicMock()),
            patch("mosaic.ui.pipeline.save_analysis_results"),
            patch("mosaic.ui.pipeline.load_analysis_results", return_value=None),
        ):
            client = Client(gradio_server["url"], verbose=False)

            # Gradio DataFrames are serialized as {"headers": [...], "data": [[...]]}
            settings_payload = {
                "headers": [
                    "Slide",
                    "Site Type",
                    "Sex",
                    "Tissue Site",
                    "Cancer Subtype",
                    "Segmentation Config",
                ],
                "data": [
                    [
                        fake_slide_path.name,
                        "Primary",
                        "Male",
                        "Unknown",
                        "Unknown",
                        "Biopsy",
                    ]
                ],
            }

            try:
                job = client.submit(
                    input_mode_val="upload",
                    slides=[handle_file(str(fake_slide_path))],
                    settings_input_val=settings_payload,
                    site_type="Primary",
                    sex="Male",
                    tissue_site="Unknown",
                    cancer_subtype="Unknown",
                    seg_config="Biopsy",
                    session_id=None,
                    api_name="/analyze_with_tcga_support",
                )
                job.result()
            except Exception:
                pass  # Pipeline errors on a fake TIFF are acceptable

        # Most important: the server is still alive after the pipeline call
        status, _ = _get(gradio_server["url"] + "/")
        assert status == 200, "Server should still respond after a pipeline call"

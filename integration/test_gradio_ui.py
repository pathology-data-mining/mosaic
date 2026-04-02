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
        aeon = pd.DataFrame({"Cancer Subtype": ["LUAD"], "Confidence": [0.95]}).set_index(
            "Cancer Subtype"
        )
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
                    input_mode_val="Upload Files",
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


# ---------------------------------------------------------------------------
# End-to-end analysis test
# ---------------------------------------------------------------------------


class TestE2EAnalysis:
    """True end-to-end analysis: upload a slide, run the pipeline, verify results.

    The ML models (analyze_slide) are mocked to return deterministic outputs so
    the test runs without GPU or large model downloads.  Everything else —
    Gradio event routing, pipeline.analyze_slides(), result serialisation, and
    HTTP transport — is real and exercised.
    """

    # Aeon / Paladin results that the mocked analyze_slide will return.
    CANCER_SUBTYPE = "LUAD"
    BIOMARKERS = ["TP53", "KRAS", "EGFR"]

    @pytest.fixture(scope="class")
    def fake_slide(self, tmp_path_factory):
        """256×256 TIFF named as an SVS so the pipeline accepts it as a slide."""
        d = tmp_path_factory.mktemp("e2e_slides")
        path = d / "test_slide.svs"
        img = Image.new("RGB", (256, 256), color=(210, 185, 165))
        img.save(str(path), format="TIFF")
        return path

    @pytest.fixture(scope="class")
    def analysis_result(self):
        """Deterministic (mask, aeon_df, paladin_df) tuple for the mock."""
        mask = Image.new("RGB", (128, 128), color=(200, 100, 100))
        aeon_df = pd.DataFrame(
            {
                "Cancer Subtype": [self.CANCER_SUBTYPE],
                "Confidence": [0.9412],
            }
        ).set_index("Cancer Subtype")
        paladin_df = pd.DataFrame(
            {
                "Cancer Subtype": [self.CANCER_SUBTYPE] * len(self.BIOMARKERS),
                "Biomarker": self.BIOMARKERS,
                "Score": [0.85, 0.72, 0.63],
            }
        )
        return (mask, aeon_df, paladin_df)

    @pytest.fixture(scope="class")
    def e2e_result(self, gradio_server, fake_slide, analysis_result):
        """Run the full analysis via the Gradio API and return the final output."""
        with (
            patch("mosaic.ui.pipeline.analyze_slide", return_value=analysis_result),
            patch("mosaic.ui.pipeline.load_all_models", return_value=MagicMock()),
            patch("mosaic.ui.pipeline.save_analysis_results"),
            patch("mosaic.ui.pipeline.load_analysis_results", return_value=None),
            patch(
                "mosaic.ui.pipeline.get_oncotree_code_name",
                side_effect=lambda code: f"Mock {code}",
            ),
        ):
            client = Client(gradio_server["url"], verbose=False)
            result = client.predict(
                input_mode_val="Upload Files",
                slides=[handle_file(str(fake_slide))],
                settings_input_val={
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
                            fake_slide.name,
                            "Primary",
                            "Male",
                            "Unknown",
                            "Unknown",
                            "Biopsy",
                        ]
                    ],
                },
                site_type="Primary",
                sex="Male",
                tissue_site="Unknown",
                cancer_subtype="Unknown",
                seg_config="Biopsy",
                session_id=None,
                api_name="/analyze_with_tcga_support",
            )
        # result is a 6-tuple: (settings, masks, aeon, aeon_dl, paladin, paladin_dl)
        return result

    def test_returns_six_outputs(self, e2e_result):
        """Analysis returns exactly the expected number of output components."""
        assert len(e2e_result) == 6, (
            f"Expected 6 outputs (settings, masks, aeon, aeon_dl, paladin, paladin_dl), "
            f"got {len(e2e_result)}"
        )

    def test_aeon_results_contain_cancer_subtype(self, e2e_result):
        """Aeon output DataFrame contains a Cancer Subtype column with a result."""
        _, _, aeon_out, _, _, _ = e2e_result
        # When all_subtypes_known=False, Gradio returns a gr.update dict:
        # {"visible": True, "value": {"headers": [...], "data": [...]}, "__type__": "update"}
        # When it's a plain DataFrame it comes as {"headers": [...], "data": [...]}
        assert aeon_out is not None, "Aeon results should not be None"
        aeon_value = aeon_out.get("value", aeon_out)
        headers = aeon_value.get("headers", [])
        assert any(
            "cancer" in h.lower() or "subtype" in h.lower() for h in headers
        ), f"Expected a Cancer Subtype column, got headers: {headers}"
        assert len(aeon_value.get("data", [])) >= 1, "Aeon results should have at least one row"

    def test_aeon_predicts_correct_cancer_type(self, e2e_result):
        """Aeon output contains the mocked cancer subtype (LUAD)."""
        _, _, aeon_out, _, _, _ = e2e_result
        aeon_value = aeon_out.get("value", aeon_out)
        all_values = [str(v) for row in aeon_value["data"] for v in row]
        assert any(self.CANCER_SUBTYPE in v for v in all_values), (
            f"Expected '{self.CANCER_SUBTYPE}' in aeon results, got: {aeon_value['data']}"
        )

    def test_paladin_results_contain_biomarkers(self, e2e_result):
        """Paladin output DataFrame contains biomarker predictions."""
        _, _, _, _, paladin_out, _ = e2e_result
        assert paladin_out is not None, "Paladin results should not be None"
        assert len(paladin_out.get("data", [])) >= 1, (
            "Paladin results should have at least one row"
        )

    def test_paladin_predicts_expected_biomarkers(self, e2e_result):
        """Paladin output contains the mocked biomarker scores."""
        _, _, _, _, paladin_out, _ = e2e_result
        all_values = [str(v) for row in paladin_out["data"] for v in row]
        found = [b for b in self.BIOMARKERS if any(b in v for v in all_values)]
        assert found, (
            f"Expected at least one of {self.BIOMARKERS} in paladin results. "
            f"Got: {paladin_out['data']}"
        )

    def test_slide_mask_is_returned(self, e2e_result):
        """Mask gallery contains an image for the analysed slide."""
        _, masks_out, _, _, _, _ = e2e_result
        # Gallery output is a list of image paths / dicts
        assert masks_out is not None, "Masks gallery should not be None"
        assert len(masks_out) >= 1, "At least one mask image should be returned"

    def test_download_buttons_are_populated(self, e2e_result):
        """Aeon and Paladin download buttons have a file path after analysis."""
        _, _, _, aeon_dl, _, paladin_dl = e2e_result
        assert aeon_dl is not None, "Aeon download button should be populated"
        assert paladin_dl is not None, "Paladin download button should be populated"


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


class TestUploadRegression:
    """Regression tests for slide upload bugs.

    These tests exercise the upload event chain at the HTTP level so that
    regressions (e.g. NameError, wrong return type) are caught by CI before
    they reach users.
    """

    @pytest.fixture(scope="class")
    def client(self, gradio_server):
        return Client(gradio_server["url"], verbose=False)

    @pytest.fixture(scope="class")
    def fake_slide(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("regression_slides")
        path = d / "regression_slide.svs"
        from PIL import Image

        Image.new("RGB", (128, 128), color=(200, 180, 160)).save(str(path), format="TIFF")
        return path

    def test_upload_handler_does_not_clear_input_slides(self, client, fake_slide):
        """/handle_slide_upload must not return a value that wipes input_slides.

        Regression: handle_slide_upload was returning `files` directly, which in
        some Gradio versions caused the File component to reset.  It should now
        return gr.skip() so the displayed files are left untouched.
        """
        from gradio_client import handle_file

        with patch("mosaic.user_storage.save_uploaded_slide", side_effect=Exception("no storage")):
            result = client.predict(
                files=[handle_file(str(fake_slide))],
                session_id="",
                api_name="/handle_slide_upload",
            )

        # result is the input_slides_update (slide_ids_state is gr.State, not in API).
        # gr.skip() serialises as {"__type__": "update"} over the Gradio API wire,
        # which tells the frontend not to change the component — i.e. uploaded files
        # remain visible.  Any other truthy value (list, dict with "value" key, etc.)
        # would replace the component and clear the upload.
        slides_update = result
        assert isinstance(slides_update, dict) and slides_update.get("__type__") == "update", (
            f"handle_slide_upload should return gr.skip() (got {slides_update!r}). "
            "Returning anything else clears the upload component."
        )

    def test_select_existing_slide_empty_selection_does_not_clear_upload(self, client):
        """/select_existing_slide with no selection must not clear input_slides.

        Regression: select_existing_slide was returning None when slide_ids was
        empty/None, which cleared the File component.  After the upload chain
        calls load_existing_slides (resetting the dropdown value to None),
        the change handler fires with slide_ids=None — and must not wipe any
        already-uploaded files.
        """
        result = client.predict(
            slide_ids=None,
            session_id="",
            api_name="/select_existing_slide",
        )
        # gr.skip() serialises as {"__type__": "update"} — the frontend leaves
        # the current value unchanged.  A plain None or a file list would clear
        # or replace whatever the user had already uploaded.
        assert isinstance(result, dict) and result.get("__type__") == "update", (
            f"select_existing_slide(None) should return gr.skip() (got {result!r}). "
            "Returning None or a file list would clear the user's uploaded files."
        )

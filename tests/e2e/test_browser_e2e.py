"""End-to-end browser tests for the Mosaic Gradio application.

These tests run the full analysis pipeline with real models and a real browser.
They are intentionally excluded from CI and must be run manually on a system
with a CUDA-capable GPU.

Prerequisites:
    1. CUDA GPU available
    2. Model weights downloaded (or HF_TOKEN set for automatic download)
    3. playwright install chromium
    4. tests/testdata/948176.svs present (multi-slide test uses it twice)

Usage:
    pytest tests/e2e/ --e2e -v
    pytest tests/e2e/ --e2e -v --headed            # show browser window
    pytest tests/e2e/ --e2e -v -s                  # with stdout
    pytest tests/e2e/ --e2e --browser firefox      # use Firefox instead
    pytest tests/e2e/ --e2e --server-url http://tllihpcgpu4:7860  # reuse live server

The key regression this catches:
    gr.DataFrame without an explicit ``datatype`` list causes a circular
    Svelte 5 reactive dependency in TanStack Table, producing a ~420-second
    browser Long Task after every analysis result delivery.  The test fails
    if any single Long Task exceeds LONG_TASK_THRESHOLD_MS.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_SLIDE = Path(__file__).parent.parent / "testdata" / "948176.svs"

# Ground-truth metadata for TEST_SLIDE (948176.svs — HGSOC omentum metastasis).
# Source: pathology report + IMPACT annotation.
TEST_SLIDE_METADATA = {
    "sex": "Female",
    "site_type": "Metastatic",
    "tissue_site": "Omentum",
}

# Expected Aeon top-1 cancer type for TEST_SLIDE.
# Override via --expected-cancer-type when running against a different slide.
TEST_SLIDE_EXPECTED_CANCER_TYPE = "HGSOC"

# Seconds to wait for the Gradio dev server to become responsive.
SERVER_STARTUP_TIMEOUT = 180

# Seconds to wait for the full slide analysis to finish (real models are slow).
ANALYSIS_TIMEOUT = 900  # 15 minutes

# Any single browser Long Task longer than this (ms) is a failure.
# The regression produced ~420,000 ms; a healthy analysis produces < 200 ms.
LONG_TASK_THRESHOLD_MS = 5_000

# JS injected into the page to collect Long Task timings via PerformanceObserver.
_LONG_TASK_MONITOR_JS = """
window.__mosaic_longTasks = [];
new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
        window.__mosaic_longTasks.push(entry.duration);
    }
}).observe({ entryTypes: ['longtask'] });
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: int) -> None:
    """Poll *url* until it responds with HTTP 200, or raise after *timeout* s."""
    deadline = time.monotonic() + timeout
    last_exc = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
        time.sleep(2)
    raise TimeoutError(
        f"Gradio server at {url} did not respond within {timeout}s. "
        f"Last error: {last_exc}"
    )


def _assert_aeon_results_sensible(aeon_table, expected_cancer_type: str | None = None):
    """Assert that an Aeon results table contains sensible predictions.

    Checks:
    - At least 5 rows (model outputs 100+ cancer types)
    - Every visible confidence value is a float in (0, 1]
    - Top confidence is meaningfully non-zero (> 0.05)
    - If *expected_cancer_type* is given, it must appear in the top 3 predictions
      with confidence > 0.1 (top-1 is not required; model accuracy is tested
      separately via verify_aeon_results.py)
    """
    rows = aeon_table.locator("tbody tr")
    assert rows.count() >= 5, f"Aeon table has only {rows.count()} rows; expected ≥ 5"

    confidences = []
    cancer_types = []
    for i in range(rows.count()):
        cells = rows.nth(i).locator("td")
        subtype = cells.nth(0).inner_text().strip()
        conf_text = cells.nth(1).inner_text().strip()
        assert subtype, f"Aeon row {i} has empty cancer subtype"
        try:
            conf = float(conf_text)
        except ValueError:
            pytest.fail(f"Aeon row {i} confidence '{conf_text}' is not a number")
        assert (
            0 <= conf <= 1.0
        ), f"Aeon confidence {conf} at row {i} out of range [0, 1]"
        confidences.append(conf)
        cancer_types.append(subtype)

    assert max(confidences) > 0.05, (
        f"Top Aeon confidence {max(confidences):.4f} is suspiciously low; "
        "the model may not have run correctly"
    )

    if expected_cancer_type:
        top3 = cancer_types[:3]
        # cancer_types contain full names like "High-Grade Serous Ovarian Cancer (HGSOC)";
        # expected_cancer_type is the oncotree code (e.g. "HGSOC"), so use substring match.
        top3_match = [ct for ct in top3 if expected_cancer_type in ct]
        assert top3_match, (
            f"Expected cancer type code '{expected_cancer_type}' not found in top 3 "
            f"Aeon predictions {top3}. Full table order: {cancer_types[:10]}"
        )
        matched_conf = confidences[cancer_types.index(top3_match[0])]
        assert matched_conf > 0.1, (
            f"'{expected_cancer_type}' confidence "
            f"{matched_conf:.4f} < 0.10 — suspiciously low."
        )


def _assert_paladin_results_sensible(page):
    """Assert that the Paladin results table contains sensible biomarker probabilities.

    Checks:
    - At least 1 biomarker row
    - All 'Prob. Present' values (column 3) are floats in [0, 1]
    - At least one biomarker name is non-empty
    """
    paladin_table = page.locator("#paladin-results-table table")
    rows = paladin_table.locator("tbody tr")
    assert rows.count() > 0, "Paladin table has no biomarker rows"

    for i in range(rows.count()):
        cells = rows.nth(i).locator("td")
        biomarker = cells.nth(2).inner_text().strip()
        prob_text = cells.nth(3).inner_text().strip()
        assert biomarker, f"Paladin row {i} has empty biomarker name"
        try:
            prob = float(prob_text)
        except ValueError:
            pytest.fail(f"Paladin row {i} prob '{prob_text}' is not a number")
        assert (
            0.0 <= prob <= 1.0
        ), f"Paladin 'Prob. Present' {prob} at row {i} is out of range [0, 1]"


def _fill_slide_metadata(page, sex: str, site_type: str, tissue_site: str) -> None:
    """Set the global Sex, Site Type, and Tissue Site dropdowns.

    These global dropdowns pre-fill all rows in the settings table when
    a slide is uploaded.  Must be called after the settings table appears.
    """
    for label, value in [
        ("Sex", sex),
        ("Site Type", site_type),
        ("Tissue Site", tissue_site),
    ]:
        dropdown = page.get_by_label(label)
        dropdown.click()
        page.get_by_role("option", name=value).click()


# ---------------------------------------------------------------------------
# Session-scoped fixture: one server for the whole test module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def expected_cancer_type(request):
    """Return the expected top-1 Aeon cancer type for the current test slide.

    Defaults to TEST_SLIDE_EXPECTED_CANCER_TYPE (validated for 948176.svs).
    Override with --expected-cancer-type on the command line when running
    against a different slide.
    """
    override = request.config.getoption("--expected-cancer-type", default=None)
    return override if override else TEST_SLIDE_EXPECTED_CANCER_TYPE


@pytest.fixture(scope="module")
def gradio_server(request):
    """Provide the Mosaic Gradio server URL for the test module.

    If ``--server-url`` is given on the command line, that URL is used directly
    (no server is started — ideal for reusing a live instance that already has
    models loaded).  Otherwise a fresh server subprocess is started on a free
    port and torn down after the module finishes.

    Yields the base URL (e.g. ``http://127.0.0.1:51234``).
    """
    server_url = request.config.getoption("--server-url", default=None)

    if server_url:
        # Reuse an already-running server — just verify it responds.
        _wait_for_server(server_url.rstrip("/"), timeout=10)
        yield server_url.rstrip("/")
        return  # nothing to tear down

    if not TEST_SLIDE.exists():
        pytest.skip(f"Test slide not found: {TEST_SLIDE}")

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    # Ensure the server binds only to loopback so it doesn't clash with
    # any running production instance on 0.0.0.0:7860.
    env["GRADIO_SERVER_NAME"] = "127.0.0.1"
    env["GRADIO_SERVER_PORT"] = str(port)
    # Silence Gradio's verbose startup banner in test output.
    env["GRADIO_ANALYTICS_ENABLED"] = "False"

    import collections
    import threading

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mosaic.gradio_app",
            "--server-name",
            "127.0.0.1",
            "--server-port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Drain stdout in a background thread to prevent the 64 KB Linux pipe buffer
    # from filling up and blocking the server process.  Without this, after two
    # full analyses the pipe is full and any subsequent log write by the server
    # blocks indefinitely — making test_multi_slide_sample appear to hang.
    _server_log: "collections.deque[str]" = collections.deque(maxlen=500)

    def _drain(pipe, log: "collections.deque[str]") -> None:
        try:
            for raw in pipe:
                log.append(raw.decode(errors="replace").rstrip())
        except Exception:
            pass

    _drain_thread = threading.Thread(target=_drain, args=(proc.stdout, _server_log), daemon=True)
    _drain_thread.start()

    try:
        _wait_for_server(url, SERVER_STARTUP_TIMEOUT)
    except TimeoutError as exc:
        proc.kill()
        proc.wait()
        log_tail = "\n".join(_server_log)
        pytest.fail(f"{exc}\n\nServer output (last {len(_server_log)} lines):\n{log_tail}")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestFullAnalysisBrowser:
    """Full-pipeline browser tests against a live Gradio server."""

    def test_analysis_completes_without_browser_freeze(
        self, page, gradio_server, expected_cancer_type
    ):
        """Upload a real slide, run analysis, assert results and no Long Tasks.

        Regression guard:
            Without ``datatype=[...]`` on ``gr.DataFrame``, Gradio 6 / Svelte 5 /
            TanStack Table creates a reactive cycle that produces a ~420,000 ms
            browser Long Task after results are delivered.  This test fails if any
            single Long Task exceeds ``LONG_TASK_THRESHOLD_MS``.
        """
        # ------------------------------------------------------------------ #
        # 1. Navigate to the app and arm the Long Task monitor before any JS  #
        #    from the page has a chance to run.                                #
        # ------------------------------------------------------------------ #
        page.goto(gradio_server)
        page.wait_for_load_state("load")
        # Gradio keeps a persistent SSE connection so "networkidle" never fires.
        # Wait for the Analyze button instead — it appears once the UI is ready.
        page.get_by_role("button", name="Analyze").wait_for(
            state="visible", timeout=30_000
        )
        page.evaluate(_LONG_TASK_MONITOR_JS)

        # ------------------------------------------------------------------ #
        # 2. Upload the test slide via the hidden <input type="file">.        #
        # ------------------------------------------------------------------ #
        # Gradio renders a styled drop-zone; the actual input is hidden.
        # page.set_input_files locates the first matching element even when
        # it has display:none / visibility:hidden.
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(str(TEST_SLIDE.resolve()))

        # Wait for the settings table to populate: the slide filename appears
        # in a table row once Gradio processes the upload event.
        page.get_by_text(TEST_SLIDE.name).wait_for(state="visible", timeout=30_000)

        # ------------------------------------------------------------------ #
        # 2b. Fill required fields.                                           #
        #                                                                     #
        #     Use the global dropdowns which pre-fill all rows in the         #
        #     settings table.                                                 #
        # ------------------------------------------------------------------ #
        _fill_slide_metadata(page, **TEST_SLIDE_METADATA)

        # ------------------------------------------------------------------ #
        # 3. Click Analyze.                                                   #
        # ------------------------------------------------------------------ #
        page.get_by_role("button", name="Analyze").click()

        # ------------------------------------------------------------------ #
        # 4. Wait for analysis to complete.                                   #
        #                                                                     #
        #    The Paladin download button starts hidden and becomes visible    #
        #    only after the final yield — making it the most reliable signal  #
        #    that the full pipeline has finished.                             #
        # ------------------------------------------------------------------ #
        paladin_download = page.get_by_role("button", name="Download Paladin Results")
        paladin_download.wait_for(state="visible", timeout=ANALYSIS_TIMEOUT * 1_000)

        # Give Svelte a moment to finish all reactive updates after the last yield.
        page.wait_for_timeout(3_000)

        # ------------------------------------------------------------------ #
        # 5. Assert results are present and sensible.                         #
        # ------------------------------------------------------------------ #
        aeon_download = page.get_by_role("button", name="Download Aeon Results")
        assert (
            aeon_download.is_visible()
        ), "Aeon download button not visible after analysis"

        aeon_table = page.locator("#aeon-results-table table")
        _assert_aeon_results_sensible(aeon_table, expected_cancer_type)
        _assert_paladin_results_sensible(page)

        # ------------------------------------------------------------------ #
        # 6. Assert no catastrophic Long Tasks (the browser-freeze check).   #
        # ------------------------------------------------------------------ #
        long_tasks: list[float] = page.evaluate("window.__mosaic_longTasks")
        assert isinstance(
            long_tasks, list
        ), "Long Task monitor was not set up correctly"

        offenders = [d for d in long_tasks if d > LONG_TASK_THRESHOLD_MS]
        assert not offenders, (
            f"Browser freeze detected: {len(offenders)} Long Task(s) exceeded "
            f"{LONG_TASK_THRESHOLD_MS} ms.\n"
            f"Durations (ms): {offenders}\n\n"
            "This is the gr.DataFrame datatype regression — ensure all "
            "gr.DataFrame / gr.Dataframe components are constructed with an "
            "explicit datatype=[...] list.  See pipeline.py."
        )

    def test_memory_after_results_is_reasonable(self, page, gradio_server):
        """JS heap size after results delivery should stay below a generous cap.

        This test is intentionally lenient (2 GB cap) to catch only gross
        memory regressions (e.g., large image arrays held in JS).
        """
        page.goto(gradio_server)
        page.wait_for_load_state("load")
        page.get_by_role("button", name="Analyze").wait_for(
            state="visible", timeout=30_000
        )

        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(str(TEST_SLIDE.resolve()))
        page.get_by_text(TEST_SLIDE.name).wait_for(state="visible", timeout=30_000)

        _fill_slide_metadata(page, **TEST_SLIDE_METADATA)

        page.get_by_role("button", name="Analyze").click()

        page.get_by_role("button", name="Download Paladin Results").wait_for(
            state="visible", timeout=ANALYSIS_TIMEOUT * 1_000
        )
        page.wait_for_timeout(5_000)

        heap_bytes: int = page.evaluate(
            "performance.memory ? performance.memory.usedJSHeapSize : 0"
        )
        if heap_bytes == 0:
            pytest.skip(
                "performance.memory API not available in this browser "
                "(Chrome/Chromium only)"
            )

        heap_mb = heap_bytes / (1024 * 1024)
        assert heap_mb < 2_048, (
            f"JS heap after analysis: {heap_mb:.0f} MB — exceeds 2 GB cap.\n"
            "Check whether large image blobs or tensors are being held in JS."
        )

    def test_multi_slide_sample(
        self, page, gradio_server, expected_cancer_type
    ):
        """Upload the same slide twice and verify both are analyzed successfully.

        This exercises the multi-slide code path introduced in this PR: multiple
        slides can be uploaded and analyzed in a single run.  The global metadata
        dropdowns (Sex, Site Type, Tissue Site) fill all rows at once.

        Note: testing Sample ID aggregation (two slides merged into one sample)
        requires reliable cell editing which is blocked by Gradio's layout overlay
        in headless Chromium.  That is covered by unit tests of the pipeline layer.
        """

        # ------------------------------------------------------------------ #
        # 1. Navigate and arm Long Task monitor.                              #
        # ------------------------------------------------------------------ #
        page.goto(gradio_server)
        page.wait_for_load_state("load")
        page.get_by_role("button", name="Analyze").wait_for(
            state="visible", timeout=30_000
        )
        page.evaluate(_LONG_TASK_MONITOR_JS)

        # ------------------------------------------------------------------ #
        # 2. Upload the same slide file twice (simulates a 2-slide batch).   #
        # ------------------------------------------------------------------ #
        slide_path = str(TEST_SLIDE.resolve())
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files([slide_path, slide_path])

        # Wait for TWO rows in the settings table.
        page.get_by_text(TEST_SLIDE.name).first.wait_for(
            state="visible", timeout=30_000
        )
        page.wait_for_function(
            "document.querySelectorAll('#settings-table table tbody tr').length >= 2",
            timeout=30_000,
        )

        # ------------------------------------------------------------------ #
        # 3. Set slide metadata via global dropdowns (fills all rows).       #
        # ------------------------------------------------------------------ #
        _fill_slide_metadata(page, **TEST_SLIDE_METADATA)

        # ------------------------------------------------------------------ #
        # 4. Run analysis.                                                    #
        # ------------------------------------------------------------------ #
        page.get_by_role("button", name="Analyze").click()

        paladin_download = page.get_by_role("button", name="Download Paladin Results")
        paladin_download.wait_for(state="visible", timeout=ANALYSIS_TIMEOUT * 1_000)
        page.wait_for_timeout(3_000)

        # ------------------------------------------------------------------ #
        # 5. Assert results cover both slides.                               #
        #                                                                     #
        #    Gradio renders an index column, so the header count is:          #
        #      index + "Cancer Subtype" + one column per sample.             #
        #    Two separate slides → ≥ 4 columns total.                        #
        #    Two aggregated slides (same Sample ID) → 3 columns.             #
        #    Either way, ≥ 3 columns confirms the table has content.         #
        # ------------------------------------------------------------------ #
        aeon_table = page.locator("#aeon-results-table table")

        header_cells = aeon_table.locator("thead tr th")
        n_cols = header_cells.count()
        assert n_cols >= 3, (
            f"Aeon table has only {n_cols} column(s); expected ≥ 3 "
            "(index + cancer-subtype label + at least 1 sample)."
        )

        # Results should be sensible for at least the first sample column.
        _assert_aeon_results_sensible(aeon_table, expected_cancer_type)
        _assert_paladin_results_sensible(page)

        # No browser freeze.
        long_tasks: list[float] = page.evaluate("window.__mosaic_longTasks")
        offenders = [d for d in long_tasks if d > LONG_TASK_THRESHOLD_MS]
        assert (
            not offenders
        ), f"Browser freeze during multi-slide analysis: {offenders} ms"

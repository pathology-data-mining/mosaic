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


# ---------------------------------------------------------------------------
# Session-scoped fixture: one server for the whole test module
# ---------------------------------------------------------------------------


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

    try:
        _wait_for_server(url, SERVER_STARTUP_TIMEOUT)
    except TimeoutError as exc:
        # Capture server output for diagnosis before giving up.
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail(f"{exc}\n\nServer output:\n{out.decode(errors='replace')[-3000:]}")

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

    def test_analysis_completes_without_browser_freeze(self, page, gradio_server):
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

        # Wait for the settings table to populate (Gradio yields it quickly).
        page.wait_for_selector(
            "table",
            state="visible",
            timeout=30_000,
        )

        # ------------------------------------------------------------------ #
        # 2b. Fill required fields.                                           #
        #                                                                     #
        #     Sex is required; use the global dropdown which feeds into the   #
        #     settings table for the uploaded slide.                          #
        # ------------------------------------------------------------------ #
        sex_input = page.get_by_label("Sex")
        sex_input.click()
        page.get_by_role("option", name="Female").click()

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
        # 5. Assert results are present.                                      #
        # ------------------------------------------------------------------ #
        # Aeon download button should also be visible.
        aeon_download = page.get_by_role("button", name="Download Aeon Results")
        assert (
            aeon_download.is_visible()
        ), "Aeon download button not visible after analysis"

        # Aeon table should contain at least one data row.
        aeon_rows = page.locator("table tbody tr")
        assert aeon_rows.count() > 0, "Aeon table has no data rows"

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
        page.wait_for_selector("table", state="visible", timeout=30_000)

        sex_input = page.get_by_label("Sex")
        sex_input.click()
        page.get_by_role("option", name="Female").click()

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

    def test_multi_slide_sample(self, page, gradio_server):
        """Upload the same slide twice with a shared Sample ID, verify results
        are aggregated into a single sample output.

        This exercises the multi-slide-per-sample code path introduced in this
        PR: features from both slides are aggregated before Aeon/Paladin
        inference, so the result tables should contain exactly ONE sample column
        (Aeon) / ONE sample block (Paladin) rather than two.
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
        # 2. Upload the same slide file twice (simulates a 2-slide sample).  #
        # ------------------------------------------------------------------ #
        slide_path = str(TEST_SLIDE.resolve())
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files([slide_path, slide_path])

        # Wait for TWO rows in the settings table.
        page.wait_for_selector("table", state="visible", timeout=30_000)
        page.wait_for_function(
            "document.querySelectorAll('table tbody tr').length >= 2",
            timeout=30_000,
        )

        # ------------------------------------------------------------------ #
        # 3. Set the same Sample ID for both rows via the editable table.    #
        #                                                                     #
        #    "Sample ID" is the last column. Click the cell in each row,      #
        #    clear it, and type a shared ID.                                  #
        # ------------------------------------------------------------------ #
        shared_sample_id = "multi_slide_test"
        rows = page.locator("table tbody tr")
        for i in range(rows.count()):
            # Last td in the row = Sample ID column.
            cell = rows.nth(i).locator("td").last
            cell.click()
            page.keyboard.press("Control+a")
            page.keyboard.type(shared_sample_id)
            page.keyboard.press("Tab")  # commit the edit

        # ------------------------------------------------------------------ #
        # 4. Set Sex (required for inference).                               #
        # ------------------------------------------------------------------ #
        sex_input = page.get_by_label("Sex")
        sex_input.click()
        page.get_by_role("option", name="Female").click()

        # ------------------------------------------------------------------ #
        # 5. Run analysis.                                                    #
        # ------------------------------------------------------------------ #
        page.get_by_role("button", name="Analyze").click()

        paladin_download = page.get_by_role("button", name="Download Paladin Results")
        paladin_download.wait_for(state="visible", timeout=ANALYSIS_TIMEOUT * 1_000)
        page.wait_for_timeout(3_000)

        # ------------------------------------------------------------------ #
        # 6. Assert results are for ONE sample, not two.                     #
        #                                                                     #
        #    Aeon table: one column per sample → 2 columns total             #
        #      (cancer-subtype label column + 1 confidence column).          #
        #    The confidence column header should equal the shared sample ID. #
        # ------------------------------------------------------------------ #
        # Locate the Aeon results table by its label (Gradio renders the
        # label as a <span> above the component; grab the next table in DOM).
        aeon_label = page.get_by_text("Aeon — Cancer Subtype Predictions").first
        aeon_table = aeon_label.locator("xpath=following::table[1]")

        header_cells = aeon_table.locator("thead tr th")
        n_cols = header_cells.count()
        assert n_cols == 2, (
            f"Aeon table has {n_cols} columns; expected 2 (label + 1 sample). "
            "Multi-slide aggregation may not have fired — two separate samples "
            "would produce 3 columns."
        )

        # The single confidence column header should equal the shared Sample ID.
        sample_col_header = header_cells.nth(1).inner_text().strip()
        assert (
            sample_col_header == shared_sample_id
        ), f"Aeon column header is '{sample_col_header}'; expected '{shared_sample_id}'."

        # No browser freeze.
        long_tasks: list[float] = page.evaluate("window.__mosaic_longTasks")
        offenders = [d for d in long_tasks if d > LONG_TASK_THRESHOLD_MS]
        assert (
            not offenders
        ), f"Browser freeze during multi-slide analysis: {offenders} ms"

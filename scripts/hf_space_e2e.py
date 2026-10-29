#!/usr/bin/env python
"""End-to-end health check of the deployed Mosaic Hugging Face Space.

Drives the live Space through a real browser (Playwright) and runs a full
analysis on a public TCGA-OV diagnostic slide.  Designed to be run on a
schedule (e.g. by a Claude Code cloud routine) from a minimal environment:
it does NOT import ``mosaic`` and only needs ``playwright`` installed::

    pip install playwright && playwright install --with-deps chromium
    python scripts/hf_space_e2e.py --out-dir hf_e2e_out/

Stages (run in order, first failure stops the run):

    runtime     HF runtime API reports RUNNING (wakes a sleeping Space)
    ui_load     App page loads and the Analyze button is visible
    tcga_fetch  "Fetch from TCGA" downloads the slide on the Space from GDC
    download    Same slide is downloaded locally from GDC for upload
    analysis    Slide uploaded, Analyze clicked, full pipeline completes
    results     Aeon/Paladin tables are sensible and the browser never froze

The analysis uses the *upload* path rather than the TCGA path because
TCGA-fetched slides have their results cached by settings hash, which would
skip model inference on every run after the first.

Writes ``report.json`` (per-stage status, timings, details) and a screenshot
for any failed browser stage into ``--out-dir``.  Exit code 0 means every
stage passed.
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_SPACE_ID = "raylim/mosaic"
DEFAULT_URL = "https://raylim-mosaic.hf.space"

# Smallest TCGA-OV diagnostic slide on GDC (~90 MB, open access).
DEFAULT_TCGA_BARCODE = "TCGA-57-1582-01Z-00-DX1"
DEFAULT_TCGA_UUID = "ac4a38b8-d5af-4ec0-9a68-80f325ac3b7a"
DEFAULT_SLIDE_METADATA = {
    "sex": "Female",
    "site_type": "Primary",
    "tissue_site": "Ovary",
}
DEFAULT_EXPECTED_CANCER_TYPE = "HGSOC"

GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
HF_SPACE_API = "https://huggingface.co/api/spaces"

# Runtime stages that will become RUNNING on their own (after a wake-up GET).
TRANSIENT_STAGES = {"SLEEPING", "BUILDING", "APP_STARTING", "RUNNING_BUILDING"}
FATAL_STAGES = {"RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "PAUSED", "STOPPED"}

# errorMessage patterns that point at the HF host rather than our image; a
# factory restart usually lands the Space on a healthy node.
INFRA_ERROR_RE = re.compile(r"OCI runtime|containerd|MOUNT_ATTR_IDMAP|nvidia", re.I)

UI_TIMEOUT_MS = 60_000
TCGA_FETCH_TIMEOUT_MS = 10 * 60_000
UPLOAD_TIMEOUT_MS = 5 * 60_000

# Mirrors tests/e2e/test_browser_e2e.py: a single browser Long Task longer than
# this is the gr.DataFrame datatype regression (~420 s freeze).
LONG_TASK_THRESHOLD_MS = 5_000
LONG_TASK_MONITOR_JS = """
window.__mosaic_longTasks = [];
new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
        window.__mosaic_longTasks.push(entry.duration);
    }
}).observe({ entryTypes: ['longtask'] });
"""


class StageFailure(Exception):
    """A stage failed; ``category`` hints at the likely owner of the failure."""

    def __init__(self, message: str, category: str = "app", details=None):
        super().__init__(message)
        self.category = category
        self.details = details or {}


def _get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _http_status(url: str, timeout: int = 30) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError):
        return 0


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_runtime(args) -> dict:
    """Wait until the HF runtime API reports RUNNING and the app answers 200."""
    runtime_url = f"{HF_SPACE_API}/{args.space_id}/runtime"
    deadline = time.monotonic() + args.startup_timeout
    woke = False
    seen = []
    while True:
        try:
            runtime = _get_json(runtime_url)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise StageFailure(f"HF runtime API unreachable: {exc}", "hf_infra")
        stage = runtime.get("stage")
        if not seen or seen[-1] != stage:
            seen.append(stage)
            print(f"  runtime stage: {stage}", flush=True)

        if stage in FATAL_STAGES:
            raise StageFailure(
                f"Space is in {stage}: {runtime.get('errorMessage', '(no message)')}",
                "build" if stage == "BUILD_ERROR" else "hf_infra",
                {"runtime": runtime},
            )
        if stage == "RUNNING" and _http_status(args.url) == 200:
            return {
                "stages_seen": seen,
                "hardware": (runtime.get("hardware") or {}).get("current"),
            }
        if stage in TRANSIENT_STAGES and not woke:
            # A request to the app URL is what wakes a sleeping Space.
            _http_status(args.url, timeout=10)
            woke = True
        if time.monotonic() > deadline:
            raise StageFailure(
                f"Space not RUNNING after {args.startup_timeout}s "
                f"(stages seen: {seen})",
                "hf_infra",
                {"runtime": runtime},
            )
        time.sleep(15)


def _is_restartable(stage_entry: dict) -> bool:
    runtime = stage_entry.get("details", {}).get("runtime", {})
    return runtime.get("stage") in {"RUNTIME_ERROR", "PAUSED", "STOPPED"} or bool(
        INFRA_ERROR_RE.search(runtime.get("errorMessage", ""))
    )


def factory_restart(space_id: str, token: str) -> int:
    """Factory-restart the Space; returns the HTTP status."""
    req = urllib.request.Request(
        f"{HF_SPACE_API}/{space_id}/restart?factory=true",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _open_app(page, url: str) -> None:
    page.goto(url, timeout=UI_TIMEOUT_MS)
    page.wait_for_load_state("load")
    # Gradio keeps an SSE connection open, so "networkidle" never fires.
    page.get_by_role("button", name="Analyze").wait_for(
        state="visible", timeout=UI_TIMEOUT_MS
    )


def stage_ui_load(page, args) -> dict:
    _open_app(page, args.url)
    return {"title": page.title()}


def stage_tcga_fetch(page, args) -> dict:
    """Fetch the slide through the Space's TCGA integration (no analysis)."""
    _open_app(page, args.url)
    page.get_by_role("radio", name="Fetch from TCGA").check()
    page.get_by_label("TCGA Slide ID").fill(args.tcga_barcode)
    page.get_by_role("button", name="Fetch Slide").click()

    status = page.get_by_label("Fetch Status")
    status.wait_for(state="visible", timeout=TCGA_FETCH_TIMEOUT_MS)
    page.wait_for_function(
        "el => el.value && el.value.includes('Slide fetched')",
        arg=status.element_handle(),
        timeout=TCGA_FETCH_TIMEOUT_MS,
    )
    return {"status": status.input_value()}


def stage_download(args) -> dict:
    """Download the public slide from GDC so it can be uploaded."""
    dest = args.cache_dir / f"{args.tcga_uuid}.svs"
    if dest.exists() and dest.stat().st_size > 0:
        return {"path": str(dest), "cached": True, "mb": dest.stat().st_size / 1e6}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        with (
            urllib.request.urlopen(
                f"{GDC_DATA_ENDPOINT}/{args.tcga_uuid}", timeout=120
            ) as resp,
            tmp.open("wb") as fh,
        ):
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        raise StageFailure(f"GDC download failed: {exc}", "external_gdc")
    tmp.rename(dest)
    return {"path": str(dest), "cached": False, "mb": dest.stat().st_size / 1e6}


def _select_dropdown(page, label: str, value: str) -> None:
    page.get_by_label(label).click()
    page.get_by_role("option", name=value).click()


def stage_analysis(page, args, slide_path: Path) -> dict:
    _open_app(page, args.url)
    page.evaluate(LONG_TASK_MONITOR_JS)

    # The filename appears in the drop zone before the upload POST finishes;
    # clicking Analyze in that window is silently dropped, so wait for the POST.
    with page.expect_response(
        lambda r: "/gradio_api/upload?" in r.url and r.request.method == "POST",
        timeout=UPLOAD_TIMEOUT_MS,
    ) as upload:
        page.locator('input[type="file"]').first.set_input_files(str(slide_path))
    if not upload.value.ok:
        raise StageFailure(f"Slide upload failed: HTTP {upload.value.status}")
    # No wait on the filename text: once the Space has stored the slide, the
    # name also appears in hidden "existing slides" elements.
    # Let the upload's .then() chain (storage refresh) settle.
    page.wait_for_timeout(3_000)
    _select_dropdown(page, "Sex", DEFAULT_SLIDE_METADATA["sex"])
    _select_dropdown(page, "Site Type", DEFAULT_SLIDE_METADATA["site_type"])
    _select_dropdown(page, "Tissue Site", DEFAULT_SLIDE_METADATA["tissue_site"])

    started = time.monotonic()
    page.get_by_role("button", name="Analyze").click()
    page.get_by_role("button", name="Download Paladin Results").wait_for(
        state="visible", timeout=args.analysis_timeout * 1_000
    )
    # Let Svelte finish reactive updates after the final yield.
    page.wait_for_timeout(3_000)
    return {"analysis_seconds": round(time.monotonic() - started, 1)}


# Finds a gr.Dataframe block by elem_id (current main) or by its label (older
# deployments lack the elem_ids) and returns the cell texts of its data table.
# Gradio renders a hidden one-row sizing table next to the real one, so the
# table with the most rows wins.
_READ_TABLE_JS = """([elemId, label]) => {
    let block = document.getElementById(elemId);
    if (!block) {
        block = [...document.querySelectorAll('.block')].find(b =>
            b.querySelector('table') && b.innerText.includes(label));
    }
    if (!block) return null;
    const tables = [...block.querySelectorAll('table')];
    const best = tables.reduce((a, t) =>
        t.querySelectorAll('tbody tr').length >
        (a ? a.querySelectorAll('tbody tr').length : -1) ? t : a, null);
    return best ? [...best.querySelectorAll('tbody tr')].map(tr =>
        [...tr.querySelectorAll('td')].map(td => td.innerText.trim())) : [];
}"""

AEON_TABLE = ("aeon-results-table", "Cancer Subtype Inference Confidence")
PALADIN_TABLE = ("paladin-results-table", "Biomarker Inference (Paladin model)")


def _read_table(page, table: tuple[str, str]) -> list[list[str]]:
    rows = page.evaluate(_READ_TABLE_JS, list(table))
    if rows is None:
        raise StageFailure(f"Results table {table[1]!r} not found", "ui_regression")
    return rows


def stage_results(page, args) -> dict:
    """Port of the sanity checks in tests/e2e/test_browser_e2e.py."""
    aeon = _read_table(page, AEON_TABLE)
    if len(aeon) < 5:
        raise StageFailure(f"Aeon table has {len(aeon)} rows; expected >= 5")
    cancer_types, confidences = [], []
    for i, row in enumerate(aeon):
        try:
            conf = float(row[1])
        except (IndexError, ValueError):
            raise StageFailure(f"Aeon row {i} has non-numeric confidence: {row}")
        if not row[0] or not 0.0 <= conf <= 1.0:
            raise StageFailure(f"Aeon row {i} is invalid: {row}")
        cancer_types.append(row[0])
        confidences.append(conf)
    top5 = list(zip(cancer_types[:5], confidences[:5]))
    if max(confidences) <= 0.05:
        raise StageFailure(
            f"Top Aeon confidence {max(confidences):.4f} is suspiciously low",
            "model_regression",
            {"aeon_top5": top5},
        )
    expected = args.expected_cancer_type
    if expected and not any(expected in ct for ct in cancer_types[:3]):
        raise StageFailure(
            f"Expected '{expected}' not in Aeon top 3",
            "model_regression",
            {"aeon_top5": top5},
        )

    paladin = _read_table(page, PALADIN_TABLE)
    if not paladin:
        raise StageFailure("Paladin table has no biomarker rows")
    for i, row in enumerate(paladin):
        try:
            prob = float(row[3])
        except (IndexError, ValueError):
            raise StageFailure(f"Paladin row {i} has non-numeric probability: {row}")
        if not row[2] or not 0.0 <= prob <= 1.0:
            raise StageFailure(f"Paladin row {i} is invalid: {row}")

    long_tasks = page.evaluate("window.__mosaic_longTasks") or []
    offenders = [d for d in long_tasks if d > LONG_TASK_THRESHOLD_MS]
    if offenders:
        raise StageFailure(
            f"Browser freeze: Long Task(s) of {offenders} ms "
            "(gr.DataFrame datatype regression?)",
            "ui_regression",
        )
    return {
        "aeon_top5": top5,
        "paladin_rows": len(paladin),
        "max_long_task_ms": max(long_tasks, default=0),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_stage(report: dict, name: str, fn, page=None):
    print(f"[{name}] ...", flush=True)
    started = time.monotonic()
    entry = {"stage": name}
    report["stages"].append(entry)
    try:
        details = fn()
    except Exception as exc:  # pylint: disable=broad-except
        entry.update(
            status="fail",
            seconds=round(time.monotonic() - started, 1),
            error=str(exc).splitlines()[0][:500] if str(exc) else repr(exc),
            category=getattr(exc, "category", "unclassified"),
            details=getattr(exc, "details", {}),
        )
        if not isinstance(exc, StageFailure):
            entry["traceback"] = traceback.format_exc()[-4000:]
        if page is not None:
            shot = report["out_dir"] / f"{name}_failure.png"
            try:
                page.screenshot(path=str(shot), full_page=True)
                entry["screenshot"] = str(shot)
            except Exception:  # pylint: disable=broad-except
                pass
        print(f"[{name}] FAIL ({entry['category']}): {entry['error']}", flush=True)
        raise
    entry.update(
        status="pass", seconds=round(time.monotonic() - started, 1), details=details
    )
    print(f"[{name}] pass ({entry['seconds']}s)", flush=True)
    return details


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--space-id", default=DEFAULT_SPACE_ID)
    p.add_argument("--tcga-barcode", default=DEFAULT_TCGA_BARCODE)
    p.add_argument("--tcga-uuid", default=DEFAULT_TCGA_UUID)
    p.add_argument("--expected-cancer-type", default=DEFAULT_EXPECTED_CANCER_TYPE)
    p.add_argument("--out-dir", type=Path, default=Path("hf_e2e_out"))
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "mosaic-hf-e2e",
        help="Where the GDC slide is downloaded (reused across runs)",
    )
    p.add_argument("--startup-timeout", type=int, default=25 * 60)
    p.add_argument("--analysis-timeout", type=int, default=25 * 60)
    p.add_argument("--skip-tcga-fetch", action="store_true")
    p.add_argument(
        "--restart-on-infra-error",
        action="store_true",
        help="If the Space is down with an HF infrastructure error, factory-restart "
        "it once (needs HF_TOKEN with write access) and retry",
    )
    p.add_argument("--headed", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "url": args.url,
        "space_id": args.space_id,
        "tcga_barcode": args.tcga_barcode,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_dir": args.out_dir,
        "stages": [],
    }
    try:
        space = _get_json(f"{HF_SPACE_API}/{args.space_id}")
        report["space_sha"] = space.get("sha")
        report["space_last_modified"] = space.get("lastModified")
    except (urllib.error.URLError, OSError, ValueError):
        pass

    ok = False
    try:
        try:
            _run_stage(report, "runtime", lambda: stage_runtime(args))
        except StageFailure:
            token = os.environ.get("HF_TOKEN")
            if not (
                args.restart_on_infra_error
                and token
                and _is_restartable(report["stages"][-1])
            ):
                raise
            status = factory_restart(args.space_id, token)
            report["restart"] = {"http_status": status}
            print(f"  factory restart requested (HTTP {status})", flush=True)
            if status != 200:
                raise
            time.sleep(30)
            _run_stage(report, "runtime_after_restart", lambda: stage_runtime(args))
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            page = browser.new_page(viewport={"width": 1600, "height": 1200})
            try:
                _run_stage(report, "ui_load", lambda: stage_ui_load(page, args), page)
                if not args.skip_tcga_fetch:
                    _run_stage(
                        report,
                        "tcga_fetch",
                        lambda: stage_tcga_fetch(page, args),
                        page,
                    )
                slide = Path(
                    _run_stage(report, "download", lambda: stage_download(args))["path"]
                )
                _run_stage(
                    report,
                    "analysis",
                    lambda: stage_analysis(page, args, slide),
                    page,
                )
                _run_stage(report, "results", lambda: stage_results(page, args), page)
                ok = True
            finally:
                browser.close()
    except Exception:  # pylint: disable=broad-except
        if not report["stages"] or report["stages"][-1].get("status") != "fail":
            # Failure outside a stage (e.g. Playwright not installed).
            report["stages"].append(
                {
                    "stage": "setup",
                    "status": "fail",
                    "category": "harness",
                    "error": traceback.format_exc()[-2000:],
                }
            )

    report["passed"] = ok
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    failed = [s for s in report["stages"] if s.get("status") == "fail"]
    report["failed_stage"] = failed[-1]["stage"] if failed and not ok else None
    report["out_dir"] = str(args.out_dir)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"{'PASS' if ok else 'FAIL'} — report: {args.out_dir / 'report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

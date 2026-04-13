#!/bin/bash
# Submit the Mosaic end-to-end browser test to SLURM.
#
# The job runs on an A100 node (tllihpcgpu[1-4]), activates the project
# virtualenv, starts a headless Chromium session via Playwright, and runs
# the full analysis pipeline against a live Gradio server.
#
# Usage:
#   sbatch scripts/submit_e2e_test.sh
#   sbatch --export=ALL,BROWSER=firefox scripts/submit_e2e_test.sh
#   sbatch --export=ALL,HEADED=1 scripts/submit_e2e_test.sh   # Xvfb window
#
# Environment variables (all optional):
#   BROWSER   - Playwright browser to use (chromium | firefox | webkit).
#               Default: chromium
#   HEADED    - Set to 1 to run with a virtual display (Xvfb) instead of
#               headless.  Requires xvfb-run to be on PATH.
#   HF_TOKEN  - HuggingFace token for model download (if not already cached).
#   PYTEST_EXTRA_ARGS - Extra arguments forwarded to pytest.

#SBATCH --job-name=mosaic-e2e
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/e2e_%j.out
#SBATCH --error=logs/e2e_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
PYTEST="$REPO_DIR/.venv/bin/pytest"
LOG_DIR="$REPO_DIR/logs"

mkdir -p "$LOG_DIR"

echo "========================================"
echo "Mosaic E2E Browser Test"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : $(hostname)"
echo "Repo     : $REPO_DIR"
echo "Started  : $(date -Iseconds)"
echo "========================================"

# ---------------------------------------------------------------------------
# Activate virtualenv
# ---------------------------------------------------------------------------
source "$REPO_DIR/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Verify prerequisites
# ---------------------------------------------------------------------------
if [ ! -f "$REPO_DIR/tests/testdata/948176.svs" ]; then
    echo "ERROR: Test slide not found at tests/testdata/948176.svs" >&2
    exit 1
fi

if ! "$VENV_PYTHON" -c "from playwright.sync_api import sync_playwright" 2>/dev/null; then
    echo "ERROR: playwright not installed in venv. Run: pip install playwright && playwright install chromium" >&2
    exit 1
fi

# Ensure Playwright browsers are installed (no-op if already present).
"$VENV_PYTHON" -m playwright install "${BROWSER:-chromium}" --with-deps 2>&1 | tail -5

# ---------------------------------------------------------------------------
# GPU sanity check
# ---------------------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
"$VENV_PYTHON" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ---------------------------------------------------------------------------
# Build pytest command
# ---------------------------------------------------------------------------
PYTEST_CMD=(
    "$PYTEST"
    "tests/e2e/"
    "--e2e"
    "-v"
    "--no-cov"
    "--browser" "${BROWSER:-chromium}"
    "--timeout=1200"          # 20-min hard timeout per test
)

# Run headed via Xvfb virtual display if HEADED=1.
if [ "${HEADED:-0}" = "1" ]; then
    if command -v xvfb-run &>/dev/null; then
        PYTEST_CMD=("xvfb-run" "--auto-servernum" "--server-args=-screen 0 1920x1080x24" "${PYTEST_CMD[@]}" "--headed")
    else
        echo "WARNING: xvfb-run not found; falling back to headless mode." >&2
    fi
fi

# Append any extra args passed via environment.
if [ -n "${PYTEST_EXTRA_ARGS:-}" ]; then
    # Word-split is intentional here.
    # shellcheck disable=SC2086
    PYTEST_CMD+=($PYTEST_EXTRA_ARGS)
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$REPO_DIR"
echo ""
echo "Command: ${PYTEST_CMD[*]}"
echo ""

"${PYTEST_CMD[@]}"
EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULT: PASSED"
else
    echo "RESULT: FAILED (exit $EXIT_CODE)"
fi
echo "Finished : $(date -Iseconds)"
echo "========================================"

exit $EXIT_CODE

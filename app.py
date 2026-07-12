"""Entry point for Hugging Face Spaces ZeroGPU deployment.

This module serves as the main entry point when deploying Mosaic to
Hugging Face Spaces using ZeroGPU (Gradio SDK). It installs dependencies
at runtime and launches the Gradio interface.

NOTE: This is NOT used for Docker SDK deployments. For Docker deployments,
the Dockerfile handles all dependency installation and uses the 'mosaic'
CLI command as the entrypoint.

Runtime dependency installation (via pip) is required for ZeroGPU deployment
because:
1. Paladin is a private repository requiring GH_TOKEN authentication
2. requirements.txt is processed before the app starts and doesn't have
   access to secrets
3. ZeroGPU Spaces use Gradio SDK which expects runtime installation
"""

import os
import subprocess
import sys

# Import spaces first to avoid CUDA initialization issues
try:
    import spaces
except ImportError:
    pass

# Add src directory to path for Hugging Face Spaces
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mosaic import __version__
from mosaic.build_info import log_runtime_build_info
from mosaic.gradio_app import download_and_process_models
from mosaic.ui import launch_gradio

if __name__ == "__main__":
    log_runtime_build_info(__version__)

    # Fix OMP_NUM_THREADS environment variable if invalid
    omp_threads = os.environ.get("OMP_NUM_THREADS", "")
    try:
        # Check if value is a valid positive integer
        if not omp_threads or int(omp_threads) <= 0:
            raise ValueError("Invalid OMP_NUM_THREADS")
    except (ValueError, AttributeError):
        # Set to a reasonable default based on CPU count
        import multiprocessing

        num_cpus = multiprocessing.cpu_count()
        # Use half of available CPUs, minimum 1, maximum 8 for stability
        os.environ["OMP_NUM_THREADS"] = str(max(1, min(8, num_cpus // 2)))

    # Enable HF transfer for faster downloads on Spaces
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Install flash-attn for H100 acceleration (if on Spaces)
    if os.environ.get("SPACE_ID"):
        try:
            subprocess.run(
                "pip install flash-attn --no-build-isolation",
                shell=True,
                check=False,
                capture_output=True,
            )
        except Exception:
            pass  # Flash-attn is optional, continue without it

    # Install Paladin from GitHub
    GIT_TOKEN = os.environ.get("GH_TOKEN")
    if GIT_TOKEN:
        subprocess.run(
            f"pip install git+https://{GIT_TOKEN}@github.com/pathology-data-mining/paladin.git@dev",
            shell=True,
        )

    # Download models and initialize cancer subtype mappings
    download_and_process_models()

    # Launch the Gradio interface
    # Use default settings suitable for Hugging Face Spaces
    launch_gradio(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

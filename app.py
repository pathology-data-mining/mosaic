"""Entry point for Hugging Face Spaces deployment.

This module serves as the main entry point when deploying Mosaic to
Hugging Face Spaces. It initializes the models and launches the Gradio interface.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mosaic.gradio_app import download_and_process_models
from mosaic.ui import launch_gradio

if __name__ == "__main__":
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

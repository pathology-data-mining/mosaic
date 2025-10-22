"""Entry point for Hugging Face Spaces deployment.

This module serves as the main entry point when deploying Mosaic to
Hugging Face Spaces. It initializes the models and launches the Gradio interface.
"""

from mosaic.gradio_app import download_and_process_models, main
from mosaic.ui import launch_gradio

if __name__ == "__main__":
    # Download models and initialize cancer subtype mappings
    download_and_process_models()
    
    # Launch the Gradio interface
    # Use default settings suitable for Hugging Face Spaces
    launch_gradio(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

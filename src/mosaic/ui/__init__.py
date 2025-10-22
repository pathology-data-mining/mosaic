"""UI module for Mosaic Gradio web interface.

This module provides the web-based user interface for Mosaic,
allowing interactive analysis of whole slide images through a browser.
"""

from .app import launch_gradio

__all__ = ["launch_gradio"]

"""CLI argument parser for Mosaic."""

from argparse import ArgumentParser

from mosaic import __version__
from mosaic.build_info import format_version_string
from mosaic.settings import IHC_SUBTYPES, SEX_OPTIONS


def build_parser() -> ArgumentParser:
    """Build and return the Mosaic CLI argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name for Gradio app",
    )
    parser.add_argument(
        "--server-port", type=int, default=None, help="Server port for Gradio app"
    )
    parser.add_argument(
        "--share", action="store_true", help="Share Gradio app publicly"
    )
    parser.add_argument(
        "--slide-csv",
        type=str,
        help="CSV file with slide settings (for batch processing), see README for format",
    )
    parser.add_argument(
        "--slide-path",
        type=str,
        help=(
            "Path to a single slide (for single slide processing), "
            "not used if --slide-csv is provided"
        ),
    )
    parser.add_argument(
        "--site-type",
        type=str,
        choices=["Primary", "Metastatic"],
        default="Primary",
        help="Site type of the slide (for single slide processing)",
    )
    parser.add_argument(
        "--sex",
        type=str,
        choices=SEX_OPTIONS,
        default=None,
        help="Sex of the patient (required for single slide processing)",
    )
    parser.add_argument(
        "--tissue-site",
        type=str,
        default="Unknown",
        help="Tissue site of the slide (for single slide processing)",
    )
    parser.add_argument(
        "--cancer-subtype",
        type=str,
        default="Unknown",
        help=(
            "Cancer subtype of the slide (for single slide processing), "
            "use 'Unknown' to infer with Aeon"
        ),
    )
    parser.add_argument(
        "--ihc-subtype",
        type=str,
        choices=IHC_SUBTYPES,
        default="",
        help="IHC subtype if cancer subtype is breast (for single slide processing)",
    )
    parser.add_argument(
        "--segmentation-config",
        type=str,
        choices=["Biopsy", "Resection", "TCGA"],
        default="Biopsy",
        help="Segmentation configuration (for single slide processing)",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to save output results"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for feature extraction",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip downloading models from HuggingFace (assumes models are already cached)",
    )
    parser.add_argument(
        "--download-models-only",
        action="store_true",
        help="Download models from HuggingFace and exit without running analysis",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=format_version_string(__version__),
    )
    return parser

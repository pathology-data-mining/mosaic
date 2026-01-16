#!/usr/bin/env python
"""
Export Aeon PyTorch Lightning checkpoint to pickle format for inference.

This script converts a PyTorch Lightning checkpoint (.ckpt) file to a pickle
(.pkl) file that can be used with the Mosaic inference pipeline.

Usage:
    python export_aeon_checkpoint.py \
        --checkpoint data/checkpoint.ckpt \
        --output data/aeon_model.pkl \
        --metadata-dir data/metadata

Requirements:
    - paladin package from git repo (must have AeonLightningModule)
    - PyTorch Lightning
    - Access to metadata files (n_classes.txt, ontology_embedding_dim.txt, target_dict.tsv)
"""

import argparse
import json
import pickle
from pathlib import Path


def load_metadata(metadata_dir: Path):
    """Load metadata required for model initialization.

    Args:
        metadata_dir: Directory containing metadata files

    Returns:
        SimpleMetadata object with n_classes, ontology_embedding_dim, and target_dicts
    """
    # Read n_classes
    with open(metadata_dir / "n_classes.txt") as f:
        n_classes = int(f.read().strip())

    # Read ontology_embedding_dim
    with open(metadata_dir / "ontology_embedding_dim.txt") as f:
        ontology_embedding_dim = int(f.read().strip())

    # Read target_dict (JSON format with single quotes)
    with open(metadata_dir / "target_dict.tsv") as f:
        target_dict_str = f.read().strip().replace("'", '"')
        target_dict = json.loads(target_dict_str)

    # Create simple metadata object
    class SimpleMetadata:
        def __init__(self, n_classes, ontology_embedding_dim, target_dict):
            self.n_classes = n_classes
            self.ontology_embedding_dim = ontology_embedding_dim
            self.target_dicts = [target_dict]

    return SimpleMetadata(n_classes, ontology_embedding_dim, target_dict)


def export_checkpoint(checkpoint_path: Path, output_path: Path, metadata_dir: Path):
    """Export PyTorch Lightning checkpoint to pickle format.

    Args:
        checkpoint_path: Path to .ckpt file
        output_path: Path to save .pkl file
        metadata_dir: Directory containing metadata files
    """
    try:
        from paladin.pl_modules.aeon import AeonLightningModule
    except ImportError:
        raise ImportError(
            "Failed to import AeonLightningModule. "
            "Make sure paladin is installed from the git repository:\n"
            "  uv sync --upgrade-package paladin"
        )

    print(f"Loading metadata from {metadata_dir}...")
    metadata = load_metadata(metadata_dir)

    print(f"Loading checkpoint from {checkpoint_path}...")
    pl_module = AeonLightningModule.load_from_checkpoint(
        str(checkpoint_path), metadata=metadata
    )

    # Extract the model
    model = pl_module.model

    print(f"Saving model to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✓ Successfully exported checkpoint to {output_path}")

    # Print model info
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  Model size: {file_size:.1f} MB")
    print(f"  Model class: {type(model).__name__}")
    print(f"  Number of classes: {metadata.n_classes}")
    print(f"  Ontology embedding dim: {metadata.ontology_embedding_dim}")
    print(f"  Number of histologies: {len(metadata.target_dicts[0]['histologies'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Aeon PyTorch Lightning checkpoint to pickle format"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PyTorch Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save exported model (.pkl)"
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/metadata"),
        help="Directory containing metadata files (default: data/metadata)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {args.metadata_dir}")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Export checkpoint
    export_checkpoint(args.checkpoint, args.output, args.metadata_dir)


if __name__ == "__main__":
    main()

"""Aeon model inference module for cancer subtype prediction.

This module provides functionality to run the Aeon deep learning model
for predicting cancer subtypes from H&E whole slide image features.
"""

import pickle  # nosec
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mosaic.inference.data import (
    SiteType,
    TileFeatureTensorDataset,
    INT_TO_CANCER_TYPE_MAP,
    CANCER_TYPE_TO_INT_MAP,
    encode_sex,
    encode_tissue_site,
)

from loguru import logger

# Cancer types excluded from prediction (too broad or ambiguous)
cancer_types_to_drop = [
    "UDMN",
    "ADNOS",
    "CUP",
    "CUPNOS",
    "BRCNOS",
    "GNOS",
    "SCCNOS",
    "PDC",
    "NSCLC",
    "BRCA",
    "SARCNOS",
    "NETNOS",
    "MEL",
    "RCC",
    "BRCANOS",
    "COADREAD",
    "MUP",
    "NECNOS",
    "UCEC",
    "NOT",
]
col_indices_to_drop = [CANCER_TYPE_TO_INT_MAP[x] for x in cancer_types_to_drop]


BATCH_SIZE = 8
NUM_WORKERS = 8


def run(
    features, model_path, metastatic=False, batch_size=8, num_workers=8, use_cpu=False,
    sex=None, tissue_site_idx=None
):
    """Run Aeon model inference for cancer subtype prediction.

    Args:
        features: NumPy array of tile features extracted from the WSI
        model_path: Path to the pickled Aeon model file
        metastatic: Whether the slide is from a metastatic site
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        use_cpu: Force CPU usage instead of GPU
        sex: Patient sex (0=Male, 1=Female), optional
        tissue_site_idx: Tissue site index (0-56), optional

    Returns:
        tuple: (results_df, part_embedding)
            - results_df: DataFrame with cancer subtypes and confidence scores
            - part_embedding: Torch tensor of the learned part representation
    """
    device = torch.device(
        "cuda" if not use_cpu and torch.cuda.is_available() else "cpu"
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)  # nosec
    model.to(device)
    model.eval()

    # Load the correct mapping from metadata for this model
    from pathlib import Path
    import json

    metadata_dir = Path("data/metadata")
    with open(metadata_dir / "target_dict.tsv") as f:
        target_dict_str = f.read().strip().replace("'", '"')
        target_dict = json.loads(target_dict_str)

    histologies = target_dict['histologies']
    INT_TO_CANCER_TYPE_MAP_LOCAL = {i: histology for i, histology in enumerate(histologies)}
    CANCER_TYPE_TO_INT_MAP_LOCAL = {v: k for k, v in INT_TO_CANCER_TYPE_MAP_LOCAL.items()}

    # Calculate col_indices_to_drop using local mapping
    cancer_types_to_drop = ["UDMN", "ADNOS", "CUP", "CUPNOS", "NOT"]
    col_indices_to_drop_local = [CANCER_TYPE_TO_INT_MAP_LOCAL[x] for x in cancer_types_to_drop if x in CANCER_TYPE_TO_INT_MAP_LOCAL]

    site_type = SiteType.METASTASIS if metastatic else SiteType.PRIMARY

    # For UI, InferenceDataset will just be a single slide.  Sample id is not relevant.
    dataset = TileFeatureTensorDataset(
        site_type=site_type,
        tile_features=features,
        sex=sex,
        tissue_site_idx=tissue_site_idx,
        n_max_tiles=20000,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    results = []
    batch = next(iter(dataloader))
    with torch.no_grad():
        batch["tile_tensor"] = batch["tile_tensor"].to(device)
        if "SEX" in batch:
            batch["SEX"] = batch["SEX"].to(device)
        if "TISSUE_SITE" in batch:
            batch["TISSUE_SITE"] = batch["TISSUE_SITE"].to(device)
        y = model(batch)
        y["logits"][:, col_indices_to_drop_local] = -1e6

        batch_size = y["logits"].shape[0]
        assert batch_size == 1

        softmax = torch.nn.functional.softmax(y["logits"][0], dim=0)
        argmax = torch.argmax(softmax, dim=0)
        class_assignment = INT_TO_CANCER_TYPE_MAP_LOCAL[argmax.item()]
        max_confidence = softmax[argmax].item()
        mean_confidence = torch.mean(softmax).item()

        logger.info(
            f"class {class_assignment} :  confidence {max_confidence:8.5f} "
            f"(mean {mean_confidence:8.5f})"
        )

        part_embedding = y["whole_part_representation"][0].cpu()

        for cancer_subtype, j in sorted(CANCER_TYPE_TO_INT_MAP_LOCAL.items()):
            confidence = softmax[j].item()
            results.append((cancer_subtype, confidence))
        results.sort(key=lambda row: row[1], reverse=True)

    results_df = pd.DataFrame(results, columns=["Cancer Subtype", "Confidence"])

    return results_df, part_embedding


def parse_args():
    parser = ArgumentParser(
        description="Run Aeon inference on a specified set of slides"
    )
    parser.add_argument(
        "-i",
        "--features-path",
        required=True,
        help="Pathname to a .pt file with optimus tile features for this slide",
    )
    parser.add_argument(
        "-o",
        "--output-prediction-path",
        help="The filename for the Aeon predictions file (CSV)",
        required=True,
    )
    parser.add_argument(
        "--output-embedding-path",
        help="The filename for the whole-part representation of the slide (.pt)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Pathname to the pickle file for an Aeon model",
        required=True,
    )
    parser.add_argument(
        "--metastatic", action="store_true", help="Tissue is from a metastatic site"
    )
    parser.add_argument(
        "--sex",
        type=str,
        choices=["Male", "Female", "Unknown"],
        default=None,
        help="Patient sex (Male or Female)",
    )
    parser.add_argument(
        "--tissue-site",
        type=str,
        default=None,
        help="Tissue site name",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=NUM_WORKERS, help="Number of workers"
    )
    parser.add_argument("--use-cpu", action="store_true", help="Use CPU")
    opt = parser.parse_args()

    return opt


def main():
    opt = parse_args()

    output_path = opt.output_prediction_path
    logger.info(f"output_path: '{output_path}'")

    embedding_path = opt.output_embedding_path
    logger.info(f"part_embedding_path: '{embedding_path}'")

    features = torch.load(opt.features_path)

    # Encode sex and tissue site if provided
    sex_encoded = None
    if opt.sex:
        sex_encoded = encode_sex(opt.sex)
        logger.info(f"Using sex: {opt.sex} (encoded as {sex_encoded})")

    tissue_site_idx = None
    if opt.tissue_site:
        tissue_site_idx = encode_tissue_site(opt.tissue_site)
        logger.info(f"Using tissue site: {opt.tissue_site} (encoded as {tissue_site_idx})")

    results_df, part_embedding = run(
        features=features,
        model_path=opt.model_path,
        metastatic=opt.metastatic,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        use_cpu=opt.use_cpu,
        sex=sex_encoded,
        tissue_site_idx=tissue_site_idx,
    )

    results_df.to_csv(output_path, index=False)
    logger.info(f"Wrote {output_path}")

    if embedding_path:
        torch.save(part_embedding, embedding_path)
        logger.info(f"Wrote {embedding_path}")


if __name__ == "__main__":
    main()

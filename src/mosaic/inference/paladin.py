import csv
import pickle  # nosec
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional

import numpy
import pandas as pd
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from mosaic.inference.data import SiteType, TileFeatureTensorDataset

# Constants
BATCH_SIZE = 8
NUM_WORKERS = 16


class UsageError(Exception):
    """A UsageError is raised when there's a problem with the command-line arguments."""

    pass


def load_model_map(model_map_path: str) -> dict[Any, Any]:
    """Load the table mapping cancer_subtypes and targets to the paladin
    model (a pickle file) that predicts that target for that cancer subtype.

    A dict is returned, mapping each cancer_subtype to a table mapping a
    target to the pathname for the model that predicts it.
    """
    models = defaultdict(dict)
    with Path(model_map_path).open() as fp:
        rdr = csv.DictReader(fp)
        for row in rdr:
            cancer_subtype = row["cancer_subtype"]
            target = row["target_name"]
            model = row["model_path"]
            models[cancer_subtype][target] = model
    return models


def load_aeon_scores(df: pd.DataFrame) -> dict[str, float]:
    """Load the output table from a single-slide Aeon run, listing Oncotree
    cancer subtypes and their confidence values.

    A dict is returned, mapping each cancersubtype to its confidence score.
    """
    score = {}
    for _, row in df.iterrows():
        subtype = row["Cancer Subtype"]
        confidence = row["Confidence"]
        score[subtype] = confidence
    return score


def select_cancer_subtypes(aeon_scores: dict[str, float], k=1) -> list[str]:
    """Return the three top-scoring cancer_subtypes, based on the given Aeon scores."""
    sorted_cancer_subtypes = list(
        sorted([(v, k) for k, v in aeon_scores.items()], reverse=True)
    )
    return [cancer_subtype for score, cancer_subtype in sorted_cancer_subtypes[:k]]


def select_models(cancer_subtypes: list[str], model_map: dict[Any, Any]) -> list[Any]:
    """ """
    models = []
    for cancer_subtype, target, model in model_map.items():
        if cancer_subtype in cancer_subtypes:
            models.append((cancer_subtype, target, model))
    return models


def run_model(device, dataset, model_path: str, num_workers, batch_size) -> float:
    """Run inference for the given embeddings and model.
    The point estimate is returned.
    """

    logger.debug(f"[loading model {model_path}]")
    with Path(model_path).open("rb") as f:
        model = pickle.load(f)  # nosec
        # model = CPU_Unpickler(f).load()  # nosec
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    results_df = []
    batch = next(iter(dataloader))
    with torch.no_grad():
        batch["tile_tensor"] = batch["tile_tensor"].to(device)
        outputs = model(batch)

        logits = outputs["logits"]
        # Apply softplus to ensure positive values for beta-binomial parameters
        logits = torch.nn.functional.softplus(logits) + 1.0  # enforce concavity
        point_estimates = logits_to_point_estimates(logits)

        # sample_id = batch['sample_id'][0]
        class_assignment = point_estimates[0].item()
    return class_assignment


def logits_to_point_estimates(logits):
    # logits is a tensor of shape (batch_size, 2 * (n_clf_tasks + n_reg_tasks))
    # need to convert it to a tensor of shape (batch_size, n_clf_tasks + n_reg_tasks)
    return logits[:, ::2] / (logits[:, ::2] + logits[:, 1::2])


def run(
    features: np.ndarray,
    aeon_results: Optional[pd.DataFrame] = None,
    cancer_subtype_codes: List[str] = None,
    model_map_path: str = None,
    model_path: str = None,
    metastatic: bool = False,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    use_cpu: bool = False,
):
    """Run Paladin inference on a single slide, using the given embeddings
    and either a single model or a table mapping cancer_subtypes and targets to models.
    If cancer_subtype_codes is given, it is a list of OncoTree codes for the slide.
    If aeon_predictions_path is given, it is the pathname to a CSV file
    with the output of an Aeon run on the slide.
    If both are given, an error is raised.
    The output is written to the given output_path (a CSV file).
    """

    if aeon_results is not None:
        aeon_scores = load_aeon_scores(aeon_results)
        target_cancer_subtypes = select_cancer_subtypes(aeon_scores)
    else:
        target_cancer_subtypes = cancer_subtype_codes

    # Build a dataset to feed to the model
    site = SiteType.METASTASIS if metastatic else SiteType.PRIMARY

    dataset = TileFeatureTensorDataset(
        tile_features=features,
        site_type=site,
        n_max_tiles=20000,
    )

    device = torch.device(
        "cuda" if not use_cpu and torch.cuda.is_available() else "cpu"
    )

    results = []
    if model_path:
        cancer_subtype, target = "None", "None"
        try:
            score = run_model(device, dataset, model_path, num_workers, batch_size)
            results.append((cancer_subtype, target, score))
            logger.info(
                f"cancer_subtype: {cancer_subtype}  target: {target}  score: {score}"
            )
        except Exception as exc:
            logger.error(
                f"Unable to run model for {cancer_subtype} target {target}\n{exc}"
            )

    elif model_map_path:
        model_map = load_model_map(model_map_path)
        for cancer_subtype in target_cancer_subtypes:
            if cancer_subtype not in model_map:
                logger.warning(f"Warning: no models found for {cancer_subtype}")
                continue

            if "MSI_TYPE" in model_map[cancer_subtype]:
                # Run MSI_TYPE model first, to determine if we should run other/MSS models
                logger.info(f"Running MSI_TYPE model for {cancer_subtype} first")
                try:
                    msi_score = run_model(
                        device,
                        dataset,
                        model_map[cancer_subtype]["MSI_TYPE"],
                        num_workers,
                        batch_size,
                    )
                    results.append((cancer_subtype, "MSI_TYPE", msi_score))
                    logger.info(
                        f"cancer_subtype: {cancer_subtype}  target: MSI score: {msi_score}"
                    )
                    # If MSI score is high, skip MSS models
                    if msi_score >= 0.5:
                        logger.info(
                            f"Skipping MSS models for {cancer_subtype} due to high MSI score"
                        )
                        continue
                    else:
                        logger.info(
                            f"Running MSS models for {cancer_subtype} due to low MSI score"
                        )
                except Exception as exc:
                    logger.error(
                        f"Unable to run model for {cancer_subtype} target MSI_TYPE\n{exc}"
                    )

            for target, model in sorted(model_map[cancer_subtype].items()):
                # Skip MSI_TYPE model, already run above
                if target == "MSI_TYPE":
                    continue
                try:
                    score = run_model(device, dataset, model, num_workers, batch_size)
                    results.append((cancer_subtype, target, score))
                    logger.info(
                        f"cancer_subtype: {cancer_subtype}  target: {target}  score: {score}"
                    )
                except Exception as exc:
                    logger.error(
                        f"Unable to run model for {cancer_subtype} target {target}\n{exc}"
                    )
    df = pd.DataFrame(results, columns=["Cancer Subtype", "Biomarker", "Score"])

    return df


def parse_args():
    parser = ArgumentParser(description="Run Paladin inference on a single slide")
    parser.add_argument(
        "-i",
        "--features-path",
        required=True,
        help="Pathname to a .pt file with optimus embeddings for this slide",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="The filename for the Paladin predictions file (CSV)",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cancer-subtype-codes",
        help="One or more cancer_subtypes (OncoTree codes, comma-separated)",
    )
    parser.add_argument(
        "-a",
        "--aeon-predictions-path",
        help="Pathname to an aeon-predictions file (CSV) for this slide",
    )
    parser.add_argument(
        "-mm",
        "--model-map-path",
        help="A CSV file mapping cancer subtypes and targets to Paladin models (.pkl files). Contains columns 'cancer_subtype', 'target_name', and 'model_path'.",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        help="The filename for a Paladin model to run inference with",
    )
    parser.add_argument(
        "--metastatic", action="store_true", help="Tissue is from a metastatic site"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for data loading",
    )
    parser.add_argument("--use-cpu", action="store_true", help="Use CPU")
    opt = parser.parse_args()

    if opt.cancer_subtype_codes and opt.aeon_predictions_path:
        raise UsageError(
            "You may specify either --codes or --aeon-predictions-path, but not both."
        )

    if opt.cancer_subtype_codes:
        opt.cancer_subtype_codes = opt.cancer_subtype_codes.split(",")

    if opt.model_path is None and opt.model_map_path is None:
        raise UsageError("You must specify either --model-path or --model-map-path")

    return opt


def main():
    opt = parse_args()
    features = torch.load(opt.features_path)
    logger.info(f"Loaded features from {opt.features_path}")
    aeon_results = None
    if opt.aeon_predictions_path:
        aeon_results = pd.read_csv(opt.aeon_predictions_path)
        logger.info(f"Loaded Aeon results from {opt.aeon_predictions_path}")
    df = run(
        features=features,
        aeon_results=aeon_results,
        cancer_subtype_codes=opt.cancer_subtype_codes,
        model_map_path=opt.model_map_path,
        model_path=opt.model_path,
        metastatic=opt.metastatic,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        use_cpu=opt.use_cpu,
    )
    df.to_csv(opt.output_path, index=False)
    logger.info(f"Wrote {opt.output_path}")


if __name__ == "__main__":
    main()

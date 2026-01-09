"""Data structures and utilities for inference modules.

This module provides:
- Cancer type to integer mappings for model inputs/outputs
- SiteType enum for primary vs metastatic classification
- TileFeatureTensorDataset for feeding features to PyTorch models
"""

from enum import Enum
from typing import List

import torch
from torch.utils.data import Dataset
import numpy as np

from mosaic.data_directory import get_data_directory

CANCER_TYPE_TO_INT_MAP = {
    "AASTR": 0,
    "ACC": 1,
    "ACRM": 2,
    "ACYC": 3,
    "ADNOS": 4,
    "ALUCA": 5,
    "AMPCA": 6,
    "ANGS": 7,
    "ANSC": 8,
    "AODG": 9,
    "APAD": 10,
    "ARMM": 11,
    "ARMS": 12,
    "ASTR": 13,
    "ATM": 14,
    "BA": 15,
    "BCC": 16,
    "BLAD": 17,
    "BLCA": 18,
    "BMGCT": 19,
    "BRCA": 20,
    "BRCANOS": 21,
    "BRCNOS": 22,
    "CCOV": 23,
    "CCRCC": 24,
    "CESC": 25,
    "CHDM": 26,
    "CHOL": 27,
    "CHRCC": 28,
    "CHS": 29,
    "COAD": 30,
    "COADREAD": 31,
    "CSCC": 32,
    "CSCLC": 33,
    "CUP": 34,
    "CUPNOS": 35,
    "DA": 36,
    "DASTR": 37,
    "DDLS": 38,
    "DES": 39,
    "DIFG": 40,
    "DSRCT": 41,
    "DSTAD": 42,
    "ECAD": 43,
    "EGC": 44,
    "EHAE": 45,
    "EHCH": 46,
    "EMPD": 47,
    "EOV": 48,
    "EPDCA": 49,
    "EPIS": 50,
    "EPM": 51,
    "ERMS": 52,
    "ES": 53,
    "ESCA": 54,
    "ESCC": 55,
    "GB": 56,
    "GBAD": 57,
    "GBC": 58,
    "GBM": 59,
    "GCCAP": 60,
    "GEJ": 61,
    "GINET": 62,
    "GIST": 63,
    "GNOS": 64,
    "GRCT": 65,
    "HCC": 66,
    "HGGNOS": 67,
    "HGNEC": 68,
    "HGSFT": 69,
    "HGSOC": 70,
    "HNMUCM": 71,
    "HNSC": 72,
    "IDC": 73,
    "IHCH": 74,
    "ILC": 75,
    "LGGNOS": 76,
    "LGSOC": 77,
    "LMS": 78,
    "LNET": 79,
    "LUAD": 80,
    "LUAS": 81,
    "LUCA": 82,
    "LUNE": 83,
    "LUPC": 84,
    "LUSC": 85,
    "LXSC": 86,
    "MAAP": 87,
    "MACR": 88,
    "MBC": 89,
    "MCC": 90,
    "MDLC": 91,
    "MEL": 92,
    "MFH": 93,
    "MFS": 94,
    "MGCT": 95,
    "MNG": 96,
    "MOV": 97,
    "MPNST": 98,
    "MRLS": 99,
    "MUP": 100,
    "MXOV": 101,
    "NBL": 102,
    "NECNOS": 103,
    "NETNOS": 104,
    "NOT": 105,
    "NPC": 106,
    "NSCLC": 107,
    "NSCLCPD": 108,
    "NSGCT": 109,
    "OCS": 110,
    "OCSC": 111,
    "ODG": 112,
    "OOVC": 113,
    "OPHSC": 114,
    "OS": 115,
    "PAAC": 116,
    "PAAD": 117,
    "PAASC": 118,
    "PAMPCA": 119,
    "PANET": 120,
    "PAST": 121,
    "PDC": 122,
    "PECOMA": 123,
    "PEMESO": 124,
    "PHC": 125,
    "PLBMESO": 126,
    "PLEMESO": 127,
    "PLMESO": 128,
    "PRAD": 129,
    "PRCC": 130,
    "PSEC": 131,
    "PTAD": 132,
    "RBL": 133,
    "RCC": 134,
    "RCSNOS": 135,
    "READ": 136,
    "RMS": 137,
    "SARCNOS": 138,
    "SBC": 139,
    "SBOV": 140,
    "SBWDNET": 141,
    "SCBC": 142,
    "SCCNOS": 143,
    "SCHW": 144,
    "SCLC": 145,
    "SCUP": 146,
    "SDCA": 147,
    "SEM": 148,
    "SFT": 149,
    "SKCM": 150,
    "SOC": 151,
    "SPDAC": 152,
    "SSRCC": 153,
    "STAD": 154,
    "SYNS": 155,
    "TAC": 156,
    "THAP": 157,
    "THHC": 158,
    "THME": 159,
    "THPA": 160,
    "THPD": 161,
    "THYC": 162,
    "THYM": 163,
    "TYST": 164,
    "UCCC": 165,
    "UCEC": 166,
    "UCP": 167,
    "UCS": 168,
    "UCU": 169,
    "UDMN": 170,
    "UEC": 171,
    "ULMS": 172,
    "UM": 173,
    "UMEC": 174,
    "URCC": 175,
    "USARC": 176,
    "USC": 177,
    "UTUC": 178,
    "VMM": 179,
    "VSC": 180,
    "WDLS": 181,
    "WT": 182,
}
INT_TO_CANCER_TYPE_MAP = {v: k for k, v in CANCER_TYPE_TO_INT_MAP.items()}


# Tissue site mapping (module-level cache)
_TISSUE_SITE_MAP = None

# Default tissue site index for "Not Applicable"
DEFAULT_TISSUE_SITE_IDX = 8


def get_tissue_site_map():
    """Load tissue site name → index mapping from CSV.

    Returns:
        dict: Mapping of tissue site names to indices (0-56)

    Raises:
        FileNotFoundError: If the tissue site CSV file is not found
    """
    global _TISSUE_SITE_MAP
    if _TISSUE_SITE_MAP is None:
        import pandas as pd

        data_dir = get_data_directory()
        csv_path = data_dir / "tissue_site_original_to_idx.csv"
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tissue site mapping file not found at {csv_path}. "
                f"Please ensure the data directory contains 'tissue_site_original_to_idx.csv'."
            ) from e

        _TISSUE_SITE_MAP = {}
        for _, row in df.iterrows():
            _TISSUE_SITE_MAP[row["TISSUE_SITE"]] = int(row["idx"])

    return _TISSUE_SITE_MAP


def get_tissue_site_options():
    """Get sorted unique tissue site names for UI dropdowns.

    Returns:
        list: Sorted list of unique tissue site names
    """
    site_map = get_tissue_site_map()
    return sorted(set(site_map.keys()))


_SEX_MAP = None


def get_sex_map():
    """Get the sex to index mapping from CSV file.

    Returns:
        dict: Mapping of sex values to indices (0-2)

    Raises:
        FileNotFoundError: If the sex mapping CSV file is not found
    """
    global _SEX_MAP
    if _SEX_MAP is None:
        import pandas as pd

        data_dir = get_data_directory()
        csv_path = data_dir / "sex_original_to_idx.csv"
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Sex mapping file not found at {csv_path}. "
                f"Please ensure the data directory contains 'sex_original_to_idx.csv'."
            ) from e

        _SEX_MAP = {}
        for _, row in df.iterrows():
            _SEX_MAP[row["SEX"]] = int(row["idx"])

    return _SEX_MAP


def encode_sex(sex):
    """Convert sex to numeric encoding.

    Args:
        sex: "Male", "Female", or "Unknown" (case insensitive)

    Returns:
        int: 0 for Male, 1 for Female, 2 for Unknown
    """
    sex_map = get_sex_map()
    unknown_idx = sex_map.get("Unknown", 2)
    return sex_map.get(sex, unknown_idx)


def encode_tissue_site(site_name):
    """Convert tissue site name to index (0-56).

    Args:
        site_name: Tissue site name from CSV

    Returns:
        int: Tissue site index, defaults to DEFAULT_TISSUE_SITE_IDX ("Not Applicable")
    """
    site_map = get_tissue_site_map()
    return site_map.get(site_name, DEFAULT_TISSUE_SITE_IDX)


def tissue_site_to_one_hot(site_idx, num_classes=57):
    """Convert tissue site index to one-hot vector.

    Args:
        site_idx: Index value (0-56 for tissue site, 0-2 for sex)
        num_classes: Number of classes (57 for tissue site, 3 for sex)

    Returns:
        list: One-hot encoded vector
    """
    one_hot = [0] * num_classes
    if 0 <= site_idx < num_classes:
        one_hot[site_idx] = 1
    return one_hot


class SiteType(Enum):
    PRIMARY = "Primary"
    METASTASIS = "Metastasis"


class TileFeatureTensorDataset(Dataset):
    def __init__(
        self,
        site_type: SiteType,
        tile_features: np.ndarray,
        sex: int = None,
        tissue_site_idx: int = None,
        n_max_tiles: int = 20000,
    ) -> None:
        """Initialize the dataset.

        Args:
            site_type: the site type as str, either "Primary" or "Metastasis"
            tile_features: the tile feature array
            sex: patient sex (0=Male, 1=Female), optional for Aeon
            tissue_site_idx: tissue site index (0-56), optional for Aeon
            n_max_tiles: the maximum number of tiles to use as int

        Returns:
            None
        """
        self.site_type = site_type
        self.sex = sex
        self.tissue_site_idx = tissue_site_idx
        self.n_max_tiles = n_max_tiles
        self.features = self._get_features(tile_features)

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset
        """
        return 1

    def _get_features(self, features) -> torch.Tensor:
        """Get the tile features

        Args:
            features: the tile features as a numpy array

        Returns:
            torch.Tensor: the tile tensor
        """
        features = torch.tensor(features, dtype=torch.float32)
        if features.shape[0] > self.n_max_tiles:
            indices = torch.randperm(features.shape[0])[: self.n_max_tiles]
            features = features[indices]
        if features.shape[0] < self.n_max_tiles:
            padding = torch.zeros(
                self.n_max_tiles - features.shape[0], features.shape[1]
            )
            features = torch.cat([features, padding], dim=0)
        return features

    def __getitem__(self, idx: int) -> dict:
        """Return an item from the dataset.

        Args:
            idx: the index of the item to return

        Returns:
            dict: the item
        """
        result = {"site": self.site_type.value, "tile_tensor": self.features}

        # Add sex and tissue_site if provided (for Aeon)
        if self.sex is not None:
            result["SEX"] = torch.tensor(
                tissue_site_to_one_hot(self.sex, num_classes=3), dtype=torch.float32
            )

        if self.tissue_site_idx is not None:
            result["TISSUE_SITE"] = torch.tensor(
                tissue_site_to_one_hot(self.tissue_site_idx, num_classes=57),
                dtype=torch.float32,
            )

        return result

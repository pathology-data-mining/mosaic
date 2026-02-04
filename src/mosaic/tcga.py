"""TCGA/GDC API client for fetching whole slide images.

This module provides functionality to:
- Fetch slides from the GDC (Genomic Data Commons) by UUID or TCGA barcode
- Cache downloaded slides locally to avoid re-downloads
- Extract clinical metadata (sex, tissue site, cancer subtype) for pre-filling UI dropdowns
- Convert GDC metadata values to Mosaic-compatible values
"""

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from loguru import logger
from PIL import Image

from mosaic.inference.data import CANCER_TYPE_TO_INT_MAP
from mosaic.data_directory import get_tcga_cache_directory

# GDC API endpoints
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

# UUID pattern (36 characters with hyphens)
UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# TCGA barcode pattern (e.g., TCGA-A1-A0SB-01Z-00-DX1)
TCGA_BARCODE_PATTERN = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(-\S+)?$")

# Retry settings for network requests
# MAX_RETRIES=3: Balance between reliability and responsiveness for transient failures
# INITIAL_BACKOFF_SECONDS=1: Start with 1s, doubles on each retry (1s, 2s, 4s)
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1

# Network timeout settings (in seconds)
# API_TIMEOUT: For metadata and query requests (typically small responses)
# DOWNLOAD_TIMEOUT: For large file downloads (multi-GB slides, can take hours)
#   Set via MOSAIC_GDC_DOWNLOAD_TIMEOUT env var for very large files
API_TIMEOUT = 30
DOWNLOAD_TIMEOUT = int(os.environ.get("MOSAIC_GDC_DOWNLOAD_TIMEOUT", "300"))

# Cache for OncoTree ancestor lookups to avoid repeated API calls
_oncotree_ancestors_cache: dict[str, list[str]] = {}

# Metadata cache settings
METADATA_CACHE_FILENAME = "metadata.json"
METADATA_CACHE_VERSION = 1


@dataclass
class TCGAMetadata:
    """Metadata extracted from TCGA/GDC for a slide.

    Attributes:
        sex: Patient sex ("Male" or "Female"), or None if not available
        tissue_site: Anatomical tissue site (e.g., "Breast", "Lung"), or None
        cancer_subtype: OncoTree cancer code (e.g., "BRCA", "LUAD"), or None
        site_type: "Primary" or "Metastatic", or None
        filename: Original filename of the slide
        file_size: Size of the file in bytes
    """

    sex: str | None
    tissue_site: str | None
    cancer_subtype: str | None
    site_type: str | None
    filename: str | None = None
    file_size: int | None = None


class GDCError(Exception):
    """Base exception for GDC API errors."""

    pass


class GDCNotFoundError(GDCError):
    """Raised when a file/slide is not found in GDC."""

    pass


class GDCAccessDeniedError(GDCError):
    """Raised when access to a controlled-access file is denied."""

    pass


class GDCNetworkError(GDCError):
    """Raised when a network error occurs during GDC API calls."""

    pass


class GDCInvalidIDError(GDCError):
    """Raised when an invalid ID format is provided."""

    pass


def _get_gdc_token() -> str | None:
    """Get the GDC authentication token from environment variable.

    Returns:
        The GDC token string, or None if not set.
    """
    return os.environ.get("GDC_TOKEN")


def _normalize_cache_dir(cache_dir: Path | None) -> Path:
    """Normalize cache directory, using default if None.

    This is a helper to eliminate duplicate cache_dir normalization logic
    across multiple functions. Uses the centralized cache directory management
    from mosaic.data_directory.

    Args:
        cache_dir: Optional cache directory

    Returns:
        The provided cache_dir or the default TCGA cache directory if None
    """
    return cache_dir if cache_dir is not None else get_tcga_cache_directory()


def _make_request_with_retry(
    method: str,
    url: str,
    token: str | None = None,
    stream: bool = False,
    params: dict | None = None,
    json_data: dict | None = None,
    timeout: int | None = None,
) -> requests.Response:
    """Make an HTTP request with exponential backoff retry logic.

    Args:
        method: HTTP method ("GET" or "POST")
        url: The URL to request
        token: Optional GDC authentication token
        stream: Whether to stream the response
        params: Optional query parameters
        json_data: Optional JSON body for POST requests
        timeout: Request timeout in seconds (defaults to API_TIMEOUT for non-streaming,
                DOWNLOAD_TIMEOUT for streaming requests)

    Returns:
        The response object

    Raises:
        GDCNetworkError: If all retries fail
        GDCNotFoundError: If the resource is not found (404)
        GDCAccessDeniedError: If access is denied (401/403)
    """
    headers = {}
    if token:
        headers["X-Auth-Token"] = token

    # Use appropriate timeout based on request type
    if timeout is None:
        timeout = DOWNLOAD_TIMEOUT if stream else API_TIMEOUT

    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES):
        try:
            if method.upper() == "GET":
                response = requests.get(
                    url,
                    headers=headers,
                    stream=stream,
                    params=params,
                    timeout=timeout,
                )
            else:
                response = requests.post(
                    url,
                    headers=headers,
                    json=json_data,
                    params=params,
                    timeout=timeout,
                )

            # Handle specific HTTP errors
            if response.status_code == 404:
                raise GDCNotFoundError(f"Resource not found: {url}")
            if response.status_code in (401, 403):
                raise GDCAccessDeniedError(
                    "Access denied. This may be controlled-access data requiring a GDC token. "
                    "Set the GDC_TOKEN environment variable with your authentication token."
                )
            if response.status_code >= 500:
                # Server error, retry
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"GDC server error (status {response.status_code}), "
                        f"retrying in {backoff}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise GDCNetworkError(
                    f"GDC server error after {MAX_RETRIES} attempts: {response.status_code}"
                )

            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"Request timeout, retrying in {backoff}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise GDCNetworkError(f"Request timeout after {MAX_RETRIES} attempts")

        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"Connection error, retrying in {backoff}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise GDCNetworkError(f"Connection error after {MAX_RETRIES} attempts: {e}")

        except (GDCNotFoundError, GDCAccessDeniedError):
            # Don't retry these errors
            raise

    raise GDCNetworkError(f"Request failed after {MAX_RETRIES} attempts")


def is_valid_uuid(id_string: str) -> bool:
    """Check if a string is a valid UUID format.

    Args:
        id_string: The string to check

    Returns:
        True if the string is a valid UUID format
    """
    return bool(UUID_PATTERN.match(id_string))


def is_valid_tcga_barcode(id_string: str) -> bool:
    """Check if a string is a valid TCGA barcode format.

    Args:
        id_string: The string to check

    Returns:
        True if the string is a valid TCGA barcode format
    """
    return bool(TCGA_BARCODE_PATTERN.match(id_string.upper()))


def validate_slide_id(slide_id: str) -> str:
    """Validate and normalize a slide ID.

    Args:
        slide_id: The slide ID (UUID or TCGA barcode)

    Returns:
        The normalized slide ID

    Raises:
        GDCInvalidIDError: If the ID format is invalid
    """
    slide_id = slide_id.strip()

    if is_valid_uuid(slide_id):
        return slide_id.lower()

    if is_valid_tcga_barcode(slide_id):
        return slide_id.upper()

    raise GDCInvalidIDError(
        f"Invalid slide ID format: '{slide_id}'. "
        "Expected a UUID (e.g., 'a1b2c3d4-e5f6-7890-abcd-ef1234567890') "
        "or TCGA barcode (e.g., 'TCGA-A1-A0SB-01Z-00-DX1')."
    )


def query_slide_uuid(barcode: str, token: str | None = None) -> str:
    """Convert a TCGA barcode to a file UUID by querying GDC.

    Args:
        barcode: TCGA barcode (e.g., TCGA-A1-A0SB-01Z-00-DX1)
        token: Optional GDC authentication token

    Returns:
        The file UUID

    Raises:
        GDCNotFoundError: If no slide is found for the barcode
        GDCError: If the query fails
    """
    # Query GDC for files matching the barcode
    # The file_name field contains the barcode (e.g., TCGA-BH-A18H-01A-01-TSA.xxx.svs)
    # We use a wildcard match on file_name since submitter_id has _slide_image suffix
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {"field": "file_name", "value": f"{barcode}*"},
            },
            {
                "op": "in",
                "content": {
                    "field": "data_format",
                    "value": ["SVS", "TIFF"],
                },
            },
        ],
    }

    params = {
        "filters": str(filters).replace("'", '"'),
        "fields": "file_id,file_name,submitter_id",
        "size": 10,
    }

    response = _make_request_with_retry(
        "GET", GDC_FILES_ENDPOINT, token=token, params=params
    )
    data = response.json()

    hits = data.get("data", {}).get("hits", [])
    if not hits:
        raise GDCNotFoundError(
            f"No slide found for TCGA barcode: {barcode}. "
            "Please verify the barcode is correct and the slide is available in GDC."
        )

    # Return the first matching file UUID
    return hits[0]["file_id"]


def get_slide_metadata(file_uuid: str, token: str | None = None) -> dict:
    """Get metadata for a slide file from GDC.

    Args:
        file_uuid: The file UUID
        token: Optional GDC authentication token

    Returns:
        Dictionary containing file metadata

    Raises:
        GDCNotFoundError: If the file is not found
        GDCError: If the query fails
    """
    params = {
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "file_size",
                "data_format",
                "cases.demographic.gender",
                "cases.primary_site",
                "cases.project.project_id",
                "cases.samples.sample_type",
            ]
        ),
    }

    url = f"{GDC_FILES_ENDPOINT}/{file_uuid}"
    response = _make_request_with_retry("GET", url, token=token, params=params)
    data = response.json()

    if "data" not in data:
        raise GDCNotFoundError(f"File not found: {file_uuid}")

    return data["data"]


def _map_gdc_gender_to_sex(gender: str | None) -> str | None:
    """Map GDC gender value to Mosaic sex value.

    Args:
        gender: GDC gender value (e.g., "male", "female")

    Returns:
        Mosaic sex value ("Male" or "Female"), or None if not mappable
    """
    if not gender:
        return None

    gender_lower = gender.lower()
    if gender_lower == "male":
        return "Male"
    if gender_lower == "female":
        return "Female"

    return None


def _map_gdc_site_to_tissue_site(
    primary_site: str | None, supported_sites: list[str]
) -> str | None:
    """Map GDC primary_site to Mosaic tissue site.

    Args:
        primary_site: GDC primary_site value (e.g., "Breast", "Lung")
        supported_sites: List of supported tissue site names

    Returns:
        Matching tissue site from supported list, or "Unknown" if not found
    """
    if not primary_site:
        return "Unknown"

    # Create case-insensitive lookup
    site_lookup = {site.lower(): site for site in supported_sites}

    primary_lower = primary_site.lower()

    # Direct match
    if primary_lower in site_lookup:
        return site_lookup[primary_lower]

    # Handle special mappings (GDC uses slightly different names)
    gdc_to_mosaic = {
        "bronchus and lung": "Lung",
        "colon": "Colon",
        "kidney": "Kidney",
        "brain": "Brain",
        "breast": "Breast",
        "ovary": "Ovary",
        "prostate gland": "Prostate",
        "uterus, nos": "Uterus",
        "cervix uteri": "Cervix",
        "thyroid gland": "Thyroid",
        "pancreas": "Pancreas",
        "stomach": "Stomach",
        "esophagus": "Esophagus",
        "liver and intrahepatic bile ducts": "Liver",
        "skin": "Skin",
        "bladder": "Bladder",
        "rectum": "Rectum",
        "adrenal gland": "Adrenal Gland",
        "testis": "Testis",
    }

    if primary_lower in gdc_to_mosaic:
        mapped = gdc_to_mosaic[primary_lower]
        if mapped.lower() in site_lookup:
            return site_lookup[mapped.lower()]

    return "Unknown"


def _get_oncotree_ancestors(code: str) -> list[str]:
    """Get all ancestor codes for an OncoTree code.

    Queries the OncoTree API and traverses up the parent chain.
    Results are cached to avoid repeated API calls.

    Args:
        code: OncoTree code (e.g., "IDC", "BRCA")

    Returns:
        List of ancestor codes (from immediate parent to root), empty if not found
    """
    # Check cache first
    if code in _oncotree_ancestors_cache:
        return _oncotree_ancestors_cache[code]

    ancestors = []
    current_code = code

    # Limit depth to prevent infinite loops
    for _ in range(10):
        try:
            url = f"https://oncotree.mskcc.org/api/tumorTypes/search/code/{current_code}?exactMatch=true"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.debug(
                    f"OncoTree API returned status {response.status_code} for code '{current_code}'"
                )
                break

            data = response.json()
            if not data:
                break

            parent = data[0].get("parent")
            if not parent or parent == "TISSUE":
                break

            ancestors.append(parent)
            current_code = parent

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"OncoTree API request failed for code '{current_code}': {e}. "
                "Cancer subtype mapping may be incomplete."
            )
            break
        except Exception as e:
            logger.warning(
                f"Unexpected error querying OncoTree for code '{current_code}': {e}"
            )
            break

    # Cache the result
    _oncotree_ancestors_cache[code] = ancestors
    return ancestors


def _find_paladin_subtype_for_tcga(tcga_code: str) -> str | None:
    """Find the best matching Paladin cancer subtype for a TCGA code.

    Uses a fast direct match first, then falls back to OncoTree hierarchy
    lookup only if needed (checking TCGA's ancestors, not all Paladin codes).

    Args:
        tcga_code: TCGA project suffix (e.g., "BRCA", "LUAD")

    Returns:
        Paladin cancer subtype code if found, or None
    """
    # Direct match - TCGA code has Paladin models (most common case)
    if tcga_code in CANCER_TYPE_TO_INT_MAP:
        return tcga_code

    # Check if TCGA code is a descendant of any Paladin subtype
    # This makes only 1 API call (for tcga_code) instead of 183 (for all Paladin codes)
    tcga_ancestors = _get_oncotree_ancestors(tcga_code)
    for ancestor in tcga_ancestors:
        if ancestor in CANCER_TYPE_TO_INT_MAP:
            logger.info(
                f"Mapped TCGA '{tcga_code}' to Paladin ancestor '{ancestor}' via OncoTree hierarchy"
            )
            return ancestor

    # Note: We removed the loop checking if TCGA is an ancestor of Paladin codes,
    # as that would require 183 API calls. If a TCGA code doesn't match directly
    # or via its ancestors, we return None (user can still run analysis with "Unknown")
    return None


def _map_project_id_to_cancer_subtype(project_id: str | None) -> str | None:
    """Map TCGA project ID to OncoTree cancer subtype code.

    Maps TCGA project IDs to the cancer subtypes used in Mosaic/Paladin models.
    Uses OncoTree hierarchy to find the best matching Paladin subtype when
    there's no direct match.

    Args:
        project_id: TCGA project ID (e.g., "TCGA-BRCA", "TCGA-LUAD")

    Returns:
        OncoTree code if valid, or "Unknown" if not found
    """
    if not project_id:
        return "Unknown"

    # Extract the suffix from TCGA-XXX format
    if not project_id.startswith("TCGA-"):
        return "Unknown"

    tcga_suffix = project_id[5:]  # Remove "TCGA-" prefix

    # Try to find a matching Paladin subtype using OncoTree hierarchy
    paladin_subtype = _find_paladin_subtype_for_tcga(tcga_suffix)
    if paladin_subtype:
        return paladin_subtype

    return "Unknown"


def _map_sample_type_to_site_type(sample_type: str | None) -> str | None:
    """Map GDC sample_type to Mosaic site type.

    Args:
        sample_type: GDC sample_type value (e.g., "Primary Tumor", "Metastatic")

    Returns:
        "Primary" or "Metastatic", defaults to "Primary"
    """
    if not sample_type:
        return "Primary"

    sample_lower = sample_type.lower()

    if "metastatic" in sample_lower:
        return "Metastatic"

    # Default to Primary for primary tumor, recurrent, etc.
    return "Primary"


def convert_gdc_metadata_to_mosaic(
    gdc_metadata: dict,
    supported_tissue_sites: list[str],
) -> TCGAMetadata:
    """Convert raw GDC metadata to Mosaic-compatible values.

    Args:
        gdc_metadata: Raw metadata dict from GDC API
        supported_tissue_sites: List of supported tissue site names

    Returns:
        TCGAMetadata with converted values
    """
    # Extract nested values safely
    cases = gdc_metadata.get("cases", [{}])
    case = cases[0] if cases else {}

    demographic = case.get("demographic", {}) or {}
    gender = demographic.get("gender")

    primary_site = case.get("primary_site")

    project = case.get("project", {}) or {}
    project_id = project.get("project_id")

    samples = case.get("samples", [{}])
    sample = samples[0] if samples else {}
    sample_type = sample.get("sample_type")

    return TCGAMetadata(
        sex=_map_gdc_gender_to_sex(gender),
        tissue_site=_map_gdc_site_to_tissue_site(primary_site, supported_tissue_sites),
        cancer_subtype=_map_project_id_to_cancer_subtype(project_id),
        site_type=_map_sample_type_to_site_type(sample_type),
        filename=gdc_metadata.get("file_name"),
        file_size=gdc_metadata.get("file_size"),
    )


def save_metadata_cache(
    file_uuid: str, metadata: TCGAMetadata, cache_dir: Path | None = None
) -> None:
    """Save TCGAMetadata to cache as JSON.

    Best-effort operation that handles expected errors gracefully while
    surfacing unexpected errors. Creates a metadata.json file in the slide's
    cache directory.

    Args:
        file_uuid: The file UUID
        metadata: TCGAMetadata object to cache
        cache_dir: Optional cache directory (uses default if not provided)
    """
    try:
        cache_dir = _normalize_cache_dir(cache_dir)
        slide_dir = cache_dir / file_uuid

        if not slide_dir.exists():
            logger.debug(
                f"Slide directory does not exist for {file_uuid}, skipping metadata cache save"
            )
            return

        cache_path = slide_dir / METADATA_CACHE_FILENAME
        cache_data = {
            "cache_version": METADATA_CACHE_VERSION,
            **asdict(metadata),
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

        logger.debug(f"Saved metadata cache for {file_uuid}")

    except (PermissionError, OSError) as e:
        # Filesystem issues - warn but don't fail
        logger.warning(
            f"Cannot save metadata cache for {file_uuid}: {e}. "
            f"Cache disabled for this slide. Check permissions and disk space."
        )
    except (TypeError, KeyError, AttributeError) as e:
        # Data structure errors - these indicate bugs
        logger.exception(
            f"Failed to save metadata cache for {file_uuid} due to data error: {e}. "
            f"This may indicate a bug in metadata conversion."
        )
    except Exception as e:
        # Truly unexpected errors
        logger.exception(
            f"Unexpected error saving metadata cache for {file_uuid}: {e}"
        )


def load_metadata_cache(
    file_uuid: str, cache_dir: Path | None = None
) -> TCGAMetadata | None:
    """Load cached TCGAMetadata from JSON.

    Returns None if cache is missing, corrupt, or has wrong version.
    Caller should fall back to API fetch on None.

    Args:
        file_uuid: The file UUID
        cache_dir: Optional cache directory (uses default if not provided)

    Returns:
        TCGAMetadata if cache is valid, None otherwise
    """
    try:
        cache_dir = _normalize_cache_dir(cache_dir)
        slide_dir = cache_dir / file_uuid
        cache_path = slide_dir / METADATA_CACHE_FILENAME

        if not cache_path.exists():
            logger.debug(f"No cached metadata found for {file_uuid}")
            return None

        with open(cache_path, encoding="utf-8") as f:
            cache_data = json.load(f)

        # Check version
        if cache_data.get("cache_version") != METADATA_CACHE_VERSION:
            logger.info(
                f"Metadata cache version mismatch for {file_uuid} "
                f"(found {cache_data.get('cache_version')}, expected {METADATA_CACHE_VERSION}). "
                f"Ignoring stale cache."
            )
            return None

        # Reconstruct TCGAMetadata (remove cache_version key)
        cache_data.pop("cache_version", None)
        return TCGAMetadata(**cache_data)

    except json.JSONDecodeError as e:
        logger.warning(
            f"Corrupted metadata cache for {file_uuid}: {e}. "
            f"Falling back to API. Consider deleting: {cache_path}"
        )
        return None
    except (KeyError, TypeError) as e:
        logger.warning(
            f"Invalid metadata cache structure for {file_uuid}: {e}. "
            f"Falling back to API. Cache file may be from incompatible version."
        )
        return None
    except (PermissionError, OSError) as e:
        logger.warning(
            f"Cannot read metadata cache for {file_uuid}: {e}. "
            f"Check file permissions. Falling back to API."
        )
        return None
    except Exception as e:
        logger.exception(
            f"Unexpected error loading metadata cache for {file_uuid}: {e}"
        )
        return None


def compute_settings_hash(
    seg_config: str | dict,
    site_type: str,
    sex: str,
    tissue_site: str,
    cancer_subtype: str,
    ihc_subtype: str | None,
) -> str:
    """Compute a deterministic hash of analysis settings.

    Uses SHA-256 hash of JSON-serialized sorted settings dict,
    truncated to 12 hex characters for compact directory names.

    Args:
        seg_config: Tissue segmentation configuration. Either a string preset
            ("Biopsy", "Resection", "TCGA") or a dict with custom parameters.
            Both are JSON-serialized for consistent hashing.
        site_type: "Primary" or "Metastatic"
        sex: Patient sex
        tissue_site: Anatomical tissue site
        cancer_subtype: Cancer subtype code
        ihc_subtype: IHC subtype (optional)

    Returns:
        12-character hex string hash
    """
    settings_dict = {
        "seg_config": seg_config,
        "site_type": site_type,
        "sex": sex,
        "tissue_site": tissue_site,
        "cancer_subtype": cancer_subtype,
        "ihc_subtype": ihc_subtype,
    }

    # Sort keys for deterministic JSON
    settings_json = json.dumps(settings_dict, sort_keys=True)
    hash_obj = hashlib.sha256(settings_json.encode("utf-8"))
    return hash_obj.hexdigest()[:12]


def save_analysis_results(
    file_uuid: str,
    settings_hash: str,
    aeon_results: pd.DataFrame | None,
    paladin_results: pd.DataFrame | None,
    slide_mask: Image.Image | None,
    cache_dir: Path | None = None,
) -> None:
    """Save analysis results to cache.

    Creates results/{settings_hash}/ directory and saves:
    - aeon_results.csv (with index)
    - paladin_results.csv (no index)
    - slide_mask.png

    Best-effort operation that handles expected errors gracefully while
    surfacing unexpected errors. Skips None components.

    Args:
        file_uuid: The file UUID
        settings_hash: Hash of analysis settings
        aeon_results: Aeon cancer subtype predictions DataFrame
        paladin_results: Paladin biomarker predictions DataFrame
        slide_mask: Tissue segmentation mask PIL Image
        cache_dir: Optional cache directory (uses default if not provided)
    """
    try:
        cache_dir = _normalize_cache_dir(cache_dir)
        results_dir = cache_dir / file_uuid / "results" / settings_hash
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save aeon results (with index - "Cancer Subtype")
        if aeon_results is not None:
            aeon_path = results_dir / "aeon_results.csv"
            aeon_results.to_csv(aeon_path, index=True)

        # Save paladin results (no index)
        if paladin_results is not None:
            paladin_path = results_dir / "paladin_results.csv"
            paladin_results.to_csv(paladin_path, index=False)

        # Save slide mask
        if slide_mask is not None:
            mask_path = results_dir / "slide_mask.png"
            slide_mask.save(mask_path)

        logger.info(
            f"Successfully cached analysis results for {file_uuid}/{settings_hash}"
        )

    except OSError as e:
        # Filesystem issues - user needs to know
        if e.errno == 28:  # ENOSPC - No space left on device
            logger.error(
                f"CRITICAL: Cannot cache analysis results for {file_uuid}/{settings_hash} - "
                f"DISK FULL. Results will need to be re-analyzed (30-120s). Free up disk space."
            )
        else:
            logger.error(
                f"Filesystem error saving analysis results for {file_uuid}/{settings_hash}: {e}. "
                f"Results will need to be re-analyzed if accessed again."
            )
    except PermissionError as e:
        logger.error(
            f"Permission denied saving analysis results for {file_uuid}/{settings_hash}: {e}. "
            f"Check directory permissions: {results_dir}"
        )
    except Exception as e:
        logger.exception(
            f"Unexpected error saving analysis results for {file_uuid}/{settings_hash}: {e}. "
            f"Results will need to be re-analyzed."
        )


def load_analysis_results(
    file_uuid: str, settings_hash: str, cache_dir: Path | None = None
) -> tuple[Image.Image | None, pd.DataFrame | None, pd.DataFrame | None] | None:
    """Load cached analysis results.

    Returns tuple of (slide_mask, aeon_results, paladin_results) or None
    if any error occurs. Restores aeon_results index to "Cancer Subtype".

    Args:
        file_uuid: The file UUID
        settings_hash: Hash of analysis settings
        cache_dir: Optional cache directory (uses default if not provided)

    Returns:
        Tuple of (slide_mask, aeon_results, paladin_results) if all files
        exist and are valid, None otherwise
    """
    try:
        cache_dir = _normalize_cache_dir(cache_dir)
        results_dir = cache_dir / file_uuid / "results" / settings_hash

        if not results_dir.exists():
            logger.debug(f"No cached results for {file_uuid}/{settings_hash}")
            return None

        # Load all three files
        aeon_path = results_dir / "aeon_results.csv"
        paladin_path = results_dir / "paladin_results.csv"
        mask_path = results_dir / "slide_mask.png"

        # All must exist for a valid cache hit (atomic all-or-nothing)
        if not (aeon_path.exists() and paladin_path.exists() and mask_path.exists()):
            missing = [
                p.name for p in [aeon_path, paladin_path, mask_path] if not p.exists()
            ]
            logger.warning(
                f"Incomplete cache for {file_uuid}/{settings_hash} (missing: {missing}). "
                f"Falling back to re-analysis."
            )
            return None

        # Load aeon results (restore index)
        aeon_results = pd.read_csv(aeon_path, index_col=0)

        # Load paladin results
        paladin_results = pd.read_csv(paladin_path)

        # Load slide mask
        with Image.open(mask_path) as im:
            slide_mask = im.copy()

        logger.info(
            f"Successfully loaded cached results for {file_uuid}/{settings_hash}"
        )
        return slide_mask, aeon_results, paladin_results

    except (pd.errors.ParserError, ValueError) as e:
        logger.warning(
            f"Corrupted CSV cache for {file_uuid}/{settings_hash}: {e}. "
            f"Falling back to re-analysis. Consider deleting: {results_dir}"
        )
        return None
    except Exception as e:
        # Catch PIL errors and others
        error_type = type(e).__name__
        if "Image" in error_type or "PIL" in error_type:
            logger.warning(
                f"Corrupted image cache for {file_uuid}/{settings_hash}: {e}. "
                f"Falling back to re-analysis. Consider deleting: {mask_path}"
            )
        elif isinstance(e, (PermissionError, OSError)):
            logger.warning(
                f"Cannot read cached results for {file_uuid}/{settings_hash}: {e}. "
                f"Check file permissions. Falling back to re-analysis."
            )
        else:
            logger.error(
                f"Unexpected error loading cached results for {file_uuid}/{settings_hash}: {e}",
                exc_info=True,
            )
        return None


def upload_results_to_hf(file_uuid: str, cache_dir: Path | None = None) -> None:
    """Upload cached results for a slide to HuggingFace Dataset.

    Uploads metadata.json and all files under results/ for this UUID.
    Uses MOSAIC_TCGA_CACHE_HF_REPO env var. No-op if not set.
    Best-effort operation that logs warnings on failure.

    Args:
        file_uuid: The file UUID
        cache_dir: Optional cache directory (uses default if not provided)
    """
    # Check if HF repo is configured
    hf_repo = os.environ.get("MOSAIC_TCGA_CACHE_HF_REPO")
    if not hf_repo:
        return  # No-op if env var not set

    try:
        from huggingface_hub import HfApi

        cache_dir = _normalize_cache_dir(cache_dir)
        slide_dir = cache_dir / file_uuid

        if not slide_dir.exists():
            logger.debug(f"No slide directory to upload for {file_uuid}")
            return

        api = HfApi()

        # Upload metadata.json if it exists
        metadata_path = slide_dir / METADATA_CACHE_FILENAME
        if metadata_path.exists():
            api.upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo=f"{file_uuid}/{METADATA_CACHE_FILENAME}",
                repo_id=hf_repo,
                repo_type="dataset",
            )
            logger.debug(f"Uploaded metadata for {file_uuid} to HF")

        # Upload all results directories
        results_dir = slide_dir / "results"
        if results_dir.exists():
            for settings_hash_dir in results_dir.iterdir():
                if not settings_hash_dir.is_dir():
                    continue

                settings_hash = settings_hash_dir.name

                # Upload each file in this settings directory
                for result_file in settings_hash_dir.iterdir():
                    if result_file.is_file():
                        api.upload_file(
                            path_or_fileobj=str(result_file),
                            path_in_repo=f"{file_uuid}/results/{settings_hash}/{result_file.name}",
                            repo_id=hf_repo,
                            repo_type="dataset",
                        )

            logger.info(
                f"Uploaded analysis results for {file_uuid} to HF Dataset {hf_repo}"
            )

    except ImportError:
        logger.warning(
            "huggingface_hub package not installed. Cannot upload to HF Dataset. "
            "Install with: pip install huggingface_hub"
        )
    except Exception as e:
        # Provide specific error messages for common failure modes
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg:
            logger.error(
                f"HF Dataset upload failed for {file_uuid}: Authentication error. "
                f"Check that HF_TOKEN environment variable is set and valid. Error: {e}"
            )
        elif "403" in error_msg or "forbidden" in error_msg:
            logger.error(
                f"HF Dataset upload failed for {file_uuid}: Permission denied. "
                f"Check that you have write access to {hf_repo}. Error: {e}"
            )
        elif "429" in error_msg or "rate limit" in error_msg:
            logger.warning(
                f"HF Dataset upload rate limited for {file_uuid}. "
                f"Results cached locally but not synced to HF. Will retry on next upload. Error: {e}"
            )
        elif "timeout" in error_msg or "timed out" in error_msg:
            logger.warning(
                f"HF Dataset upload timed out for {file_uuid}. "
                f"Check network connection. Results cached locally. Error: {e}"
            )
        else:
            logger.exception(
                f"Failed to upload results for {file_uuid} to HF Dataset {hf_repo}: {e}"
            )


def download_results_from_hf(file_uuid: str, cache_dir: Path | None = None) -> bool:
    """Download cached results for a slide from HuggingFace Dataset.

    Downloads matching files from HF Dataset repo to local cache.
    Uses MOSAIC_TCGA_CACHE_HF_REPO env var. Returns False if not set.
    Best-effort operation that returns False on any error.

    Optimization: Uses path-scoped listing (list_repo_tree with path_in_repo)
    to avoid listing the entire repository, which becomes a latency/memory
    bottleneck as the dataset grows.

    Args:
        file_uuid: The file UUID
        cache_dir: Optional cache directory (uses default if not provided)

    Returns:
        True if any files were downloaded, False otherwise
    """
    # Check if HF repo is configured
    hf_repo = os.environ.get("MOSAIC_TCGA_CACHE_HF_REPO")
    if not hf_repo:
        return False  # No-op if env var not set

    try:
        from huggingface_hub import HfApi, hf_hub_download

        cache_dir = _normalize_cache_dir(cache_dir)
        slide_dir = cache_dir / file_uuid
        slide_dir.mkdir(parents=True, exist_ok=True)

        api = HfApi()

        # List only files under the specific UUID path (not the entire repo)
        # This avoids O(N) scaling as the dataset grows to thousands of slides
        try:
            repo_items = list(
                api.list_repo_tree(
                    repo_id=hf_repo,
                    path_in_repo=file_uuid,
                    recursive=True,
                    repo_type="dataset",
                )
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "unauthorized" in error_msg:
                logger.error(
                    f"HF authentication failed when listing files for {file_uuid}. "
                    f"Check HF_TOKEN. Error: {e}"
                )
            elif "404" in error_msg or "not found" in error_msg:
                # 404 could mean either repo not found OR path not found
                # Try to distinguish by checking if it's a path-specific 404
                if file_uuid in str(e):
                    logger.debug(f"No cached results found in HF for {file_uuid}")
                else:
                    logger.error(
                        f"HF Dataset repo '{hf_repo}' not found. "
                        f"Check MOSAIC_TCGA_CACHE_HF_REPO value. Error: {e}"
                    )
                return False
            else:
                logger.warning(
                    f"Failed to list files for {file_uuid} in HF repo: {e}. "
                    f"Falling back to local cache only."
                )
                return False

        # Extract file paths (filter out folders, keep only RepoFile objects)
        matching_files = []
        for item in repo_items:
            # RepoFile has a 'path' attribute, RepoFolder doesn't have file content
            if hasattr(item, "path") and not item.path.endswith("/"):
                matching_files.append(item.path)

        if not matching_files:
            logger.debug(f"No cached results found in HF for {file_uuid}")
            return False

        # Download each matching file
        downloaded_count = 0
        failed_files = []
        for repo_path in matching_files:
            try:
                # Download to temporary location first
                downloaded_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=repo_path,
                    repo_type="dataset",
                )

                # Determine local path (remove UUID prefix)
                # repo_path is like "uuid/metadata.json" or "uuid/results/hash/file.csv"
                uuid_prefix = f"{file_uuid}/"
                if repo_path.startswith(uuid_prefix):
                    relative_path = repo_path[len(uuid_prefix) :]
                else:
                    # Shouldn't happen with path_in_repo, but be defensive
                    relative_path = repo_path

                local_path = slide_dir / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy to cache directory
                import shutil

                shutil.copy2(downloaded_path, local_path)

                downloaded_count += 1

            except Exception as e:
                failed_files.append(repo_path)
                logger.warning(f"Failed to download {repo_path} from HF: {e}")
                continue

        if failed_files:
            logger.warning(
                f"Failed to download {len(failed_files)}/{len(matching_files)} "
                f"files for {file_uuid} from HF"
            )

        if downloaded_count > 0:
            logger.info(
                f"Downloaded {downloaded_count} cached files for {file_uuid} from HF Dataset"
            )
            return True

        return False

    except ImportError:
        logger.warning(
            "huggingface_hub package not installed. Cannot download from HF Dataset. "
            "Install with: pip install huggingface_hub"
        )
        return False
    except Exception as e:
        logger.exception(
            f"Unexpected error downloading results for {file_uuid} from HF: {e}"
        )
        return False


def get_case_metadata(
    file_uuid: str,
    supported_tissue_sites: list[str],
    token: str | None = None,
    cache_dir: Path | None = None,
) -> TCGAMetadata:
    """Get clinical metadata for a slide file and convert to Mosaic format.

    Checks local cache first, falls back to GDC API if not cached.

    Args:
        file_uuid: The file UUID
        supported_tissue_sites: List of supported tissue site names for validation
        token: Optional GDC authentication token
        cache_dir: Optional cache directory (uses default if not provided)

    Returns:
        TCGAMetadata with pre-filled values for UI dropdowns

    Raises:
        GDCError: If the query fails
    """
    # Try to load from cache first
    cached_metadata = load_metadata_cache(file_uuid, cache_dir)
    if cached_metadata is not None:
        logger.debug(f"Using cached metadata for {file_uuid}")
        return cached_metadata

    # Cache miss, fetch from API
    raw_metadata = get_slide_metadata(file_uuid, token)
    metadata = convert_gdc_metadata_to_mosaic(raw_metadata, supported_tissue_sites)

    # Save to cache for next time
    save_metadata_cache(file_uuid, metadata, cache_dir)

    return metadata


def is_slide_cached(file_uuid: str, cache_dir: Path | None = None) -> bool:
    """Check if a slide is already cached locally.

    Args:
        file_uuid: The file UUID
        cache_dir: Optional cache directory (uses default if not provided)

    Returns:
        True if the slide is cached
    """
    cache_dir = _normalize_cache_dir(cache_dir)

    slide_dir = cache_dir / file_uuid
    if not slide_dir.exists():
        return False

    # Check if there's at least one SVS/TIFF file in the directory
    for ext in ["*.svs", "*.tiff", "*.tif"]:
        if list(slide_dir.glob(ext)):
            return True

    return False


def get_cached_slide_path(file_uuid: str, cache_dir: Path | None = None) -> Path | None:
    """Get the path to a cached slide file.

    Args:
        file_uuid: The file UUID
        cache_dir: Optional cache directory (uses default if not provided)

    Returns:
        Path to the cached slide file, or None if not cached
    """
    cache_dir = _normalize_cache_dir(cache_dir)

    slide_dir = cache_dir / file_uuid

    if not slide_dir.exists():
        return None

    # Find the slide file
    for ext in ["*.svs", "*.tiff", "*.tif"]:
        files = list(slide_dir.glob(ext))
        if files:
            return files[0]

    return None


def fetch_slide_by_uuid(
    file_uuid: str,
    cache_dir: Path | None = None,
    token: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download a slide from GDC by UUID.

    Args:
        file_uuid: The file UUID
        cache_dir: Optional cache directory (uses default if not provided)
        token: Optional GDC authentication token (uses env var if not provided)
        progress_callback: Optional callback(bytes_downloaded, total_bytes) for progress

    Returns:
        Path to the downloaded slide file

    Raises:
        GDCError: If the download fails
    """
    cache_dir = _normalize_cache_dir(cache_dir)

    if token is None:
        token = _get_gdc_token()

    # Check if already cached
    cached_path = get_cached_slide_path(file_uuid, cache_dir)
    if cached_path:
        logger.info(f"Slide already cached: {cached_path}")
        return cached_path

    # Get metadata to determine filename
    metadata = get_slide_metadata(file_uuid, token)
    filename = metadata.get("file_name", f"{file_uuid}.svs")
    file_size = metadata.get("file_size")

    # Create cache directory
    slide_dir = cache_dir / file_uuid
    slide_dir.mkdir(parents=True, exist_ok=True)
    slide_path = slide_dir / filename

    logger.info(f"Downloading slide {file_uuid} to {slide_path}")
    if file_size:
        logger.info(f"File size: {file_size / (1024**3):.2f} GB")

    # Download the file with streaming
    url = f"{GDC_DATA_ENDPOINT}/{file_uuid}"
    response = _make_request_with_retry("GET", url, token=token, stream=True)

    # Get total size from headers if not in metadata
    if not file_size:
        file_size = int(response.headers.get("content-length", 0))

    bytes_downloaded = 0
    chunk_size = 8192  # 8KB chunks

    try:
        with open(slide_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if progress_callback:
                        if file_size:
                            # Report progress with percentage
                            progress_callback(bytes_downloaded, file_size)
                        else:
                            # Report bytes downloaded without total (file_size unknown)
                            # Some progress callbacks can handle this gracefully
                            progress_callback(bytes_downloaded, bytes_downloaded)

        logger.info(f"Download complete: {slide_path}")
        return slide_path

    except Exception as e:
        # Clean up partial download
        if slide_path.exists():
            slide_path.unlink()
        raise GDCNetworkError(f"Download failed: {e}") from e


def fetch_slide_by_barcode(
    barcode: str,
    cache_dir: Path | None = None,
    token: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download a slide from GDC by TCGA barcode.

    Args:
        barcode: TCGA barcode (e.g., TCGA-A1-A0SB-01Z-00-DX1)
        cache_dir: Optional cache directory (uses default if not provided)
        token: Optional GDC authentication token (uses env var if not provided)
        progress_callback: Optional callback(bytes_downloaded, total_bytes) for progress

    Returns:
        Path to the downloaded slide file

    Raises:
        GDCError: If the download fails
    """
    if token is None:
        token = _get_gdc_token()

    # Convert barcode to UUID
    file_uuid = query_slide_uuid(barcode, token)
    logger.info(f"Resolved barcode {barcode} to UUID {file_uuid}")

    return fetch_slide_by_uuid(file_uuid, cache_dir, token, progress_callback)


def fetch_slide(
    slide_id: str,
    cache_dir: Path | None = None,
    token: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[Path, str]:
    """Download a slide from GDC by UUID or TCGA barcode.

    This is the main entry point that automatically detects the ID format.

    Args:
        slide_id: The slide ID (UUID or TCGA barcode)
        cache_dir: Optional cache directory (uses default if not provided)
        token: Optional GDC authentication token (uses env var if not provided)
        progress_callback: Optional callback(bytes_downloaded, total_bytes) for progress

    Returns:
        Tuple of (path to downloaded slide, file UUID)

    Raises:
        GDCInvalidIDError: If the ID format is invalid
        GDCError: If the download fails
    """
    slide_id = validate_slide_id(slide_id)

    if token is None:
        token = _get_gdc_token()

    if is_valid_uuid(slide_id):
        file_uuid = slide_id
    else:
        # It's a barcode, convert to UUID
        file_uuid = query_slide_uuid(slide_id, token)
        logger.info(f"Resolved barcode {slide_id} to UUID {file_uuid}")

    slide_path = fetch_slide_by_uuid(file_uuid, cache_dir, token, progress_callback)
    return slide_path, file_uuid

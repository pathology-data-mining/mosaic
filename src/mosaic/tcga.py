"""TCGA/GDC API client for fetching whole slide images.

This module provides functionality to:
- Fetch slides from the GDC (Genomic Data Commons) by UUID or TCGA barcode
- Cache downloaded slides locally to avoid re-downloads
- Extract clinical metadata (sex, tissue site, cancer subtype) for pre-filling UI dropdowns
- Convert GDC metadata values to Mosaic-compatible values
"""

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
from loguru import logger

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

    Checks if the TCGA code or any of its OncoTree descendants/ancestors
    have Paladin models available.

    Args:
        tcga_code: TCGA project suffix (e.g., "BRCA", "LUAD")

    Returns:
        Paladin cancer subtype code if found, or None
    """
    # Direct match - TCGA code has Paladin models
    if tcga_code in CANCER_TYPE_TO_INT_MAP:
        return tcga_code

    # Check if any Paladin subtype is a descendant of the TCGA code
    # by checking if TCGA code is an ancestor of the Paladin subtype
    for paladin_code in CANCER_TYPE_TO_INT_MAP.keys():
        ancestors = _get_oncotree_ancestors(paladin_code)
        if tcga_code in ancestors:
            # Found a Paladin subtype that is a descendant of the TCGA code
            return paladin_code

    # Check if TCGA code is a descendant of any Paladin subtype
    tcga_ancestors = _get_oncotree_ancestors(tcga_code)
    for ancestor in tcga_ancestors:
        if ancestor in CANCER_TYPE_TO_INT_MAP:
            return ancestor

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


def get_case_metadata(
    file_uuid: str,
    supported_tissue_sites: list[str],
    token: str | None = None,
) -> TCGAMetadata:
    """Get clinical metadata for a slide file and convert to Mosaic format.

    Args:
        file_uuid: The file UUID
        supported_tissue_sites: List of supported tissue site names for validation
        token: Optional GDC authentication token

    Returns:
        TCGAMetadata with pre-filled values for UI dropdowns

    Raises:
        GDCError: If the query fails
    """
    raw_metadata = get_slide_metadata(file_uuid, token)
    return convert_gdc_metadata_to_mosaic(raw_metadata, supported_tissue_sites)


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

"""Tests for TCGA/GDC API client functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest

from mosaic.tcga import (
    is_valid_uuid,
    is_valid_tcga_barcode,
    validate_slide_id,
    query_slide_uuid,
    get_slide_metadata,
    get_case_metadata,
    convert_gdc_metadata_to_mosaic,
    is_slide_cached,
    get_cached_slide_path,
    fetch_slide_by_uuid,
    fetch_slide,
    TCGAMetadata,
    GDCError,
    GDCInvalidIDError,
    GDCNotFoundError,
    GDCAccessDeniedError,
    GDCNetworkError,
    _map_gdc_gender_to_sex,
    _map_gdc_site_to_tissue_site,
    _map_project_id_to_cancer_subtype,
    _map_sample_type_to_site_type,
    _find_paladin_subtype_for_tcga,
    _get_oncotree_ancestors,
    _TCGA_TO_PALADIN_MAP,
    _oncotree_ancestors_cache,
)


class TestUUIDValidation:
    """Tests for UUID validation."""

    def test_valid_uuid_lowercase(self):
        """Test that lowercase UUID is valid."""
        assert is_valid_uuid("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    def test_valid_uuid_uppercase(self):
        """Test that uppercase UUID is valid."""
        assert is_valid_uuid("A1B2C3D4-E5F6-7890-ABCD-EF1234567890")

    def test_valid_uuid_mixed_case(self):
        """Test that mixed case UUID is valid."""
        assert is_valid_uuid("A1b2C3d4-E5f6-7890-AbCd-Ef1234567890")

    def test_invalid_uuid_too_short(self):
        """Test that too short string is invalid."""
        assert not is_valid_uuid("a1b2c3d4-e5f6-7890-abcd-ef123456789")

    def test_invalid_uuid_too_long(self):
        """Test that too long string is invalid."""
        assert not is_valid_uuid("a1b2c3d4-e5f6-7890-abcd-ef12345678901")

    def test_invalid_uuid_no_hyphens(self):
        """Test that UUID without hyphens is invalid."""
        assert not is_valid_uuid("a1b2c3d4e5f67890abcdef1234567890")

    def test_invalid_uuid_wrong_format(self):
        """Test that incorrectly formatted UUID is invalid."""
        assert not is_valid_uuid("a1b2c3d4-e5f67890-abcd-ef1234567890")

    def test_invalid_uuid_empty(self):
        """Test that empty string is invalid."""
        assert not is_valid_uuid("")

    def test_invalid_uuid_random_string(self):
        """Test that random string is invalid."""
        assert not is_valid_uuid("not-a-uuid")


class TestTCGABarcodeValidation:
    """Tests for TCGA barcode validation."""

    def test_valid_barcode_full(self):
        """Test that full TCGA barcode is valid."""
        assert is_valid_tcga_barcode("TCGA-A1-A0SB-01Z-00-DX1")

    def test_valid_barcode_short(self):
        """Test that short TCGA barcode is valid."""
        assert is_valid_tcga_barcode("TCGA-A1-A0SB")

    def test_valid_barcode_lowercase(self):
        """Test that lowercase barcode is valid (gets uppercased)."""
        assert is_valid_tcga_barcode("tcga-a1-a0sb-01z-00-dx1")

    def test_invalid_barcode_wrong_prefix(self):
        """Test that non-TCGA prefix is invalid."""
        assert not is_valid_tcga_barcode("NOTC-A1-A0SB-01Z-00-DX1")

    def test_invalid_barcode_empty(self):
        """Test that empty string is invalid."""
        assert not is_valid_tcga_barcode("")

    def test_invalid_barcode_random_string(self):
        """Test that random string is invalid."""
        assert not is_valid_tcga_barcode("random-string")


class TestValidateSlideID:
    """Tests for slide ID validation and normalization."""

    def test_valid_uuid_normalized(self):
        """Test that UUID is normalized to lowercase."""
        result = validate_slide_id("A1B2C3D4-E5F6-7890-ABCD-EF1234567890")
        assert result == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def test_valid_barcode_normalized(self):
        """Test that barcode is normalized to uppercase."""
        result = validate_slide_id("tcga-a1-a0sb-01z-00-dx1")
        assert result == "TCGA-A1-A0SB-01Z-00-DX1"

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped."""
        result = validate_slide_id("  a1b2c3d4-e5f6-7890-abcd-ef1234567890  ")
        assert result == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def test_invalid_id_raises_error(self):
        """Test that invalid ID raises GDCInvalidIDError."""
        with pytest.raises(GDCInvalidIDError) as exc_info:
            validate_slide_id("invalid-id")
        assert "Invalid slide ID format" in str(exc_info.value)


class TestMetadataMapping:
    """Tests for GDC to Mosaic metadata mapping functions."""

    def test_map_gender_male(self):
        """Test mapping male gender."""
        assert _map_gdc_gender_to_sex("male") == "Male"
        assert _map_gdc_gender_to_sex("Male") == "Male"
        assert _map_gdc_gender_to_sex("MALE") == "Male"

    def test_map_gender_female(self):
        """Test mapping female gender."""
        assert _map_gdc_gender_to_sex("female") == "Female"
        assert _map_gdc_gender_to_sex("Female") == "Female"
        assert _map_gdc_gender_to_sex("FEMALE") == "Female"

    def test_map_gender_none(self):
        """Test mapping None gender."""
        assert _map_gdc_gender_to_sex(None) is None

    def test_map_gender_unknown(self):
        """Test mapping unknown gender value."""
        assert _map_gdc_gender_to_sex("unknown") is None
        assert _map_gdc_gender_to_sex("other") is None

    def test_map_tissue_site_direct_match(self):
        """Test direct tissue site matching."""
        supported = ["Lung", "Breast", "Brain", "Kidney"]
        assert _map_gdc_site_to_tissue_site("Lung", supported) == "Lung"
        assert _map_gdc_site_to_tissue_site("lung", supported) == "Lung"
        assert _map_gdc_site_to_tissue_site("LUNG", supported) == "Lung"

    def test_map_tissue_site_gdc_mapping(self):
        """Test GDC-specific tissue site mappings."""
        supported = ["Lung", "Liver", "Prostate"]
        assert _map_gdc_site_to_tissue_site("Bronchus and lung", supported) == "Lung"
        assert _map_gdc_site_to_tissue_site("Prostate gland", supported) == "Prostate"

    def test_map_tissue_site_unknown(self):
        """Test unknown tissue site falls back to Unknown."""
        supported = ["Lung", "Breast"]
        assert _map_gdc_site_to_tissue_site("Spleen", supported) == "Unknown"
        assert _map_gdc_site_to_tissue_site(None, supported) == "Unknown"

    def test_map_project_id_direct_match(self):
        """Test TCGA project IDs that directly match Paladin subtypes."""
        # These TCGA codes have direct Paladin models
        assert _map_project_id_to_cancer_subtype("TCGA-LUAD") == "LUAD"
        assert _map_project_id_to_cancer_subtype("TCGA-PRAD") == "PRAD"
        assert _map_project_id_to_cancer_subtype("TCGA-COAD") == "COAD"
        assert _map_project_id_to_cancer_subtype("TCGA-BLCA") == "BLCA"

    @patch("mosaic.tcga._get_oncotree_ancestors")
    def test_map_project_id_oncotree_hierarchy(self, mock_get_ancestors):
        """Test TCGA project IDs that map via OncoTree hierarchy.

        Mocks OncoTree API to test ancestor/descendant matching logic
        without making real network calls.
        BRCA -> IDC (IDC is a child of BRCA in OncoTree)
        """

        # Mock the OncoTree hierarchy:
        # IDC is a descendant of BRCA (IDC has BRCA as ancestor)
        def mock_ancestors(code):
            if code == "IDC":
                return ["BRCA", "TISSUE"]  # IDC's ancestors
            elif code == "BRCA":
                return ["TISSUE"]  # BRCA's ancestors
            return []

        mock_get_ancestors.side_effect = mock_ancestors

        # TCGA-BRCA should map to IDC (child in OncoTree with Paladin models)
        result = _map_project_id_to_cancer_subtype("TCGA-BRCA")
        # Should find IDC as it's a descendant of BRCA
        assert result in ["BRCA", "IDC"], f"Expected BRCA or IDC, got {result}"

    def test_map_project_id_invalid(self):
        """Test invalid project ID returns Unknown."""
        assert _map_project_id_to_cancer_subtype("TCGA-XYZINVALID") == "Unknown"
        assert _map_project_id_to_cancer_subtype("NOT-TCGA") == "Unknown"
        assert _map_project_id_to_cancer_subtype(None) == "Unknown"

    def test_map_sample_type_primary(self):
        """Test primary tumor sample type mapping."""
        assert _map_sample_type_to_site_type("Primary Tumor") == "Primary"
        assert _map_sample_type_to_site_type("Recurrent Tumor") == "Primary"
        assert _map_sample_type_to_site_type("Solid Tissue Normal") == "Primary"

    def test_map_sample_type_metastatic(self):
        """Test metastatic sample type mapping."""
        assert _map_sample_type_to_site_type("Metastatic") == "Metastatic"
        assert _map_sample_type_to_site_type("Additional Metastatic") == "Metastatic"

    def test_map_sample_type_default(self):
        """Test default sample type mapping."""
        assert _map_sample_type_to_site_type(None) == "Primary"
        assert _map_sample_type_to_site_type("") == "Primary"


class TestOncoTreeMapping:
    """Tests for OncoTree hierarchy and cancer subtype mapping."""

    def test_precomputed_tcga_map_coverage(self):
        """Test that the precomputed TCGA map has expected common codes."""
        # Verify some of the most common TCGA codes are in the precomputed map
        common_codes = ["LUAD", "BRCA", "PRAD", "COAD", "BLCA", "LUSC", "HNSC", "STAD"]
        for code in common_codes:
            assert code in _TCGA_TO_PALADIN_MAP, f"{code} should be in precomputed map"

    def test_find_paladin_subtype_direct_match(self):
        """Test _find_paladin_subtype_for_tcga with direct match in CANCER_TYPE_TO_INT_MAP."""
        # LUAD is in CANCER_TYPE_TO_INT_MAP and precomputed map
        result = _find_paladin_subtype_for_tcga("LUAD")
        assert result == "LUAD"

        # BRCA is in CANCER_TYPE_TO_INT_MAP and precomputed map
        result = _find_paladin_subtype_for_tcga("BRCA")
        assert result == "BRCA"

    @patch("mosaic.tcga._get_oncotree_ancestors")
    def test_find_paladin_subtype_via_ancestors(self, mock_get_ancestors):
        """Test _find_paladin_subtype_for_tcga using OncoTree ancestor lookup.

        Tests that if TCGA code is not in CANCER_TYPE_TO_INT_MAP,
        it checks ancestors and returns the first matching Paladin subtype.
        """
        # Mock a TCGA code "TESTCODE" that has "LUAD" as an ancestor
        mock_get_ancestors.return_value = ["LUAD", "NSCLC", "TISSUE"]

        result = _find_paladin_subtype_for_tcga("TESTCODE")
        assert result == "LUAD"
        mock_get_ancestors.assert_called_once_with("TESTCODE")

    @patch("mosaic.tcga._get_oncotree_ancestors")
    def test_find_paladin_subtype_no_match(self, mock_get_ancestors):
        """Test _find_paladin_subtype_for_tcga with no matching Paladin subtype."""
        # Mock a TCGA code with no Paladin ancestors
        mock_get_ancestors.return_value = ["UNKNOWN_PARENT", "TISSUE"]

        result = _find_paladin_subtype_for_tcga("UNKNOWN_CODE")
        assert result is None

    @patch("mosaic.tcga.requests.get")
    def test_get_oncotree_ancestors_success(self, mock_requests_get):
        """Test _get_oncotree_ancestors with successful API calls."""
        # Clear cache for this test
        _oncotree_ancestors_cache.clear()

        # Mock the OncoTree API responses
        # IDC -> BRCA -> TISSUE
        mock_responses = [
            # First call for "IDC"
            Mock(
                status_code=200,
                json=lambda: [{"code": "IDC", "parent": "BRCA"}]
            ),
            # Second call for "BRCA"
            Mock(
                status_code=200,
                json=lambda: [{"code": "BRCA", "parent": "TISSUE"}]
            ),
        ]
        mock_requests_get.side_effect = mock_responses

        result = _get_oncotree_ancestors("IDC")
        assert result == ["BRCA"]  # Only returns ancestors, not TISSUE

        # Verify caching
        assert "IDC" in _oncotree_ancestors_cache
        assert _oncotree_ancestors_cache["IDC"] == ["BRCA"]

    @patch("mosaic.tcga.requests.get")
    def test_get_oncotree_ancestors_api_failure(self, mock_requests_get):
        """Test _get_oncotree_ancestors handles API failures gracefully."""
        # Clear cache for this test
        _oncotree_ancestors_cache.clear()

        # Mock API failure
        mock_requests_get.side_effect = Exception("API error")

        result = _get_oncotree_ancestors("TESTCODE")
        assert result == []  # Should return empty list on error

    @patch("mosaic.tcga.requests.get")
    def test_get_oncotree_ancestors_caching(self, mock_requests_get):
        """Test _get_oncotree_ancestors uses cache to avoid repeated API calls."""
        # Clear cache for this test
        _oncotree_ancestors_cache.clear()

        # Mock response - simulate a single-level hierarchy (IDC -> TISSUE)
        mock_requests_get.return_value = Mock(
            status_code=200,
            json=lambda: [{"code": "IDC", "parent": "TISSUE"}]
        )

        # First call should hit the API
        result1 = _get_oncotree_ancestors("IDC")
        initial_call_count = mock_requests_get.call_count
        assert initial_call_count >= 1  # At least one API call

        # Second call should use cache (no additional API calls)
        result2 = _get_oncotree_ancestors("IDC")
        assert mock_requests_get.call_count == initial_call_count  # No additional calls
        assert result1 == result2

    @patch("mosaic.tcga.requests.get")
    def test_get_oncotree_ancestors_not_found(self, mock_requests_get):
        """Test _get_oncotree_ancestors handles 404 responses."""
        # Clear cache for this test
        _oncotree_ancestors_cache.clear()

        mock_requests_get.return_value = Mock(status_code=404)

        result = _get_oncotree_ancestors("INVALID_CODE")
        assert result == []


class TestConvertGDCMetadata:
    """Tests for full GDC metadata conversion."""

    def test_convert_complete_metadata(self):
        """Test converting complete GDC metadata.

        Note: cancer_subtype mapping uses OncoTree hierarchy to find
        the best matching Paladin subtype. BRCA may map to IDC.
        """
        gdc_metadata = {
            "file_name": "test_slide.svs",
            "file_size": 1000000,
            "cases": [
                {
                    "demographic": {"gender": "female"},
                    "primary_site": "Breast",
                    "project": {
                        "project_id": "TCGA-LUAD"
                    },  # Use LUAD for predictable result
                    "samples": [{"sample_type": "Primary Tumor"}],
                }
            ],
        }
        supported_sites = ["Breast", "Lung", "Brain"]

        result = convert_gdc_metadata_to_mosaic(gdc_metadata, supported_sites)

        assert result.sex == "Female"
        assert result.tissue_site == "Breast"
        assert result.cancer_subtype == "LUAD"  # LUAD has direct Paladin models
        assert result.site_type == "Primary"
        assert result.filename == "test_slide.svs"
        assert result.file_size == 1000000

    def test_convert_empty_metadata(self):
        """Test converting empty GDC metadata."""
        gdc_metadata = {}
        supported_sites = ["Breast", "Lung"]

        result = convert_gdc_metadata_to_mosaic(gdc_metadata, supported_sites)

        assert result.sex is None
        assert result.tissue_site == "Unknown"
        assert result.cancer_subtype == "Unknown"
        assert result.site_type == "Primary"

    def test_convert_partial_metadata(self):
        """Test converting partial GDC metadata."""
        gdc_metadata = {
            "cases": [
                {
                    "demographic": {"gender": "male"},
                    # Missing primary_site, project, samples
                }
            ],
        }
        supported_sites = ["Breast", "Lung"]

        result = convert_gdc_metadata_to_mosaic(gdc_metadata, supported_sites)

        assert result.sex == "Male"
        assert result.tissue_site == "Unknown"
        assert result.cancer_subtype == "Unknown"
        assert result.site_type == "Primary"


class TestCaching:
    """Tests for slide caching functionality."""

    def test_is_slide_cached_not_cached(self):
        """Test is_slide_cached returns False when not cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            assert not is_slide_cached("test-uuid", cache_dir)

    def test_is_slide_cached_cached(self):
        """Test is_slide_cached returns True when cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            slide_dir = cache_dir / "test-uuid"
            slide_dir.mkdir(parents=True)
            (slide_dir / "test.svs").touch()

            assert is_slide_cached("test-uuid", cache_dir)

    def test_get_cached_slide_path_not_cached(self):
        """Test get_cached_slide_path returns None when not cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            assert get_cached_slide_path("test-uuid", cache_dir) is None

    def test_get_cached_slide_path_cached(self):
        """Test get_cached_slide_path returns path when cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            slide_dir = cache_dir / "test-uuid"
            slide_dir.mkdir(parents=True)
            slide_path = slide_dir / "test.svs"
            slide_path.touch()

            result = get_cached_slide_path("test-uuid", cache_dir)
            assert result == slide_path


class TestGDCAPIIntegration:
    """Tests for GDC API integration (mocked)."""

    @patch("mosaic.tcga._make_request_with_retry")
    def test_query_slide_uuid_success(self, mock_request):
        """Test successful barcode to UUID conversion."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"hits": [{"file_id": "test-uuid-12345", "file_name": "slide.svs"}]}
        }
        mock_request.return_value = mock_response

        result = query_slide_uuid("TCGA-A1-A0SB-01Z-00-DX1")
        assert result == "test-uuid-12345"

    @patch("mosaic.tcga._make_request_with_retry")
    def test_query_slide_uuid_not_found(self, mock_request):
        """Test barcode not found raises error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"hits": []}}
        mock_request.return_value = mock_response

        with pytest.raises(GDCNotFoundError):
            query_slide_uuid("TCGA-XX-XXXX")

    @patch("mosaic.tcga._make_request_with_retry")
    def test_get_slide_metadata_success(self, mock_request):
        """Test successful metadata retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "file_id": "test-uuid",
                "file_name": "slide.svs",
                "file_size": 1000000,
                "cases": [
                    {
                        "demographic": {"gender": "female"},
                        "primary_site": "Breast",
                    }
                ],
            }
        }
        mock_request.return_value = mock_response

        result = get_slide_metadata("test-uuid")
        assert result["file_name"] == "slide.svs"
        assert result["file_size"] == 1000000

    @patch("mosaic.tcga._make_request_with_retry")
    def test_get_slide_metadata_not_found(self, mock_request):
        """Test metadata not found raises error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_request.return_value = mock_response

        with pytest.raises(GDCNotFoundError):
            get_slide_metadata("nonexistent-uuid")

    @patch("mosaic.tcga.get_slide_metadata")
    @patch("mosaic.tcga.get_cached_slide_path")
    def test_fetch_slide_by_uuid_cached(self, mock_cached, mock_metadata):
        """Test fetching a cached slide returns cached path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cached_path = cache_dir / "test-uuid" / "slide.svs"
            cached_path.parent.mkdir(parents=True)
            cached_path.touch()

            mock_cached.return_value = cached_path

            result = fetch_slide_by_uuid("test-uuid", cache_dir)
            assert result == cached_path
            mock_metadata.assert_not_called()

    @patch("mosaic.tcga._make_request_with_retry")
    @patch("mosaic.tcga.get_slide_metadata")
    @patch("mosaic.tcga.get_cached_slide_path")
    def test_fetch_slide_by_uuid_download(
        self, mock_cached, mock_metadata, mock_request
    ):
        """Test fetching an uncached slide downloads it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            mock_cached.return_value = None
            mock_metadata.return_value = {
                "file_name": "slide.svs",
                "file_size": 100,
            }

            # Mock the download response
            mock_response = MagicMock()
            mock_response.headers = {"content-length": "100"}
            mock_response.iter_content.return_value = [b"test content"]
            mock_request.return_value = mock_response

            result = fetch_slide_by_uuid("test-uuid", cache_dir)

            assert result.name == "slide.svs"
            assert result.exists()

    @patch("mosaic.tcga.fetch_slide_by_uuid")
    def test_fetch_slide_with_uuid(self, mock_fetch):
        """Test fetch_slide with UUID input."""
        mock_fetch.return_value = Path("/cache/test-uuid/slide.svs")

        path, uuid = fetch_slide("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

        assert uuid == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        mock_fetch.assert_called_once()

    @patch("mosaic.tcga.query_slide_uuid")
    @patch("mosaic.tcga.fetch_slide_by_uuid")
    def test_fetch_slide_with_barcode(self, mock_fetch, mock_query):
        """Test fetch_slide with barcode input."""
        mock_query.return_value = "resolved-uuid"
        mock_fetch.return_value = Path("/cache/resolved-uuid/slide.svs")

        path, uuid = fetch_slide("TCGA-A1-A0SB-01Z-00-DX1")

        assert uuid == "resolved-uuid"
        mock_query.assert_called_once_with("TCGA-A1-A0SB-01Z-00-DX1", None)


class TestTCGAMetadataDataclass:
    """Tests for TCGAMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating TCGAMetadata instance."""
        metadata = TCGAMetadata(
            sex="Female",
            tissue_site="Breast",
            cancer_subtype="BRCA",
            site_type="Primary",
            filename="slide.svs",
            file_size=1000000,
        )

        assert metadata.sex == "Female"
        assert metadata.tissue_site == "Breast"
        assert metadata.cancer_subtype == "BRCA"
        assert metadata.site_type == "Primary"
        assert metadata.filename == "slide.svs"
        assert metadata.file_size == 1000000

    def test_create_metadata_defaults(self):
        """Test TCGAMetadata with default values."""
        metadata = TCGAMetadata(
            sex=None,
            tissue_site=None,
            cancer_subtype=None,
            site_type=None,
        )

        assert metadata.sex is None
        assert metadata.filename is None
        assert metadata.file_size is None


class TestErrorClasses:
    """Tests for custom exception classes."""

    def test_gdc_error_base(self):
        """Test GDCError base class."""
        error = GDCError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_gdc_not_found_error(self):
        """Test GDCNotFoundError."""
        error = GDCNotFoundError("File not found")
        assert isinstance(error, GDCError)

    def test_gdc_access_denied_error(self):
        """Test GDCAccessDeniedError."""
        error = GDCAccessDeniedError("Access denied")
        assert isinstance(error, GDCError)

    def test_gdc_network_error(self):
        """Test GDCNetworkError."""
        error = GDCNetworkError("Network error")
        assert isinstance(error, GDCError)

    def test_gdc_invalid_id_error(self):
        """Test GDCInvalidIDError."""
        error = GDCInvalidIDError("Invalid ID")
        assert isinstance(error, GDCError)


# Optional: Integration test with real GDC API (marked as slow)
@pytest.mark.skip(reason="Integration test - requires network access")
class TestGDCAPIRealIntegration:
    """Real integration tests with GDC API (skipped by default)."""

    def test_fetch_public_slide_metadata(self):
        """Test fetching metadata for a known public TCGA slide."""
        # This would use a known public slide UUID
        # Example: TCGA-A1-A0SB slide
        pass

    def test_query_known_barcode(self):
        """Test querying a known TCGA barcode."""
        pass

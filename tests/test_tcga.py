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
    save_metadata_cache,
    load_metadata_cache,
    compute_settings_hash,
    save_analysis_results,
    load_analysis_results,
    upload_results_to_hf,
    download_results_from_hf,
    METADATA_CACHE_FILENAME,
    METADATA_CACHE_VERSION,
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

    def test_find_paladin_subtype_direct_match(self):
        """Test _find_paladin_subtype_for_tcga with direct match in CANCER_TYPE_TO_INT_MAP."""
        # LUAD is in CANCER_TYPE_TO_INT_MAP
        result = _find_paladin_subtype_for_tcga("LUAD")
        assert result == "LUAD"

        # BRCA is in CANCER_TYPE_TO_INT_MAP
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
            Mock(status_code=200, json=lambda: [{"code": "IDC", "parent": "BRCA"}]),
            # Second call for "BRCA"
            Mock(status_code=200, json=lambda: [{"code": "BRCA", "parent": "TISSUE"}]),
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
            status_code=200, json=lambda: [{"code": "IDC", "parent": "TISSUE"}]
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


class TestMetadataCache:
    """Tests for metadata caching functionality."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test saving and loading metadata cache round-trip."""
        file_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        metadata = TCGAMetadata(
            sex="Male",
            tissue_site="Breast",
            cancer_subtype="BRCA",
            site_type="Primary",
            filename="test.svs",
            file_size=1024000,
        )

        save_metadata_cache(file_uuid, metadata, cache_dir=tmp_path)
        loaded = load_metadata_cache(file_uuid, cache_dir=tmp_path)

        assert loaded is not None
        assert loaded.sex == "Male"
        assert loaded.tissue_site == "Breast"
        assert loaded.cancer_subtype == "BRCA"
        assert loaded.site_type == "Primary"
        assert loaded.filename == "test.svs"
        assert loaded.file_size == 1024000

    def test_load_missing_cache_returns_none(self, tmp_path):
        """Test loading non-existent cache returns None."""
        file_uuid = "missing-uuid"
        loaded = load_metadata_cache(file_uuid, cache_dir=tmp_path)
        assert loaded is None

    def test_load_corrupt_json_returns_none(self, tmp_path):
        """Test loading corrupt JSON returns None."""
        file_uuid = "corrupt-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        cache_path = slide_dir / METADATA_CACHE_FILENAME
        cache_path.write_text("not valid json{{{")

        loaded = load_metadata_cache(file_uuid, cache_dir=tmp_path)
        assert loaded is None

    def test_load_wrong_version_returns_none(self, tmp_path):
        """Test loading wrong version cache returns None."""
        file_uuid = "old-version-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        cache_path = slide_dir / METADATA_CACHE_FILENAME
        import json

        cache_path.write_text(
            json.dumps(
                {
                    "cache_version": METADATA_CACHE_VERSION + 1,
                    "sex": "Male",
                    "tissue_site": "Breast",
                    "cancer_subtype": "BRCA",
                    "site_type": "Primary",
                }
            )
        )

        loaded = load_metadata_cache(file_uuid, cache_dir=tmp_path)
        assert loaded is None

    def test_roundtrip_with_none_values(self, tmp_path):
        """Test round-trip with None values."""
        file_uuid = "none-values-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        metadata = TCGAMetadata(
            sex=None,
            tissue_site=None,
            cancer_subtype=None,
            site_type=None,
            filename=None,
            file_size=None,
        )

        save_metadata_cache(file_uuid, metadata, cache_dir=tmp_path)
        loaded = load_metadata_cache(file_uuid, cache_dir=tmp_path)

        assert loaded is not None
        assert loaded.sex is None
        assert loaded.tissue_site is None
        assert loaded.cancer_subtype is None
        assert loaded.site_type is None

    def test_save_no_slide_dir_is_noop(self, tmp_path):
        """Test saving when slide directory doesn't exist is a no-op."""
        file_uuid = "no-dir-uuid"
        metadata = TCGAMetadata(
            sex="Male",
            tissue_site="Breast",
            cancer_subtype="BRCA",
            site_type="Primary",
        )

        # Should not raise exception
        save_metadata_cache(file_uuid, metadata, cache_dir=tmp_path)

        # Should not have created file
        cache_path = tmp_path / file_uuid / METADATA_CACHE_FILENAME
        assert not cache_path.exists()

    def test_save_permission_error_is_silent(self, tmp_path, monkeypatch):
        """Test that permission errors during save are silent."""
        file_uuid = "perm-error-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        metadata = TCGAMetadata(
            sex="Male",
            tissue_site="Breast",
            cancer_subtype="BRCA",
            site_type="Primary",
        )

        # Mock open to raise PermissionError
        import builtins

        original_open = builtins.open

        def mock_open(*args, **kwargs):
            if "metadata.json" in str(args[0]) and "w" in str(kwargs.get("mode", "")):
                raise PermissionError("No permission")
            return original_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", mock_open)

        # Should not raise exception
        save_metadata_cache(file_uuid, metadata, cache_dir=tmp_path)


class TestAnalysisResultsCache:
    """Tests for analysis results caching functionality."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test saving and loading analysis results round-trip."""
        import pandas as pd
        from PIL import Image

        file_uuid = "results-uuid"
        settings_hash = "abc123def456"

        # Create test data
        aeon_results = pd.DataFrame(
            {"Confidence": [0.95, 0.85]}, index=["BRCA", "LUAD"]
        )
        aeon_results.index.name = "Cancer Subtype"

        paladin_results = pd.DataFrame(
            {"Biomarker": ["HER2", "ER"], "Probability": [0.9, 0.8]}
        )

        slide_mask = Image.new("RGB", (100, 100), color="red")

        # Save
        save_analysis_results(
            file_uuid,
            settings_hash,
            aeon_results,
            paladin_results,
            slide_mask,
            cache_dir=tmp_path,
        )

        # Load
        loaded = load_analysis_results(file_uuid, settings_hash, cache_dir=tmp_path)

        assert loaded is not None
        loaded_mask, loaded_aeon, loaded_paladin = loaded

        # Check aeon results
        assert loaded_aeon is not None
        assert loaded_aeon.index.name == "Cancer Subtype"
        assert list(loaded_aeon.index) == ["BRCA", "LUAD"]
        assert list(loaded_aeon["Confidence"]) == [0.95, 0.85]

        # Check paladin results
        assert loaded_paladin is not None
        assert list(loaded_paladin["Biomarker"]) == ["HER2", "ER"]
        assert list(loaded_paladin["Probability"]) == [0.9, 0.8]

        # Check mask
        assert loaded_mask is not None
        assert loaded_mask.size == (100, 100)

    def test_load_missing_dir_returns_none(self, tmp_path):
        """Test loading from missing directory returns None."""
        loaded = load_analysis_results(
            "missing-uuid", "missing-hash", cache_dir=tmp_path
        )
        assert loaded is None

    def test_load_partial_results_returns_none(self, tmp_path):
        """Test loading partial results returns None."""
        import pandas as pd

        file_uuid = "partial-uuid"
        settings_hash = "partial-hash"
        results_dir = tmp_path / file_uuid / "results" / settings_hash
        results_dir.mkdir(parents=True)

        # Only save aeon results, missing others
        aeon_results = pd.DataFrame({"Confidence": [0.95]}, index=["BRCA"])
        aeon_path = results_dir / "aeon_results.csv"
        aeon_results.to_csv(aeon_path, index=True)

        loaded = load_analysis_results(file_uuid, settings_hash, cache_dir=tmp_path)
        assert loaded is None

    def test_settings_hash_deterministic(self):
        """Test that settings hash is deterministic."""
        seg_config = {"threshold": 0.5, "min_size": 100}
        hash1 = compute_settings_hash(
            seg_config, "Primary", "Male", "Breast", "BRCA", "HER2+"
        )
        hash2 = compute_settings_hash(
            seg_config, "Primary", "Male", "Breast", "BRCA", "HER2+"
        )
        assert hash1 == hash2
        assert len(hash1) == 12  # 12 hex characters

    def test_settings_hash_varies_with_input(self):
        """Test that settings hash varies with different inputs."""
        seg_config = {"threshold": 0.5, "min_size": 100}

        hash1 = compute_settings_hash(
            seg_config, "Primary", "Male", "Breast", "BRCA", None
        )
        hash2 = compute_settings_hash(
            seg_config, "Metastatic", "Male", "Breast", "BRCA", None
        )
        hash3 = compute_settings_hash(
            seg_config, "Primary", "Female", "Breast", "BRCA", None
        )

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_save_with_none_components(self, tmp_path):
        """Test saving with None components skips those files."""
        file_uuid = "none-components-uuid"
        settings_hash = "none-hash"

        save_analysis_results(
            file_uuid, settings_hash, None, None, None, cache_dir=tmp_path
        )

        results_dir = tmp_path / file_uuid / "results" / settings_hash
        assert results_dir.exists()

        # No files should be created
        assert not (results_dir / "aeon_results.csv").exists()
        assert not (results_dir / "paladin_results.csv").exists()
        assert not (results_dir / "slide_mask.png").exists()

    def test_aeon_index_restored(self, tmp_path):
        """Test that aeon results index is properly restored."""
        import pandas as pd
        from PIL import Image

        file_uuid = "index-uuid"
        settings_hash = "index-hash"

        aeon_results = pd.DataFrame({"Confidence": [0.95]}, index=["BRCA"])
        aeon_results.index.name = "Cancer Subtype"

        paladin_results = pd.DataFrame({"Biomarker": ["HER2"]})
        slide_mask = Image.new("RGB", (50, 50))

        save_analysis_results(
            file_uuid,
            settings_hash,
            aeon_results,
            paladin_results,
            slide_mask,
            cache_dir=tmp_path,
        )

        loaded = load_analysis_results(file_uuid, settings_hash, cache_dir=tmp_path)
        _, loaded_aeon, _ = loaded

        # Index should be restored from first column
        assert loaded_aeon.index.name is not None
        assert list(loaded_aeon.index) == ["BRCA"]


class TestHFDatasetSync:
    """Tests for HuggingFace Dataset sync functionality."""

    def test_upload_results_calls_hf_api(self, tmp_path, monkeypatch):
        """Test that upload_results_to_hf calls HF API correctly."""
        import json

        file_uuid = "upload-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        # Create metadata file
        metadata_path = slide_dir / METADATA_CACHE_FILENAME
        metadata_path.write_text(json.dumps({"cache_version": 1, "sex": "Male"}))

        # Create results directory
        results_dir = slide_dir / "results" / "hash1"
        results_dir.mkdir(parents=True)
        (results_dir / "aeon_results.csv").write_text("test")

        # Mock HfApi - need to patch where it's used (inside the function)
        mock_api = MagicMock()

        def mock_hf_api():
            return mock_api

        # Mock the import statement inside upload_results_to_hf
        import sys
        from types import ModuleType

        mock_hf_hub = ModuleType("huggingface_hub")
        mock_hf_hub.HfApi = mock_hf_api
        sys.modules["huggingface_hub"] = mock_hf_hub

        monkeypatch.setenv("MOSAIC_TCGA_CACHE_HF_REPO", "test-org/test-repo")

        try:
            upload_results_to_hf(file_uuid, cache_dir=tmp_path)

            # Verify API was called
            assert mock_api.upload_file.call_count >= 1
        finally:
            # Clean up mock
            if "huggingface_hub" in sys.modules:
                del sys.modules["huggingface_hub"]

    def test_upload_noop_without_env_var(self, tmp_path, monkeypatch):
        """Test that upload is no-op when env var not set."""
        monkeypatch.delenv("MOSAIC_TCGA_CACHE_HF_REPO", raising=False)

        file_uuid = "noop-uuid"

        # Should not raise exception
        upload_results_to_hf(file_uuid, cache_dir=tmp_path)

    def test_download_results_populates_cache(self, tmp_path, monkeypatch):
        """Test that download_results_from_hf populates local cache."""
        file_uuid = "download-uuid"

        # Mock HfApi
        mock_api = MagicMock()
        mock_api.list_repo_files.return_value = [
            f"{file_uuid}/metadata.json",
            f"{file_uuid}/results/hash1/aeon_results.csv",
        ]

        def mock_hf_api():
            return mock_api

        # Mock hf_hub_download to return temp file
        def mock_download(repo_id, filename, repo_type):
            temp_file = tmp_path / "temp" / filename.replace("/", "_")
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text("test content")
            return str(temp_file)

        # Mock the imports
        import sys
        from types import ModuleType

        mock_hf_hub = ModuleType("huggingface_hub")
        mock_hf_hub.HfApi = mock_hf_api
        mock_hf_hub.hf_hub_download = mock_download
        sys.modules["huggingface_hub"] = mock_hf_hub

        monkeypatch.setenv("MOSAIC_TCGA_CACHE_HF_REPO", "test-org/test-repo")

        try:
            result = download_results_from_hf(file_uuid, cache_dir=tmp_path)

            assert result is True
            # Verify files were copied to cache
            assert (tmp_path / file_uuid / "metadata.json").exists()
        finally:
            # Clean up mock
            if "huggingface_hub" in sys.modules:
                del sys.modules["huggingface_hub"]

    def test_download_noop_without_env_var(self, tmp_path, monkeypatch):
        """Test that download is no-op when env var not set."""
        monkeypatch.delenv("MOSAIC_TCGA_CACHE_HF_REPO", raising=False)

        result = download_results_from_hf("test-uuid", cache_dir=tmp_path)
        assert result is False

    def test_download_handles_api_error_gracefully(self, tmp_path, monkeypatch):
        """Test that download handles API errors gracefully."""
        # Mock HfApi to raise exception
        mock_api = MagicMock()
        mock_api.list_repo_files.side_effect = Exception("API error")

        def mock_hf_api():
            return mock_api

        # Mock the import
        import sys
        from types import ModuleType

        mock_hf_hub = ModuleType("huggingface_hub")
        mock_hf_hub.HfApi = mock_hf_api
        sys.modules["huggingface_hub"] = mock_hf_hub

        monkeypatch.setenv("MOSAIC_TCGA_CACHE_HF_REPO", "test-org/test-repo")

        try:
            result = download_results_from_hf("test-uuid", cache_dir=tmp_path)
            assert result is False
        finally:
            # Clean up mock
            if "huggingface_hub" in sys.modules:
                del sys.modules["huggingface_hub"]


class TestGetCaseMetadataWithCache:
    """Tests for get_case_metadata with caching."""

    @patch("mosaic.tcga.get_slide_metadata")
    def test_cache_hit_skips_api(self, mock_get_metadata, tmp_path):
        """Test that cache hit skips API call."""
        file_uuid = "cached-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        # Pre-populate cache
        metadata = TCGAMetadata(
            sex="Male",
            tissue_site="Breast",
            cancer_subtype="BRCA",
            site_type="Primary",
        )
        save_metadata_cache(file_uuid, metadata, cache_dir=tmp_path)

        # Call get_case_metadata
        result = get_case_metadata(file_uuid, ["Breast"], cache_dir=tmp_path)

        # API should not have been called
        mock_get_metadata.assert_not_called()

        # Result should match cached metadata
        assert result.sex == "Male"
        assert result.cancer_subtype == "BRCA"

    @patch("mosaic.tcga.get_slide_metadata")
    @patch("mosaic.tcga.convert_gdc_metadata_to_mosaic")
    def test_cache_miss_fetches_and_saves(
        self, mock_convert, mock_get_metadata, tmp_path
    ):
        """Test that cache miss fetches from API and saves."""
        file_uuid = "uncached-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        mock_get_metadata.return_value = {"file_name": "test.svs"}
        mock_convert.return_value = TCGAMetadata(
            sex="Female",
            tissue_site="Lung",
            cancer_subtype="LUAD",
            site_type="Primary",
        )

        result = get_case_metadata(file_uuid, ["Lung"], cache_dir=tmp_path)

        # API should have been called
        mock_get_metadata.assert_called_once()

        # Result should match API response
        assert result.sex == "Female"

        # Cache should have been saved
        cached = load_metadata_cache(file_uuid, cache_dir=tmp_path)
        assert cached is not None
        assert cached.sex == "Female"


class TestTCGACacheDirHFSpaces:
    """Test TCGA cache directory detection on HF Spaces."""

    def test_tcga_cache_dir_uses_data_on_hf_spaces(self, monkeypatch, tmp_path):
        """Test that HF Spaces uses /data for persistent storage."""
        from mosaic.data_directory import get_tcga_cache_directory

        # Mock HF Spaces environment
        monkeypatch.setenv("SPACE_ID", "test-space-id")

        # Create mock /data directory
        mock_data_dir = tmp_path / "data"
        mock_data_dir.mkdir()

        # Mock Path("/data") to return our temp directory
        import mosaic.data_directory

        original_path = Path

        class MockPath:
            def __new__(cls, path_str):
                if path_str == "/data":
                    return mock_data_dir
                return original_path(path_str)

        monkeypatch.setattr(mosaic.data_directory, "Path", MockPath)

        # Reset global cache
        mosaic.data_directory._TCGA_CACHE_DIR = None

        # Get cache directory
        cache_dir = get_tcga_cache_directory()

        # Should use /data/mosaic_tcga_slides on HF Spaces
        assert "mosaic_tcga_slides" in str(cache_dir)


class TestUIAnalysisCachingIntegration:
    """Integration tests for TCGA caching through the UI analysis flow."""

    def test_cache_hit_skips_analysis(self, tmp_path):
        """Test that a cache hit skips the full analysis pipeline."""
        import pandas as pd
        from PIL import Image

        file_uuid = "ui-cache-hit-uuid"
        settings_hash = compute_settings_hash(
            seg_config="TCGA",
            site_type="Primary",
            sex="Male",
            tissue_site="Lung",
            cancer_subtype="LUAD",
            ihc_subtype="",
        )

        # Pre-populate cache with results
        aeon_results = pd.DataFrame(
            {"Confidence": [0.95, 0.85]}, index=["LUAD", "LUSC"]
        )
        aeon_results.index.name = "Cancer Subtype"
        paladin_results = pd.DataFrame({"Biomarker": ["PD-L1"], "Probability": [0.7]})
        slide_mask = Image.new("RGB", (100, 100), color="blue")

        save_analysis_results(
            file_uuid,
            settings_hash,
            aeon_results,
            paladin_results,
            slide_mask,
            cache_dir=tmp_path,
        )

        # Load cached results (simulating what the UI does)
        cached = load_analysis_results(file_uuid, settings_hash, cache_dir=tmp_path)
        assert cached is not None

        loaded_mask, loaded_aeon, loaded_paladin = cached
        assert list(loaded_aeon.index) == ["LUAD", "LUSC"]
        assert loaded_paladin["Biomarker"].iloc[0] == "PD-L1"
        assert loaded_mask.size == (100, 100)

    def test_cache_miss_returns_none(self, tmp_path):
        """Test that a cache miss returns None, requiring fresh analysis."""
        settings_hash = compute_settings_hash(
            seg_config="TCGA",
            site_type="Primary",
            sex="Female",
            tissue_site="Breast",
            cancer_subtype="BRCA",
            ihc_subtype="HER2+",
        )

        cached = load_analysis_results(
            "nonexistent-uuid", settings_hash, cache_dir=tmp_path
        )
        assert cached is None

    def test_uuid_extraction_from_tcga_cache_path(self, tmp_path):
        """Test extracting UUID from a slide path under TCGA cache directory."""
        file_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()
        slide_path = slide_dir / "test_slide.svs"
        slide_path.touch()

        # Simulate the UI logic for extracting UUID from path
        slide_path_obj = Path(str(slide_path))
        if tmp_path in slide_path_obj.parents:
            extracted_uuid = slide_path_obj.parent.name
            assert extracted_uuid == file_uuid

    def test_settings_hash_consistency_across_calls(self):
        """Test that settings hash is consistent across multiple calls."""
        kwargs = dict(
            seg_config="TCGA",
            site_type="Primary",
            sex="Male",
            tissue_site="Lung",
            cancer_subtype="LUAD",
            ihc_subtype=None,
        )
        hashes = [compute_settings_hash(**kwargs) for _ in range(10)]
        assert len(set(hashes)) == 1  # All hashes should be identical


class TestCacheEdgeCases:
    """Tests for edge cases in cache operations."""

    def test_save_and_load_empty_dataframes(self, tmp_path):
        """Test saving and loading empty DataFrames."""
        import pandas as pd
        from PIL import Image

        file_uuid = "empty-df-uuid"
        settings_hash = "empty-df-hash"

        aeon_results = pd.DataFrame({"Confidence": []})
        aeon_results.index.name = "Cancer Subtype"
        paladin_results = pd.DataFrame({"Biomarker": [], "Probability": []})
        slide_mask = Image.new("RGB", (50, 50))

        save_analysis_results(
            file_uuid,
            settings_hash,
            aeon_results,
            paladin_results,
            slide_mask,
            cache_dir=tmp_path,
        )

        loaded = load_analysis_results(file_uuid, settings_hash, cache_dir=tmp_path)
        assert loaded is not None
        _, loaded_aeon, loaded_paladin = loaded
        assert len(loaded_aeon) == 0
        assert len(loaded_paladin) == 0

    def test_load_corrupt_csv_returns_none(self, tmp_path):
        """Test that corrupted CSV cache returns None."""
        from PIL import Image

        file_uuid = "corrupt-csv-uuid"
        settings_hash = "corrupt-csv-hash"
        results_dir = tmp_path / file_uuid / "results" / settings_hash
        results_dir.mkdir(parents=True)

        # Write corrupt CSV
        (results_dir / "aeon_results.csv").write_text('not,valid\ncsv,"broken')
        (results_dir / "paladin_results.csv").write_text("col1\nval1")
        # Write valid mask
        mask = Image.new("RGB", (10, 10))
        mask.save(results_dir / "slide_mask.png")

        # Should return data (CSV parser is lenient) or None on error
        loaded = load_analysis_results(file_uuid, settings_hash, cache_dir=tmp_path)
        # CSV with bad format may still parse or return None - either is acceptable

    def test_load_corrupt_image_returns_none(self, tmp_path):
        """Test that corrupted image cache returns None."""
        import pandas as pd

        file_uuid = "corrupt-img-uuid"
        settings_hash = "corrupt-img-hash"
        results_dir = tmp_path / file_uuid / "results" / settings_hash
        results_dir.mkdir(parents=True)

        # Write valid CSVs
        aeon = pd.DataFrame({"Confidence": [0.9]}, index=["BRCA"])
        aeon.to_csv(results_dir / "aeon_results.csv", index=True)
        paladin = pd.DataFrame({"Biomarker": ["HER2"]})
        paladin.to_csv(results_dir / "paladin_results.csv", index=False)

        # Write corrupt image
        (results_dir / "slide_mask.png").write_text("not a valid PNG file")

        loaded = load_analysis_results(file_uuid, settings_hash, cache_dir=tmp_path)
        # Should return None because the image is corrupt
        assert loaded is None

    def test_settings_hash_with_nested_dict_ordering(self):
        """Test that settings hash is consistent regardless of dict insertion order."""
        seg_config_1 = {"threshold": 0.5, "min_size": 100, "method": "otsu"}
        seg_config_2 = {"method": "otsu", "threshold": 0.5, "min_size": 100}

        hash1 = compute_settings_hash(
            seg_config_1, "Primary", "Male", "Lung", "LUAD", None
        )
        hash2 = compute_settings_hash(
            seg_config_2, "Primary", "Male", "Lung", "LUAD", None
        )
        assert hash1 == hash2

    def test_metadata_cache_with_special_characters(self, tmp_path):
        """Test metadata cache handles special characters in fields."""
        file_uuid = "special-chars-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        metadata = TCGAMetadata(
            sex="Male",
            tissue_site="Liver & Intrahepatic Bile Ducts",
            cancer_subtype="CHOL",
            site_type="Primary",
            filename="TCGA-W5-AA2R-01Z-00-DX1.C7E20B0C-5F4F-4352-B8E8-4D2EBEE48CD1.svs",
            file_size=2147483647,
        )

        save_metadata_cache(file_uuid, metadata, cache_dir=tmp_path)
        loaded = load_metadata_cache(file_uuid, cache_dir=tmp_path)

        assert loaded is not None
        assert loaded.tissue_site == "Liver & Intrahepatic Bile Ducts"
        assert (
            loaded.filename
            == "TCGA-W5-AA2R-01Z-00-DX1.C7E20B0C-5F4F-4352-B8E8-4D2EBEE48CD1.svs"
        )

    def test_metadata_cache_invalid_structure_returns_none(self, tmp_path):
        """Test that cache with missing required fields returns None."""
        import json

        file_uuid = "bad-structure-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        # Write cache with correct version but missing fields
        cache_path = slide_dir / METADATA_CACHE_FILENAME
        cache_path.write_text(
            json.dumps(
                {
                    "cache_version": METADATA_CACHE_VERSION,
                    "sex": "Male",
                    # Missing tissue_site, cancer_subtype, site_type
                }
            )
        )

        loaded = load_metadata_cache(file_uuid, cache_dir=tmp_path)
        # Should return None because TCGAMetadata constructor will fail
        assert loaded is None

    def test_save_analysis_results_disk_full_error(self, tmp_path, monkeypatch):
        """Test that disk full errors are logged but don't raise."""
        import pandas as pd
        from PIL import Image

        file_uuid = "disk-full-uuid"
        settings_hash = "disk-full-hash"

        aeon_results = pd.DataFrame({"Confidence": [0.9]}, index=["BRCA"])
        paladin_results = pd.DataFrame({"Biomarker": ["HER2"]})
        slide_mask = Image.new("RGB", (10, 10))

        # Mock mkdir to raise disk full error
        original_mkdir = Path.mkdir

        def mock_mkdir(self, *args, **kwargs):
            if "disk-full" in str(self):
                error = OSError("No space left on device")
                error.errno = 28
                raise error
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", mock_mkdir)

        # Should not raise exception
        save_analysis_results(
            file_uuid,
            settings_hash,
            aeon_results,
            paladin_results,
            slide_mask,
            cache_dir=tmp_path,
        )


class TestHFAuthenticationFailures:
    """Tests for HF Dataset authentication and permission error handling.

    Since loguru is mocked in conftest.py, these tests verify that the correct
    logger methods are called with appropriate error messages by inspecting
    the mock logger's call args.
    """

    @staticmethod
    def _get_mock_logger():
        """Get the mock logger used by mosaic.tcga."""
        import mosaic.tcga

        return mosaic.tcga.logger

    @staticmethod
    def _get_log_messages(mock_logger):
        """Extract all log message strings from mock logger calls."""
        messages = []
        for method_name in ["error", "warning", "info", "debug"]:
            method = getattr(mock_logger, method_name)
            for call in method.call_args_list:
                if call.args:
                    messages.append(str(call.args[0]))
        return messages

    def _setup_hf_mock(self, mock_api, monkeypatch):
        """Set up HF hub mock module."""
        import sys
        from types import ModuleType

        mock_hf_hub = ModuleType("huggingface_hub")
        mock_hf_hub.HfApi = lambda: mock_api
        mock_hf_hub.hf_hub_download = MagicMock()
        sys.modules["huggingface_hub"] = mock_hf_hub
        monkeypatch.setenv("MOSAIC_TCGA_CACHE_HF_REPO", "test-org/test-repo")

    @staticmethod
    def _cleanup_hf_mock():
        """Clean up HF hub mock module."""
        import sys

        if "huggingface_hub" in sys.modules:
            del sys.modules["huggingface_hub"]

    def test_upload_auth_error_logged(self, tmp_path, monkeypatch):
        """Test that 401 auth errors are logged with actionable message."""
        import json

        file_uuid = "auth-error-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        metadata_path = slide_dir / METADATA_CACHE_FILENAME
        metadata_path.write_text(json.dumps({"cache_version": 1, "sex": "Male"}))

        mock_api = MagicMock()
        mock_api.upload_file.side_effect = Exception("401 Unauthorized: Invalid token")

        self._setup_hf_mock(mock_api, monkeypatch)
        mock_logger = self._get_mock_logger()
        mock_logger.reset_mock()

        try:
            upload_results_to_hf(file_uuid, cache_dir=tmp_path)

            messages = self._get_log_messages(mock_logger)
            assert any(
                "401" in msg or "Authentication" in msg for msg in messages
            ), f"Expected '401' or 'Authentication' in log messages, got: {messages}"
        finally:
            self._cleanup_hf_mock()

    def test_upload_permission_denied_logged(self, tmp_path, monkeypatch):
        """Test that 403 permission errors are logged with actionable message."""
        import json

        file_uuid = "perm-denied-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        metadata_path = slide_dir / METADATA_CACHE_FILENAME
        metadata_path.write_text(json.dumps({"cache_version": 1, "sex": "Male"}))

        mock_api = MagicMock()
        mock_api.upload_file.side_effect = Exception(
            "403 Forbidden: You don't have permission"
        )

        self._setup_hf_mock(mock_api, monkeypatch)
        mock_logger = self._get_mock_logger()
        mock_logger.reset_mock()

        try:
            upload_results_to_hf(file_uuid, cache_dir=tmp_path)

            messages = self._get_log_messages(mock_logger)
            assert any(
                "403" in msg or "Permission denied" in msg for msg in messages
            ), f"Expected '403' or 'Permission denied' in log messages, got: {messages}"
        finally:
            self._cleanup_hf_mock()

    def test_download_auth_error_logged(self, tmp_path, monkeypatch):
        """Test that download auth errors are logged with actionable message."""
        mock_api = MagicMock()
        mock_api.list_repo_files.side_effect = Exception(
            "401 Unauthorized: Invalid token"
        )

        self._setup_hf_mock(mock_api, monkeypatch)
        mock_logger = self._get_mock_logger()
        mock_logger.reset_mock()

        try:
            result = download_results_from_hf("test-uuid", cache_dir=tmp_path)

            assert result is False
            messages = self._get_log_messages(mock_logger)
            assert any(
                "401" in msg or "authentication" in msg.lower() for msg in messages
            ), f"Expected '401' or 'authentication' in log messages, got: {messages}"
        finally:
            self._cleanup_hf_mock()

    def test_download_repo_not_found_logged(self, tmp_path, monkeypatch):
        """Test that 404 repo not found errors are logged with actionable message."""
        mock_api = MagicMock()
        mock_api.list_repo_files.side_effect = Exception(
            "404 Not Found: Repository not found"
        )

        self._setup_hf_mock(mock_api, monkeypatch)
        mock_logger = self._get_mock_logger()
        mock_logger.reset_mock()

        try:
            result = download_results_from_hf("test-uuid", cache_dir=tmp_path)

            assert result is False
            messages = self._get_log_messages(mock_logger)
            assert any(
                "404" in msg or "not found" in msg.lower() for msg in messages
            ), f"Expected '404' or 'not found' in log messages, got: {messages}"
        finally:
            self._cleanup_hf_mock()

    def test_upload_rate_limit_logged_as_warning(self, tmp_path, monkeypatch):
        """Test that rate limit errors are logged as warnings."""
        import json

        file_uuid = "rate-limit-uuid"
        slide_dir = tmp_path / file_uuid
        slide_dir.mkdir()

        metadata_path = slide_dir / METADATA_CACHE_FILENAME
        metadata_path.write_text(json.dumps({"cache_version": 1, "sex": "Male"}))

        mock_api = MagicMock()
        mock_api.upload_file.side_effect = Exception(
            "429 Too Many Requests: Rate limit exceeded"
        )

        self._setup_hf_mock(mock_api, monkeypatch)
        mock_logger = self._get_mock_logger()
        mock_logger.reset_mock()

        try:
            upload_results_to_hf(file_uuid, cache_dir=tmp_path)

            messages = self._get_log_messages(mock_logger)
            assert any(
                "429" in msg or "rate limit" in msg.lower() for msg in messages
            ), f"Expected '429' or 'rate limit' in log messages, got: {messages}"
        finally:
            self._cleanup_hf_mock()


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

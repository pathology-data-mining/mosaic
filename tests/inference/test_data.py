"""Unit tests for mosaic.inference.data module."""

import numpy as np
import pytest
import torch

from mosaic.inference.data import (
    CANCER_TYPE_TO_INT_MAP,
    INT_TO_CANCER_TYPE_MAP,
    SiteType,
    TileFeatureTensorDataset,
)


class TestCancerTypeMaps:
    """Test cancer type mapping constants."""

    def test_cancer_type_to_int_map_has_entries(self):
        """Test that CANCER_TYPE_TO_INT_MAP has entries."""
        assert len(CANCER_TYPE_TO_INT_MAP) > 0

    def test_int_to_cancer_type_map_has_entries(self):
        """Test that INT_TO_CANCER_TYPE_MAP has entries."""
        assert len(INT_TO_CANCER_TYPE_MAP) > 0

    def test_maps_are_inverse(self):
        """Test that the two maps are inverses of each other."""
        assert len(CANCER_TYPE_TO_INT_MAP) == len(INT_TO_CANCER_TYPE_MAP)
        for cancer_type, idx in CANCER_TYPE_TO_INT_MAP.items():
            assert INT_TO_CANCER_TYPE_MAP[idx] == cancer_type

    def test_cancer_type_to_int_map_contains_known_types(self):
        """Test that the map contains some known cancer types."""
        known_types = ["LUAD", "BRCA", "PRAD", "COAD"]
        for cancer_type in known_types:
            assert cancer_type in CANCER_TYPE_TO_INT_MAP

    def test_indices_are_unique(self):
        """Test that all indices in CANCER_TYPE_TO_INT_MAP are unique."""
        indices = list(CANCER_TYPE_TO_INT_MAP.values())
        assert len(indices) == len(set(indices))


class TestSiteType:
    """Test SiteType enum."""

    def test_site_type_primary_value(self):
        """Test that PRIMARY has correct value."""
        assert SiteType.PRIMARY.value == "Primary"

    def test_site_type_metastasis_value(self):
        """Test that METASTASIS has correct value."""
        assert SiteType.METASTASIS.value == "Metastasis"

    def test_site_type_has_two_members(self):
        """Test that SiteType enum has exactly two members."""
        assert len(list(SiteType)) == 2


class TestTileFeatureTensorDataset:
    """Test TileFeatureTensorDataset class."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return np.random.rand(100, 768).astype(np.float32)

    @pytest.fixture
    def large_features(self):
        """Create large sample features for testing padding/truncation."""
        return np.random.rand(25000, 768).astype(np.float32)

    @pytest.fixture
    def small_features(self):
        """Create small sample features for testing padding."""
        return np.random.rand(50, 768).astype(np.float32)

    def test_dataset_initialization(self, sample_features):
        """Test basic dataset initialization."""
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.PRIMARY,
            tile_features=sample_features,
            n_max_tiles=20000,
        )
        assert dataset.site_type == SiteType.PRIMARY
        assert dataset.n_max_tiles == 20000
        assert isinstance(dataset.features, torch.Tensor)

    def test_dataset_length(self, sample_features):
        """Test that dataset length is always 1."""
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.PRIMARY,
            tile_features=sample_features,
        )
        assert len(dataset) == 1

    def test_dataset_getitem_structure(self, sample_features):
        """Test that __getitem__ returns correct structure."""
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.METASTASIS,
            tile_features=sample_features,
        )
        item = dataset[0]
        assert isinstance(item, dict)
        assert "site" in item
        assert "tile_tensor" in item
        assert item["site"] == "Metastasis"
        assert isinstance(item["tile_tensor"], torch.Tensor)

    def test_features_are_padded_when_small(self, small_features):
        """Test that features are padded when fewer than n_max_tiles."""
        n_max_tiles = 1000
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.PRIMARY,
            tile_features=small_features,
            n_max_tiles=n_max_tiles,
        )
        assert dataset.features.shape[0] == n_max_tiles
        assert dataset.features.shape[1] == small_features.shape[1]

    def test_features_are_truncated_when_large(self, large_features):
        """Test that features are truncated when more than n_max_tiles."""
        n_max_tiles = 20000
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.PRIMARY,
            tile_features=large_features,
            n_max_tiles=n_max_tiles,
        )
        assert dataset.features.shape[0] == n_max_tiles
        assert dataset.features.shape[1] == large_features.shape[1]

    def test_features_dtype_is_float32(self, sample_features):
        """Test that features are converted to float32."""
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.PRIMARY,
            tile_features=sample_features,
        )
        assert dataset.features.dtype == torch.float32

    def test_site_type_primary(self, sample_features):
        """Test dataset with PRIMARY site type."""
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.PRIMARY,
            tile_features=sample_features,
        )
        item = dataset[0]
        assert item["site"] == "Primary"

    def test_site_type_metastasis(self, sample_features):
        """Test dataset with METASTASIS site type."""
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.METASTASIS,
            tile_features=sample_features,
        )
        item = dataset[0]
        assert item["site"] == "Metastasis"

    def test_features_exact_size(self):
        """Test that features of exactly n_max_tiles are not modified."""
        n_max_tiles = 100
        features = np.random.rand(n_max_tiles, 768).astype(np.float32)
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.PRIMARY,
            tile_features=features,
            n_max_tiles=n_max_tiles,
        )
        assert dataset.features.shape[0] == n_max_tiles
        assert dataset.features.shape[1] == 768

    def test_features_shape_preserved(self, sample_features):
        """Test that feature dimensionality is preserved."""
        original_dim = sample_features.shape[1]
        dataset = TileFeatureTensorDataset(
            site_type=SiteType.PRIMARY,
            tile_features=sample_features,
        )
        assert dataset.features.shape[1] == original_dim

    def test_different_feature_dimensions(self):
        """Test dataset with different feature dimensions."""
        for dim in [256, 512, 768, 1024]:
            features = np.random.rand(100, dim).astype(np.float32)
            dataset = TileFeatureTensorDataset(
                site_type=SiteType.PRIMARY,
                tile_features=features,
            )
            assert dataset.features.shape[1] == dim

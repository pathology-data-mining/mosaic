"""Unit tests for mosaic.inference.aeon module."""

import numpy as np
import pytest
import torch

from mosaic.inference.aeon import CANCER_TYPES_TO_DROP
from mosaic.inference.data import (
    CANCER_TYPE_TO_INT_MAP,
    INT_TO_CANCER_TYPE_MAP,
)


class TestAeonConstants:
    """Test constants defined in aeon module."""

    def test_cancer_types_to_drop_is_list(self):
        """Test that CANCER_TYPES_TO_DROP is a list."""
        assert isinstance(CANCER_TYPES_TO_DROP, list)

    def test_cancer_types_to_drop_has_entries(self):
        """Test that CANCER_TYPES_TO_DROP has entries."""
        assert len(CANCER_TYPES_TO_DROP) > 0

    def test_cancer_types_to_drop_are_strings(self):
        """Test that all cancer types are strings."""
        for cancer_type in CANCER_TYPES_TO_DROP:
            assert isinstance(cancer_type, str)

    def test_cancer_types_to_drop_are_valid(self):
        """Test that all cancer types to drop are valid cancer type codes."""
        # They should all be uppercase alphanumeric codes
        for cancer_type in CANCER_TYPES_TO_DROP:
            assert cancer_type.isupper()
            assert len(cancer_type) >= 2
            assert len(cancer_type) <= 10

    def test_cancer_types_to_drop_contains_expected_types(self):
        """Test that specific cancer types are in the drop list."""
        # Check that some known cancer types to drop are in the list
        expected_types = ["UDMN", "CUP", "NOT"]
        for cancer_type in expected_types:
            assert cancer_type in CANCER_TYPES_TO_DROP

    def test_cancer_type_maps_available(self):
        """Test that cancer type maps are available."""
        assert CANCER_TYPE_TO_INT_MAP is not None
        assert INT_TO_CANCER_TYPE_MAP is not None
        assert len(CANCER_TYPE_TO_INT_MAP) > 0
        assert len(INT_TO_CANCER_TYPE_MAP) > 0

    def test_batch_size_constant(self):
        """Test that BATCH_SIZE constant is defined."""
        from mosaic.inference.aeon import BATCH_SIZE

        assert isinstance(BATCH_SIZE, int)
        assert BATCH_SIZE > 0

    def test_num_workers_constant(self):
        """Test that NUM_WORKERS constant is defined."""
        from mosaic.inference.aeon import NUM_WORKERS

        assert isinstance(NUM_WORKERS, int)
        assert NUM_WORKERS > 0

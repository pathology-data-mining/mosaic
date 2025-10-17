"""Unit tests for mosaic.inference.aeon module."""

import numpy as np
import pytest
import torch

from mosaic.inference.aeon import (
    CANCER_TYPE_TO_INT_MAP,
    INT_TO_CANCER_TYPE_MAP,
    col_indices_to_drop,
)


class TestAeonConstants:
    """Test constants defined in aeon module."""

    def test_col_indices_to_drop_is_list(self):
        """Test that col_indices_to_drop is a list."""
        assert isinstance(col_indices_to_drop, list)

    def test_col_indices_to_drop_has_entries(self):
        """Test that col_indices_to_drop has entries."""
        assert len(col_indices_to_drop) > 0

    def test_col_indices_to_drop_are_integers(self):
        """Test that all indices are integers."""
        for idx in col_indices_to_drop:
            assert isinstance(idx, int)

    def test_col_indices_to_drop_are_valid(self):
        """Test that all indices are valid cancer type indices."""
        max_idx = max(CANCER_TYPE_TO_INT_MAP.values())
        for idx in col_indices_to_drop:
            assert 0 <= idx <= max_idx

    def test_col_indices_to_drop_contains_expected_types(self):
        """Test that specific cancer types are in the drop list."""
        # Check that some known cancer types to drop are in the list
        drop_types = ["UDMN", "CUP", "BRCA", "MEL"]
        for cancer_type in drop_types:
            if cancer_type in CANCER_TYPE_TO_INT_MAP:
                idx = CANCER_TYPE_TO_INT_MAP[cancer_type]
                assert idx in col_indices_to_drop

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

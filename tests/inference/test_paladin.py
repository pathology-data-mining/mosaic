"""Unit tests for mosaic.inference.paladin module."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.inference.paladin import (
    UsageError,
    load_aeon_scores,
    load_model_map,
    select_cancer_subtypes,
    logits_to_point_estimates,
)
import torch


class TestLoadModelMap:
    """Test load_model_map function."""

    @pytest.fixture
    def temp_model_map_csv(self):
        """Create a temporary model map CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["cancer_subtype", "target_name", "model_path"])
            writer.writerow(["LUAD", "PD-L1", "/path/to/luad_pdl1.pkl"])
            writer.writerow(["LUAD", "EGFR", "/path/to/luad_egfr.pkl"])
            writer.writerow(["BRCA", "HER2", "/path/to/brca_her2.pkl"])
            writer.writerow(["COAD", "MSI_TYPE", "/path/to/coad_msi.pkl"])
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink()

    def test_load_model_map_structure(self, temp_model_map_csv):
        """Test that load_model_map returns correct structure."""
        model_map = load_model_map(temp_model_map_csv)
        assert isinstance(model_map, dict)
        assert "LUAD" in model_map
        assert "BRCA" in model_map
        assert "COAD" in model_map

    def test_load_model_map_nested_dict(self, temp_model_map_csv):
        """Test that model_map contains nested dictionaries."""
        model_map = load_model_map(temp_model_map_csv)
        assert isinstance(model_map["LUAD"], dict)
        assert "PD-L1" in model_map["LUAD"]
        assert "EGFR" in model_map["LUAD"]

    def test_load_model_map_values(self, temp_model_map_csv):
        """Test that model_map contains correct values."""
        model_map = load_model_map(temp_model_map_csv)
        assert model_map["LUAD"]["PD-L1"] == "/path/to/luad_pdl1.pkl"
        assert model_map["LUAD"]["EGFR"] == "/path/to/luad_egfr.pkl"
        assert model_map["BRCA"]["HER2"] == "/path/to/brca_her2.pkl"
        assert model_map["COAD"]["MSI_TYPE"] == "/path/to/coad_msi.pkl"

    def test_load_model_map_multiple_targets_per_subtype(self, temp_model_map_csv):
        """Test that cancer subtypes can have multiple targets."""
        model_map = load_model_map(temp_model_map_csv)
        assert len(model_map["LUAD"]) == 2


class TestLoadAeonScores:
    """Test load_aeon_scores function."""

    @pytest.fixture
    def sample_aeon_df(self):
        """Create a sample Aeon results DataFrame."""
        return pd.DataFrame(
            {
                "Cancer Subtype": ["LUAD", "BRCA", "COAD", "PRAD"],
                "Confidence": [0.85, 0.10, 0.03, 0.02],
            }
        )

    def test_load_aeon_scores_returns_dict(self, sample_aeon_df):
        """Test that load_aeon_scores returns a dictionary."""
        scores = load_aeon_scores(sample_aeon_df)
        assert isinstance(scores, dict)

    def test_load_aeon_scores_correct_mapping(self, sample_aeon_df):
        """Test that scores are correctly mapped."""
        scores = load_aeon_scores(sample_aeon_df)
        assert scores["LUAD"] == 0.85
        assert scores["BRCA"] == 0.10
        assert scores["COAD"] == 0.03
        assert scores["PRAD"] == 0.02

    def test_load_aeon_scores_all_entries(self, sample_aeon_df):
        """Test that all entries are loaded."""
        scores = load_aeon_scores(sample_aeon_df)
        assert len(scores) == 4

    def test_load_aeon_scores_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame({"Cancer Subtype": [], "Confidence": []})
        scores = load_aeon_scores(empty_df)
        assert isinstance(scores, dict)
        assert len(scores) == 0


class TestSelectCancerSubtypes:
    """Test select_cancer_subtypes function."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample Aeon scores."""
        return {
            "LUAD": 0.85,
            "BRCA": 0.10,
            "COAD": 0.03,
            "PRAD": 0.02,
        }

    def test_select_top_one_cancer_subtype(self, sample_scores):
        """Test selecting the top cancer subtype."""
        result = select_cancer_subtypes(sample_scores, k=1)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "LUAD"

    def test_select_top_three_cancer_subtypes(self, sample_scores):
        """Test selecting the top three cancer subtypes."""
        result = select_cancer_subtypes(sample_scores, k=3)
        assert len(result) == 3
        assert result[0] == "LUAD"
        assert result[1] == "BRCA"
        assert result[2] == "COAD"

    def test_select_all_cancer_subtypes(self, sample_scores):
        """Test selecting all cancer subtypes."""
        result = select_cancer_subtypes(sample_scores, k=10)
        assert len(result) == 4
        assert result[0] == "LUAD"
        assert result[-1] == "PRAD"

    def test_select_default_k_value(self, sample_scores):
        """Test that default k=1 is used."""
        result = select_cancer_subtypes(sample_scores)
        assert len(result) == 1
        assert result[0] == "LUAD"

    def test_select_with_empty_scores(self):
        """Test handling of empty scores dictionary."""
        result = select_cancer_subtypes({}, k=1)
        assert isinstance(result, list)
        assert len(result) == 0


class TestLogitsToPointEstimates:
    """Test logits_to_point_estimates function."""

    def test_logits_to_point_estimates_shape(self):
        """Test that output shape is correct."""
        # logits shape: (batch_size, 2 * n_tasks)
        torch.manual_seed(42)
        batch_size = 4
        n_tasks = 5
        logits = torch.rand(batch_size, 2 * n_tasks)
        result = logits_to_point_estimates(logits)
        assert result.shape == (batch_size, n_tasks)

    def test_logits_to_point_estimates_values_in_range(self):
        """Test that point estimates are in [0, 1] range."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 1.0, 1.0]])
        result = logits_to_point_estimates(logits)
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_logits_to_point_estimates_calculation(self):
        """Test that calculation is correct."""
        logits = torch.tensor([[2.0, 4.0]])  # alpha=2, beta=4
        result = logits_to_point_estimates(logits)
        expected = 2.0 / (2.0 + 4.0)
        assert torch.isclose(result[0, 0], torch.tensor(expected))

    def test_logits_to_point_estimates_single_batch(self):
        """Test with single batch."""
        logits = torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]])
        result = logits_to_point_estimates(logits)
        assert result.shape == (1, 3)

    def test_logits_to_point_estimates_multiple_batches(self):
        """Test with multiple batches."""
        torch.manual_seed(42)
        logits = torch.rand(10, 8)  # 10 batches, 4 tasks
        result = logits_to_point_estimates(logits)
        assert result.shape == (10, 4)


class TestUsageError:
    """Test UsageError exception class."""

    def test_usage_error_is_exception(self):
        """Test that UsageError is an Exception."""
        assert issubclass(UsageError, Exception)

    def test_usage_error_can_be_raised(self):
        """Test that UsageError can be raised."""
        with pytest.raises(UsageError):
            raise UsageError("Test error message")

    def test_usage_error_message(self):
        """Test that UsageError message is preserved."""
        message = "Test error message"
        with pytest.raises(UsageError, match=message):
            raise UsageError(message)

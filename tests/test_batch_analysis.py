"""Integration tests for batch_analysis module.

Tests the batch processing coordinator and end-to-end batch workflow.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from mosaic.batch_analysis import analyze_slides_batch


class TestAnalyzeSlidesBatch:
    """Test analyze_slides_batch function."""

    @pytest.fixture
    def sample_settings_df(self):
        """Create sample settings DataFrame for testing."""
        return pd.DataFrame({
            "Slide": ["slide1.svs", "slide2.svs", "slide3.svs"],
            "Site Type": ["Primary", "Primary", "Metastatic"],
            "Sex": ["Male", "Female", "Unknown"],
            "Tissue Site": ["Lung", "Breast", "Unknown"],
            "Cancer Subtype": ["Unknown", "Unknown", "LUAD"],
            "IHC Subtype": ["", "HR+/HER2-", ""],
            "Segmentation Config": ["Biopsy", "Resection", "Biopsy"],
        })

    @pytest.fixture
    def cancer_subtype_name_map(self):
        """Sample cancer subtype name mapping."""
        return {
            "Unknown": "Unknown",
            "Lung Adenocarcinoma": "LUAD",
            "Breast Invasive Ductal Carcinoma": "IDC",
        }

    @patch('mosaic.batch_analysis.load_all_models')
    @patch('mosaic.batch_analysis.analyze_slide_with_models')
    def test_batch_analysis_basic(
        self, mock_analyze_slide, mock_load_models, sample_settings_df, cancer_subtype_name_map
    ):
        """Test basic batch analysis workflow."""
        # Mock model cache
        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_load_models.return_value = mock_cache

        # Mock analyze_slide_with_models results
        mock_mask = Mock()
        mock_aeon = pd.DataFrame({"Cancer Subtype": ["LUAD"], "Confidence": [0.95]})
        mock_paladin = pd.DataFrame({
            "Cancer Subtype": ["LUAD"],
            "Biomarker": ["EGFR"],
            "Score": [0.85]
        })
        mock_analyze_slide.return_value = (mock_mask, mock_aeon, mock_paladin)

        slides = ["slide1.svs", "slide2.svs", "slide3.svs"]

        # Run batch analysis
        masks, aeon_results, paladin_results = analyze_slides_batch(
            slides=slides,
            settings_df=sample_settings_df,
            cancer_subtype_name_map=cancer_subtype_name_map,
            num_workers=4,
        )

        # Verify models were loaded once
        mock_load_models.assert_called_once()

        # Verify analyze_slide_with_models was called for each slide
        assert mock_analyze_slide.call_count == 3

        # Verify cleanup was called
        mock_cache.cleanup.assert_called_once()

        # Verify results structure
        assert len(masks) == 3
        assert len(aeon_results) == 3
        assert len(paladin_results) == 3

    @patch('mosaic.batch_analysis.load_all_models')
    @patch('mosaic.batch_analysis.analyze_slide_with_models')
    def test_batch_analysis_with_failures(
        self, mock_analyze_slide, mock_load_models, sample_settings_df, cancer_subtype_name_map
    ):
        """Test batch analysis continues when individual slides fail."""
        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_load_models.return_value = mock_cache

        # First slide succeeds, second fails, third succeeds
        mock_mask = Mock()
        mock_aeon = pd.DataFrame({"Cancer Subtype": ["LUAD"], "Confidence": [0.95]})
        mock_paladin = pd.DataFrame({
            "Cancer Subtype": ["LUAD"],
            "Biomarker": ["EGFR"],
            "Score": [0.85]
        })

        mock_analyze_slide.side_effect = [
            (mock_mask, mock_aeon, mock_paladin),  # Slide 1: success
            Exception("Tissue segmentation failed"),  # Slide 2: failure
            (mock_mask, mock_aeon, mock_paladin),  # Slide 3: success
        ]

        slides = ["slide1.svs", "slide2.svs", "slide3.svs"]

        # Should not raise exception
        masks, aeon_results, paladin_results = analyze_slides_batch(
            slides=slides,
            settings_df=sample_settings_df,
            cancer_subtype_name_map=cancer_subtype_name_map,
        )

        # Should have results for 2 out of 3 slides
        assert len(masks) == 2
        assert len(aeon_results) == 2
        assert len(paladin_results) == 2

        # Cleanup should still be called
        mock_cache.cleanup.assert_called_once()

    @patch('mosaic.batch_analysis.load_all_models')
    def test_batch_analysis_cleanup_on_error(
        self, mock_load_models, sample_settings_df, cancer_subtype_name_map
    ):
        """Test cleanup is called even when load_all_models fails."""
        mock_load_models.side_effect = RuntimeError("Failed to load models")

        slides = ["slide1.svs"]

        with pytest.raises(RuntimeError, match="Failed to load models"):
            analyze_slides_batch(
                slides=slides,
                settings_df=sample_settings_df,
                cancer_subtype_name_map=cancer_subtype_name_map,
            )

    @patch('mosaic.batch_analysis.load_all_models')
    @patch('mosaic.batch_analysis.analyze_slide_with_models')
    def test_batch_analysis_empty_results(
        self, mock_analyze_slide, mock_load_models, sample_settings_df, cancer_subtype_name_map
    ):
        """Test batch analysis with slides that have no tissue."""
        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_load_models.return_value = mock_cache

        # All slides return None (no tissue found)
        mock_analyze_slide.return_value = (None, None, None)

        slides = ["slide1.svs", "slide2.svs"]

        masks, aeon_results, paladin_results = analyze_slides_batch(
            slides=slides,
            settings_df=sample_settings_df[:2],
            cancer_subtype_name_map=cancer_subtype_name_map,
        )

        # Should have empty results
        assert len(masks) == 0
        assert len(aeon_results) == 0
        assert len(paladin_results) == 0

        # Cleanup should still be called
        mock_cache.cleanup.assert_called_once()

    @patch('mosaic.batch_analysis.load_all_models')
    @patch('mosaic.batch_analysis.analyze_slide_with_models')
    def test_batch_analysis_aggressive_memory_management(
        self, mock_analyze_slide, mock_load_models, sample_settings_df, cancer_subtype_name_map
    ):
        """Test batch analysis with explicit aggressive memory management."""
        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_cache.aggressive_memory_mgmt = True
        mock_load_models.return_value = mock_cache

        mock_analyze_slide.return_value = (Mock(), Mock(), Mock())

        slides = ["slide1.svs"]

        analyze_slides_batch(
            slides=slides,
            settings_df=sample_settings_df[:1],
            cancer_subtype_name_map=cancer_subtype_name_map,
            aggressive_memory_mgmt=True,
        )

        # Verify aggressive_memory_mgmt was passed to load_all_models
        mock_load_models.assert_called_once_with(
            use_gpu=True,
            aggressive_memory_mgmt=True,
        )

    @patch('mosaic.batch_analysis.load_all_models')
    @patch('mosaic.batch_analysis.analyze_slide_with_models')
    def test_batch_analysis_progress_tracking(
        self, mock_analyze_slide, mock_load_models, sample_settings_df, cancer_subtype_name_map
    ):
        """Test batch analysis updates progress correctly."""
        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_load_models.return_value = mock_cache

        mock_analyze_slide.return_value = (Mock(), Mock(), Mock())

        mock_progress = Mock()
        slides = ["slide1.svs", "slide2.svs", "slide3.svs"]

        analyze_slides_batch(
            slides=slides,
            settings_df=sample_settings_df,
            cancer_subtype_name_map=cancer_subtype_name_map,
            progress=mock_progress,
        )

        # Verify progress was called
        assert mock_progress.call_count > 0

        # Verify final progress call
        final_call = mock_progress.call_args_list[-1]
        assert final_call[0][0] == 1.0  # Should be 100% at end

    @patch('mosaic.batch_analysis.load_all_models')
    @patch('mosaic.batch_analysis.analyze_slide_with_models')
    def test_batch_analysis_multi_slide_naming(
        self, mock_analyze_slide, mock_load_models, sample_settings_df, cancer_subtype_name_map
    ):
        """Test that multi-slide results include slide names."""
        mock_cache = Mock()
        mock_cache.cleanup = Mock()
        mock_load_models.return_value = mock_cache

        mock_mask = Mock()
        mock_aeon = pd.DataFrame({"Cancer Subtype": ["LUAD"], "Confidence": [0.95]})
        mock_paladin = pd.DataFrame({
            "Cancer Subtype": ["LUAD"],
            "Biomarker": ["EGFR"],
            "Score": [0.85]
        })
        mock_analyze_slide.return_value = (mock_mask, mock_aeon, mock_paladin)

        slides = ["slide1.svs", "slide2.svs"]

        masks, aeon_results, paladin_results = analyze_slides_batch(
            slides=slides,
            settings_df=sample_settings_df[:2],
            cancer_subtype_name_map=cancer_subtype_name_map,
        )

        # Verify slide names are in results
        assert len(masks) == 2
        assert masks[0][1] == "slide1.svs"
        assert masks[1][1] == "slide2.svs"

        # Paladin results should have Slide column
        assert "Slide" in paladin_results[0].columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

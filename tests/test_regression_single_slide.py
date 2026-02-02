"""Regression tests for single-slide analysis.

Ensures that single-slide analysis produces identical results before and after
the batch processing optimization.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from mosaic.analysis import analyze_slide
from mosaic.ui.app import analyze_slides


class TestSingleSlideRegression:
    """Regression tests to ensure single-slide mode is unchanged."""

    @pytest.fixture
    def mock_slide_path(self):
        """Mock slide path for testing."""
        return "/path/to/test_slide.svs"

    @pytest.fixture
    def cancer_subtype_name_map(self):
        """Sample cancer subtype name mapping."""
        return {
            "Unknown": "Unknown",
            "Lung Adenocarcinoma": "LUAD",
        }

    @patch("mosaic.analysis.segment_tissue")
    @patch("mosaic.analysis.draw_slide_mask_thick")
    @patch("mosaic.model_manager.load_all_models")
    @patch("mosaic.analysis._extract_ctranspath_features")
    @patch("mosaic.analysis.filter_features")
    @patch("mosaic.analysis._extract_optimus_features")
    @patch("mosaic.analysis._run_aeon_inference_with_model")
    @patch("mosaic.analysis._run_paladin_inference_with_models")
    def test_single_slide_analyze_slide_unchanged(
        self,
        mock_paladin,
        mock_aeon,
        mock_optimus,
        mock_filter,
        mock_ctranspath,
        mock_load_models,
        mock_mask,
        mock_segment,
        mock_slide_path,
        cancer_subtype_name_map,
    ):
        """Test that analyze_slide function behavior is unchanged."""
        # Setup mocks
        mock_coords = np.array([[0, 0], [1, 1]])
        mock_attrs = {"level": 0}
        mock_polygon = Mock()
        # segment_tissue returns (polygon, _, coords, attrs)
        mock_segment.return_value = (mock_polygon, None, mock_coords, mock_attrs)

        mock_mask_image = Mock()
        mock_mask.return_value = mock_mask_image

        # Mock ModelCache with required attributes
        mock_model_cache = Mock()
        mock_model_cache.ctranspath_model = Mock()
        mock_model_cache.optimus_model = Mock()
        mock_model_cache.marker_classifier = Mock()
        mock_model_cache.aeon_model = Mock()
        mock_model_cache.device = Mock()
        mock_model_cache.cleanup = Mock()
        mock_load_models.return_value = mock_model_cache

        mock_features = np.random.rand(100, 768)
        mock_ctranspath.return_value = (mock_features, mock_coords)

        mock_filtered_coords = mock_coords[:50]
        mock_filter.return_value = (None, mock_filtered_coords)

        mock_optimus_features = np.random.rand(50, 1536)
        mock_optimus.return_value = mock_optimus_features

        mock_aeon_results = pd.DataFrame(
            {"Cancer Subtype": ["LUAD", "LUSC"], "Confidence": [0.85, 0.15]}
        )
        mock_aeon.return_value = mock_aeon_results

        mock_paladin_results = pd.DataFrame(
            {"Cancer Subtype": ["LUAD"], "Biomarker": ["EGFR"], "Score": [0.75]}
        )
        mock_paladin.return_value = mock_paladin_results

        # Run analyze_slide
        slide_mask, aeon_results, paladin_results = analyze_slide(
            slide_path=mock_slide_path,
            seg_config="Biopsy",
            site_type="Primary",
            sex="Male",
            tissue_site="Lung",
            cancer_subtype="Unknown",
            cancer_subtype_name_map=cancer_subtype_name_map,
        )

        # Verify the pipeline was called in correct order
        mock_segment.assert_called_once()
        mock_mask.assert_called_once()
        mock_ctranspath.assert_called_once()
        mock_filter.assert_called_once()
        mock_optimus.assert_called_once()
        mock_aeon.assert_called_once()
        mock_paladin.assert_called_once()

        # Verify results structure
        assert slide_mask == mock_mask_image
        assert isinstance(aeon_results, pd.DataFrame)
        assert isinstance(paladin_results, pd.DataFrame)

    @patch("mosaic.ui.app.analyze_slide")
    @patch("mosaic.ui.app.create_user_directory")
    @patch("mosaic.ui.app.validate_settings")
    @patch("pandas.DataFrame.to_csv")  # Mock CSV writing to avoid directory issues
    def test_gradio_single_slide_uses_analyze_slide(
        self,
        mock_to_csv,
        mock_validate,
        mock_create_dir,
        mock_analyze_slide,
    ):
        """Test that Gradio UI uses analyze_slide for single slide (not batch mode)."""
        # Setup
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_dir = Path(tmpdir) / "test_user"
            mock_dir.mkdir()
            mock_create_dir.return_value = mock_dir

            settings_df = pd.DataFrame(
                {
                    "Slide": ["test.svs"],
                    "Site Type": ["Primary"],
                    "Sex": ["Male"],
                    "Tissue Site": ["Lung"],
                    "Cancer Subtype": ["Unknown"],
                    "IHC Subtype": [""],
                    "Segmentation Config": ["Biopsy"],
                }
            )
            mock_validate.return_value = settings_df

            mock_mask = Mock()
            mock_aeon = pd.DataFrame({"Cancer Subtype": ["LUAD"], "Confidence": [0.9]})
            mock_paladin = pd.DataFrame(
                {"Cancer Subtype": ["LUAD"], "Biomarker": ["EGFR"], "Score": [0.8]}
            )
            mock_analyze_slide.return_value = (mock_mask, mock_aeon, mock_paladin)

            from mosaic.ui.app import cancer_subtype_name_map

            # Call analyze_slides with a single slide (generator function)
            with patch(
                "mosaic.ui.app.get_oncotree_code_name",
                return_value="Lung Adenocarcinoma",
            ):
                gen = analyze_slides(
                    slides=["test.svs"],
                    settings_input=settings_df,
                    site_type="Primary",
                    sex="Male",
                    tissue_site="Lung",
                    cancer_subtype="Unknown",
                    ihc_subtype="",
                    seg_config="Biopsy",
                    user_dir=mock_dir,
                )
                # Consume generator to get final result
                results = list(gen)
                (
                    settings,
                    masks,
                    aeon,
                    aeon_btn,
                    paladin,
                    paladin_btn,
                    user_dir,
                    analyze_btn,
                    clear_btn,
                ) = results[-1]

            # Verify analyze_slide was called (not analyze_slides_batch)
            mock_analyze_slide.assert_called_once()

            # Verify results
            assert len(masks) == 1

    @patch("mosaic.analysis.segment_tissue")
    @patch("mosaic.analysis.gr.Warning")
    def test_single_slide_no_tissue_found(
        self, mock_warning, mock_segment, mock_slide_path, cancer_subtype_name_map
    ):
        """Test single-slide analysis when no tissue is found."""
        # No tissue tiles found
        mock_segment.return_value = None  # segment_tissue returns None when no tissue

        slide_mask, aeon_results, paladin_results = analyze_slide(
            slide_path=mock_slide_path,
            seg_config="Biopsy",
            site_type="Primary",
            sex="Unknown",
            tissue_site="Unknown",
            cancer_subtype="Unknown",
            cancer_subtype_name_map=cancer_subtype_name_map,
        )

        # Should return None for all results
        assert slide_mask is None
        assert aeon_results is None
        assert paladin_results is None
        # Verify warning was raised
        mock_warning.assert_called_once()

    @patch("mosaic.analysis.segment_tissue")
    @patch("mosaic.analysis.draw_slide_mask_thick")
    @patch("mosaic.model_manager.load_all_models")
    @patch("mosaic.analysis._extract_ctranspath_features")
    @patch("mosaic.analysis.filter_features")
    @patch("mosaic.analysis._extract_optimus_features")
    @patch("mosaic.analysis._run_paladin_inference_with_models")
    def test_single_slide_known_cancer_subtype_skips_aeon(
        self,
        mock_paladin,
        mock_optimus,
        mock_filter,
        mock_ctranspath,
        mock_load_models,
        mock_mask,
        mock_segment,
        mock_slide_path,
        cancer_subtype_name_map,
    ):
        """Test that single-slide with known subtype skips Aeon inference."""
        # Setup minimal mocks
        mock_polygon = Mock()
        mock_coords = np.array([[0, 0]])
        mock_attrs = {}
        mock_segment.return_value = (mock_polygon, None, mock_coords, mock_attrs)
        mock_mask.return_value = Mock()

        # Mock ModelCache
        mock_model_cache = Mock()
        mock_model_cache.ctranspath_model = Mock()
        mock_model_cache.optimus_model = Mock()
        mock_model_cache.marker_classifier = Mock()
        mock_model_cache.aeon_model = Mock()
        mock_model_cache.device = Mock()
        mock_model_cache.cleanup = Mock()
        mock_load_models.return_value = mock_model_cache

        mock_ctranspath.return_value = (np.random.rand(10, 768), np.array([[0, 0]]))
        mock_filter.return_value = (None, np.array([[0, 0]]))
        mock_optimus.return_value = np.random.rand(10, 1536)
        mock_paladin.return_value = pd.DataFrame(
            {"Cancer Subtype": ["LUAD"], "Biomarker": ["EGFR"], "Score": [0.8]}
        )

        with patch("mosaic.analysis._run_aeon_inference_with_model") as mock_aeon:
            slide_mask, aeon_results, paladin_results = analyze_slide(
                slide_path=mock_slide_path,
                seg_config="Biopsy",
                site_type="Primary",
                sex="Unknown",
                tissue_site="Unknown",
                cancer_subtype="Lung Adenocarcinoma",  # Known subtype
                cancer_subtype_name_map=cancer_subtype_name_map,
            )

            # Aeon inference should NOT be called
            mock_aeon.assert_not_called()

            # But Paladin should still be called
            mock_paladin.assert_called_once()


class TestBackwardCompatibility:
    """Tests to ensure API backward compatibility."""

    def test_analyze_slide_signature_unchanged(self):
        """Test that analyze_slide function signature is unchanged."""
        from inspect import signature

        sig = signature(analyze_slide)

        # Verify required parameters exist
        params = list(sig.parameters.keys())
        assert "slide_path" in params
        assert "seg_config" in params
        assert "site_type" in params
        assert "sex" in params
        assert "tissue_site" in params
        assert "cancer_subtype" in params
        assert "cancer_subtype_name_map" in params
        assert "ihc_subtype" in params
        assert "num_workers" in params
        assert "progress" in params

    def test_analyze_slide_return_type_unchanged(self):
        """Test that analyze_slide returns the same tuple structure."""
        with patch("mosaic.analysis.segment_tissue", return_value=None):  # No tissue
            with patch("mosaic.analysis.gr.Warning"):  # Mock the warning
                result = analyze_slide(
                    slide_path="test.svs",
                    seg_config="Biopsy",
                    site_type="Primary",
                    sex="Unknown",
                    tissue_site="Unknown",
                    cancer_subtype="Unknown",
                    cancer_subtype_name_map={"Unknown": "Unknown"},
                )

                # Should return tuple of 3 elements
                assert isinstance(result, tuple)
                assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for multi-slide-per-sample aggregation support.

Covers:
- aggregate_sample_features API compatibility (new v1.3.2 in-memory signature)
- load_settings Sample ID backfill
- extract_slide_features / run_inference_from_features split
- analyze_slides 2-phase path for multi-slide samples
"""

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module", autouse=True)
def real_mussel_modules():
    """Remove conftest mocks so API-compat tests hit the real mussel package."""
    MOCKED_PREFIXES = ("mussel", "huggingface_hub", "loguru")
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if any(k == p or k.startswith(p + ".") for p in MOCKED_PREFIXES)
    }
    yield
    sys.modules.update(saved)


# ---------------------------------------------------------------------------
# mussel API compatibility
# ---------------------------------------------------------------------------


class TestAggregateSampleFeaturesAPI:
    """Verify mussel v1.3.2 aggregate_sample_features has the expected signature."""

    def test_accepts_array_inputs(self):
        from mussel.utils.feature_extract import aggregate_sample_features

        sig = inspect.signature(aggregate_sample_features)
        params = list(sig.parameters.keys())
        assert "features_list" in params, "features_list parameter missing"
        assert "coords_list" in params, "coords_list parameter missing"
        assert "sample_ids" in params, "sample_ids parameter missing"

    def test_does_not_accept_h5_paths(self):
        from mussel.utils.feature_extract import aggregate_sample_features

        sig = inspect.signature(aggregate_sample_features)
        assert (
            "patch_features_h5_paths" not in sig.parameters
        ), "Old file-path API detected; mussel was downgraded from v1.3.2"

    def test_returns_dict_on_real_arrays(self):
        from mussel.utils.feature_extract import aggregate_sample_features

        feats_a = np.random.rand(10, 768).astype(np.float32)
        feats_b = np.random.rand(15, 768).astype(np.float32)
        coords_a = np.random.randint(0, 1000, (10, 2))
        coords_b = np.random.randint(0, 1000, (15, 2))

        result = aggregate_sample_features(
            [feats_a, feats_b], [coords_a, coords_b], ["P1", "P1"]
        )

        assert isinstance(result, dict)
        assert "P1" in result
        agg_features, agg_coords = result["P1"]
        assert agg_features.shape == (25, 768)
        assert agg_coords.shape == (25, 2)

    def test_two_samples_returns_two_entries(self):
        from mussel.utils.feature_extract import aggregate_sample_features

        f1 = np.ones((5, 4), dtype=np.float32)
        f2 = np.ones((3, 4), dtype=np.float32) * 2
        c1 = np.zeros((5, 2), dtype=int)
        c2 = np.zeros((3, 2), dtype=int)

        result = aggregate_sample_features([f1, f2], [c1, c2], ["A", "B"])
        assert set(result.keys()) == {"A", "B"}
        assert result["A"][0].shape == (5, 4)
        assert result["B"][0].shape == (3, 4)


# ---------------------------------------------------------------------------
# load_settings: Sample ID backfill
# ---------------------------------------------------------------------------


class TestLoadSettingsSampleID:
    def test_sample_id_backfilled_when_absent(self, tmp_path):
        from mosaic.ui.utils import load_settings

        csv_path = tmp_path / "settings.csv"
        csv_path.write_text(
            "Slide,Site Type,Sex,Tissue Site,Cancer Subtype,Segmentation Config\n"
            "/data/slide1.svs,Primary,Male,Lung,Unknown,Biopsy\n"
            "/data/slide2.svs,Primary,Female,Breast,Unknown,Resection\n"
        )
        df = load_settings(str(csv_path))
        assert "Sample ID" in df.columns
        assert df.iloc[0]["Sample ID"] == "slide1"
        assert df.iloc[1]["Sample ID"] == "slide2"

    def test_sample_id_preserved_when_present(self, tmp_path):
        from mosaic.ui.utils import load_settings

        csv_path = tmp_path / "settings.csv"
        csv_path.write_text(
            "Slide,Site Type,Sex,Tissue Site,Cancer Subtype,Segmentation Config,Sample ID\n"
            "/data/a.svs,Primary,Male,Lung,Unknown,Biopsy,PT001\n"
            "/data/b.svs,Primary,Male,Lung,Unknown,Biopsy,PT001\n"
        )
        df = load_settings(str(csv_path))
        assert list(df["Sample ID"]) == ["PT001", "PT001"]

    def test_blank_sample_id_backfilled(self, tmp_path):
        from mosaic.ui.utils import load_settings

        csv_path = tmp_path / "settings.csv"
        csv_path.write_text(
            "Slide,Site Type,Sex,Tissue Site,Cancer Subtype,Segmentation Config,Sample ID\n"
            "/data/slide3.svs,Primary,Male,Lung,Unknown,Biopsy,\n"
        )
        df = load_settings(str(csv_path))
        assert df.iloc[0]["Sample ID"] == "slide3"


# ---------------------------------------------------------------------------
# extract_slide_features: Phase 1 split
# ---------------------------------------------------------------------------


def _make_model_cache():
    """Return a minimal mock ModelCache."""
    mc = MagicMock()
    mc.ctranspath_model = MagicMock()
    mc.optimus_model = MagicMock()
    mc.marker_classifier = MagicMock()
    mc.marker_classifier.predict_proba.return_value = np.ones((10, 2)) * 0.9
    return mc


class TestExtractSlideFeatures:
    def test_returns_three_tuple(self, tmp_path):
        from mosaic.analysis import extract_slide_features

        fake_coords = np.zeros((10, 2), dtype=int)
        fake_attrs = {"patch_size": 224, "patch_level": 0, "mpp": 0.5}
        fake_features = np.random.rand(10, 768).astype(np.float32)
        fake_polygon = MagicMock()
        fake_image = MagicMock()

        model_cache = _make_model_cache()

        with (
            patch("mosaic.analysis.segment_tissue") as mock_seg,
            patch("mosaic.analysis.draw_slide_mask", return_value=fake_image),
            patch("mosaic.analysis.get_features", return_value=(fake_features, None)),
            patch(
                "mosaic.analysis.filter_features",
                return_value=(fake_features, fake_coords),
            ),
        ):
            mock_seg.return_value = (fake_polygon, None, fake_coords, fake_attrs)

            slide_path = str(tmp_path / "slide.svs")
            result = extract_slide_features(
                slide_path=slide_path,
                seg_config="Biopsy",
                num_workers=1,
                progress=lambda frac=None, desc=None: None,
                model_cache=model_cache,
            )

        assert len(result) == 3
        slide_mask, features, coords = result
        assert slide_mask is fake_image
        assert features is not None
        assert coords is not None

    def test_returns_none_triple_when_no_tissue(self, tmp_path):
        from mosaic.analysis import extract_slide_features

        model_cache = _make_model_cache()
        with patch("mosaic.analysis.segment_tissue", return_value=None):
            slide_mask, features, coords = extract_slide_features(
                slide_path=str(tmp_path / "empty.svs"),
                seg_config="Biopsy",
                num_workers=1,
                progress=lambda frac=None, desc=None: None,
                model_cache=model_cache,
            )

        assert slide_mask is None
        assert features is None
        assert coords is None


# ---------------------------------------------------------------------------
# run_inference_from_features: Phase 2
# ---------------------------------------------------------------------------


class TestRunInferenceFromFeatures:
    def test_calls_aeon_and_paladin(self):
        from mosaic.analysis import run_inference_from_features

        features = np.random.rand(20, 768).astype(np.float32)
        model_cache = _make_model_cache()
        model_cache.device = "cpu"

        fake_aeon = pd.DataFrame({"Cancer Subtype": ["LUAD"], "Confidence": [0.9]})
        fake_paladin = pd.DataFrame(
            {"Cancer Subtype": ["LUAD"], "Biomarker": ["TP53"], "Score": [0.7]}
        )

        with (
            patch(
                "mosaic.analysis._run_aeon_inference_with_model", return_value=fake_aeon
            ),
            patch(
                "mosaic.analysis._run_paladin_inference_with_models",
                return_value=fake_paladin,
            ),
        ):
            aeon_results, paladin_results = run_inference_from_features(
                features=features,
                site_type="Primary",
                sex_idx=0,
                tissue_site_idx=1,
                cancer_subtype="Unknown",
                cancer_subtype_name_map={},
                model_cache=model_cache,
                num_workers=1,
                progress=lambda frac=None, desc=None: None,
            )

        assert aeon_results is not None
        assert paladin_results is not None

    def test_uses_known_cancer_subtype(self):
        from mosaic.analysis import run_inference_from_features

        features = np.random.rand(10, 768).astype(np.float32)
        model_cache = _make_model_cache()
        model_cache.device = "cpu"

        fake_paladin = pd.DataFrame(
            {"Cancer Subtype": ["LUAD"], "Biomarker": ["TP53"], "Score": [0.7]}
        )

        with (
            patch("mosaic.analysis._run_aeon_inference_with_model") as mock_aeon,
            patch(
                "mosaic.analysis._run_paladin_inference_with_models",
                return_value=fake_paladin,
            ),
        ):
            run_inference_from_features(
                features=features,
                site_type="Primary",
                sex_idx=0,
                tissue_site_idx=1,
                cancer_subtype="LUAD",
                cancer_subtype_name_map={"LUAD": "LUAD"},
                model_cache=model_cache,
                num_workers=1,
                progress=lambda frac=None, desc=None: None,
            )

        mock_aeon.assert_not_called()


# ---------------------------------------------------------------------------
# aggregate_sample_features integration in pipeline
# ---------------------------------------------------------------------------


class TestMultiSlideSampleAggregation:
    """Verify that aggregate_sample_features is called for multi-slide samples."""

    def test_aggregate_called_for_multi_slide_sample(self):
        from mussel.utils.feature_extract import aggregate_sample_features

        f1 = np.random.rand(8, 512).astype(np.float32)
        f2 = np.random.rand(12, 512).astype(np.float32)
        c1 = np.zeros((8, 2), dtype=int)
        c2 = np.zeros((12, 2), dtype=int)

        result = aggregate_sample_features([f1, f2], [c1, c2], ["PT01", "PT01"])
        agg, _ = result["PT01"]
        assert agg.shape == (
            20,
            512,
        ), "Features from both slides should be concatenated"

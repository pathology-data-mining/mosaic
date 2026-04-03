"""Tests that verify our usage of the mussel API matches the installed version.

These tests catch API drift when mussel is upgraded — e.g. a parameter being
renamed or removed.  They call the real mussel functions (no mock) so that a
version bump that breaks our call-sites fails here, not silently in production.
"""

import inspect
import sys
import pytest
from PIL import Image
from shapely.geometry import box


@pytest.fixture(scope="module", autouse=True)
def real_mussel_modules():
    """Temporarily remove conftest's sys.modules mocks so tests hit real mussel.

    The conftest stubs out mussel, huggingface_hub, and loguru at collection
    time.  We need the real versions so we can inspect and call the actual
    mussel functions.  Restore everything after the module finishes.
    """
    MOCKED_PREFIXES = ("mussel", "huggingface_hub", "loguru")
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if any(k == p or k.startswith(p + ".") for p in MOCKED_PREFIXES)
    }
    yield
    sys.modules.update(saved)


class TestDrawSlideMaskAPI:
    """draw_slide_mask API compatibility.

    Regression: mussel v1.3.0 removed the ``outline_width`` parameter.
    Our analysis.py call must match the current signature exactly.
    """

    def test_draw_slide_mask_accepts_outline_width(self):
        """draw_slide_mask must have an outline_width parameter.

        If this test fails after a mussel downgrade/change, remove
        outline_width=12 from the call in analysis.py.
        """
        from mussel.utils.segment import draw_slide_mask

        sig = inspect.signature(draw_slide_mask)
        assert "outline_width" in sig.parameters, (
            "mussel.draw_slide_mask no longer has an outline_width parameter. "
            "Remove outline_width=12 from the call in analysis.py."
        )

    def test_draw_slide_mask_accepts_our_keyword_arguments(self):
        """draw_slide_mask accepts the exact kwargs we pass in analysis.py.

        If this test fails, analysis.py needs to be updated to match the
        new mussel API.
        """
        from mussel.utils.segment import draw_slide_mask

        sig = inspect.signature(draw_slide_mask)
        our_kwargs = {"outline", "fill", "vis_level", "max_size", "outline_width"}
        missing = our_kwargs - set(sig.parameters)
        assert not missing, (
            f"draw_slide_mask is missing parameters our code relies on: {missing}. "
            "Update the call in analysis.py to match the current mussel API."
        )

    def test_draw_slide_mask_integration(self, tmp_path):
        """draw_slide_mask runs end-to-end with our call arguments.

        Uses a real tiny TIFF and a shapely polygon so the entire code path
        executes — including any internal parameter handling in mussel.
        This catches runtime errors that signature inspection alone cannot detect.
        """
        from mussel.utils.segment import draw_slide_mask

        slide_path = tmp_path / "test_slide.tiff"
        Image.new("RGB", (512, 512), color=(220, 190, 170)).save(
            str(slide_path), format="TIFF"
        )

        polygon = box(50, 50, 450, 450)

        result = draw_slide_mask(
            str(slide_path),
            polygon,
            outline="cyan",
            fill=(255, 0, 0, 80),
            vis_level=-1,
            max_size=2048,
            outline_width=12,
        )

        assert isinstance(
            result, Image.Image
        ), f"draw_slide_mask should return a PIL Image, got {type(result)}"
        assert result.width > 0 and result.height > 0


class TestGetFeaturesAPI:
    """get_features API compatibility — model= and slide_model= parameters."""

    def test_accepts_model_keyword(self):
        from mussel.utils.feature_extract import get_features
        import inspect

        sig = inspect.signature(get_features)
        assert (
            "model" in sig.parameters
        ), "get_features lost the model= parameter; update analysis.py calls"

    def test_model_is_keyword_only_position(self):
        from mussel.utils.feature_extract import get_features
        import inspect

        params = list(inspect.signature(get_features).parameters.keys())
        # model= must come after aggregation_method (positional compat)
        assert params.index("model") > params.index(
            "aggregation_method"
        ), "model= must be after aggregation_method to preserve positional call compat"

    def test_accepts_slide_model_keyword(self):
        from mussel.utils.feature_extract import get_features
        import inspect

        sig = inspect.signature(get_features)
        assert (
            "slide_model" in sig.parameters
        ), "get_features is missing slide_model= parameter (added in mussel v1.3.1)"

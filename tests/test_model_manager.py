"""Unit tests for model_manager module.

Tests the ModelCache class and model loading functionality for batch processing.
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pickle
import gc

from mosaic.model_manager import ModelCache, load_all_models, load_paladin_model_for_inference


class TestModelCache:
    """Test ModelCache class functionality."""

    def test_model_cache_initialization(self):
        """Test ModelCache can be initialized with default values."""
        cache = ModelCache()

        assert cache.ctranspath_model is None
        assert cache.optimus_model is None
        assert cache.marker_classifier is None
        assert cache.aeon_model is None
        assert cache.paladin_models == {}
        assert cache.is_t4_gpu is False
        assert cache.aggressive_memory_mgmt is False

    def test_model_cache_with_parameters(self):
        """Test ModelCache initialization with custom parameters."""
        mock_model = Mock()
        device = torch.device("cpu")

        cache = ModelCache(
            ctranspath_model="ctranspath_path",
            optimus_model="optimus_path",
            marker_classifier=mock_model,
            aeon_model=mock_model,
            is_t4_gpu=True,
            aggressive_memory_mgmt=True,
            device=device,
        )

        assert cache.ctranspath_model == "ctranspath_path"
        assert cache.optimus_model == "optimus_path"
        assert cache.marker_classifier == mock_model
        assert cache.aeon_model == mock_model
        assert cache.is_t4_gpu is True
        assert cache.aggressive_memory_mgmt is True
        assert cache.device == device

    def test_cleanup_paladin_empty_cache(self):
        """Test cleanup_paladin with no models loaded."""
        cache = ModelCache()

        # Should not raise an error
        cache.cleanup_paladin()

        assert cache.paladin_models == {}

    def test_cleanup_paladin_with_models(self):
        """Test cleanup_paladin removes all Paladin models."""
        cache = ModelCache()
        cache.paladin_models = {
            "model1": Mock(),
            "model2": Mock(),
            "model3": Mock(),
        }

        cache.cleanup_paladin()

        assert cache.paladin_models == {}

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    def test_cleanup_paladin_clears_cuda_cache(self, mock_empty_cache, mock_cuda_available):
        """Test cleanup_paladin calls torch.cuda.empty_cache()."""
        cache = ModelCache()
        cache.paladin_models = {"model1": Mock()}

        cache.cleanup_paladin()

        mock_empty_cache.assert_called_once()

    def test_cleanup_all_models(self):
        """Test cleanup removes all models."""
        mock_model = Mock()
        cache = ModelCache(
            ctranspath_model="path1",
            optimus_model="path2",
            marker_classifier=mock_model,
            aeon_model=mock_model,
        )
        cache.paladin_models = {"model1": mock_model}

        cache.cleanup()

        assert cache.ctranspath_model is None
        assert cache.optimus_model is None
        assert cache.marker_classifier is None
        assert cache.aeon_model is None
        assert cache.paladin_models == {}


class TestLoadAllModels:
    """Test load_all_models function."""

    @patch('torch.cuda.is_available', return_value=False)
    def test_load_models_cpu_only(self, mock_cuda_available):
        """Test loading models when CUDA is not available."""
        with patch('builtins.open', create=True) as mock_open:
            with patch('pickle.load') as mock_pickle:
                # Mock the pickle loads
                mock_pickle.return_value = Mock()

                # Mock file exists checks
                with patch.object(Path, 'exists', return_value=True):
                    cache = load_all_models(use_gpu=False)

        assert cache is not None
        assert cache.device == torch.device("cpu")
        assert cache.aggressive_memory_mgmt is False

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="NVIDIA A100")
    def test_load_models_a100_gpu(self, mock_get_device, mock_cuda_available):
        """Test loading models on A100 GPU (high memory)."""
        with patch('builtins.open', create=True):
            with patch('pickle.load') as mock_pickle:
                mock_model = Mock()
                mock_model.to = Mock(return_value=mock_model)
                mock_model.eval = Mock()
                mock_pickle.return_value = mock_model

                with patch.object(Path, 'exists', return_value=True):
                    cache = load_all_models(use_gpu=True, aggressive_memory_mgmt=None)

        assert cache.device == torch.device("cuda")
        assert cache.is_t4_gpu is False
        assert cache.aggressive_memory_mgmt is False  # A100 should use caching

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="Tesla T4")
    def test_load_models_t4_gpu(self, mock_get_device, mock_cuda_available):
        """Test loading models on T4 GPU (low memory)."""
        with patch('builtins.open', create=True):
            with patch('pickle.load') as mock_pickle:
                mock_model = Mock()
                mock_model.to = Mock(return_value=mock_model)
                mock_model.eval = Mock()
                mock_pickle.return_value = mock_model

                with patch.object(Path, 'exists', return_value=True):
                    cache = load_all_models(use_gpu=True, aggressive_memory_mgmt=None)

        assert cache.device == torch.device("cuda")
        assert cache.is_t4_gpu is True
        assert cache.aggressive_memory_mgmt is True  # T4 should use aggressive mode

    def test_load_models_missing_aeon_file(self):
        """Test load_all_models raises error when Aeon model file is missing."""
        with patch.object(Path, 'exists') as mock_exists:
            # marker_classifier exists, aeon_model doesn't
            mock_exists.side_effect = lambda: mock_exists.call_count <= 1

            with pytest.raises(FileNotFoundError, match="Aeon model not found"):
                with patch('builtins.open', create=True):
                    with patch('pickle.load'):
                        load_all_models(use_gpu=False)

    @patch('torch.cuda.is_available', return_value=True)
    def test_load_models_explicit_aggressive_mode(self, mock_cuda_available):
        """Test explicit aggressive memory management setting."""
        with patch('torch.cuda.get_device_name', return_value="NVIDIA A100"):
            with patch('builtins.open', create=True):
                with patch('pickle.load') as mock_pickle:
                    mock_model = Mock()
                    mock_model.to = Mock(return_value=mock_model)
                    mock_model.eval = Mock()
                    mock_pickle.return_value = mock_model

                    with patch.object(Path, 'exists', return_value=True):
                        # Force aggressive mode even on A100
                        cache = load_all_models(use_gpu=True, aggressive_memory_mgmt=True)

        assert cache.aggressive_memory_mgmt is True  # Should respect explicit setting


class TestLoadPaladinModelForInference:
    """Test load_paladin_model_for_inference function."""

    def test_load_paladin_model_aggressive_mode(self):
        """Test loading Paladin model in aggressive mode (T4)."""
        cache = ModelCache(aggressive_memory_mgmt=True, device=torch.device("cpu"))
        model_path = Path("data/paladin/test_model.pkl")

        with patch('builtins.open', create=True):
            with patch('pickle.load') as mock_pickle:
                mock_model = Mock()
                mock_model.to = Mock(return_value=mock_model)
                mock_model.eval = Mock()
                mock_pickle.return_value = mock_model

                model = load_paladin_model_for_inference(cache, model_path)

        # In aggressive mode, model should NOT be cached
        assert str(model_path) not in cache.paladin_models
        assert model is not None
        mock_model.to.assert_called_once_with(cache.device)
        mock_model.eval.assert_called_once()

    def test_load_paladin_model_caching_mode(self):
        """Test loading Paladin model in caching mode (A100)."""
        cache = ModelCache(aggressive_memory_mgmt=False, device=torch.device("cpu"))
        model_path = Path("data/paladin/test_model.pkl")

        with patch('builtins.open', create=True):
            with patch('pickle.load') as mock_pickle:
                mock_model = Mock()
                mock_model.to = Mock(return_value=mock_model)
                mock_model.eval = Mock()
                mock_pickle.return_value = mock_model

                model = load_paladin_model_for_inference(cache, model_path)

        # In caching mode, model SHOULD be cached
        assert str(model_path) in cache.paladin_models
        assert cache.paladin_models[str(model_path)] == mock_model

    def test_load_paladin_model_from_cache(self):
        """Test loading Paladin model from cache (second call)."""
        cache = ModelCache(aggressive_memory_mgmt=False, device=torch.device("cpu"))
        model_path = Path("data/paladin/test_model.pkl")

        # Pre-populate cache
        cached_model = Mock()
        cache.paladin_models[str(model_path)] = cached_model

        # Load model - should return cached version without pickle.load
        with patch('pickle.load') as mock_pickle:
            model = load_paladin_model_for_inference(cache, model_path)

        assert model == cached_model
        mock_pickle.assert_not_called()  # Should not load from disk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

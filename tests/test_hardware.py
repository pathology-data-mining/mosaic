"""Tests for hardware compatibility between torch and the NVIDIA driver.

These tests skip silently on machines without a GPU (e.g. CI runners) but
fail loudly on machines that *have* a GPU where torch cannot use it — which
indicates a driver/torch-build version mismatch (e.g. torch compiled for
CUDA 13.0 on a server that only supports CUDA 12.x).
"""

import subprocess
import pytest
import torch


def _nvidia_smi_available() -> bool:
    """Return True if nvidia-smi is installed and detects at least one GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not _nvidia_smi_available(),
    reason="No GPU detected by nvidia-smi — skipping hardware tests",
)


def test_cuda_available():
    """torch must be able to use the GPU when one is present.

    If this fails, the installed torch wheel was compiled for a newer CUDA
    version than the NVIDIA driver supports.  Fix: pin torch to a version
    whose CUDA build matches the driver (check `torch.__version__` and the
    driver version reported in the error message).
    """
    assert torch.cuda.is_available(), (
        f"GPU detected by nvidia-smi but torch.cuda.is_available() is False. "
        f"torch={torch.__version__!r}. "
        "The torch CUDA build is likely newer than the installed NVIDIA driver. "
        "Pin torch to a compatible version in pyproject.toml override-dependencies."
    )


def test_cuda_device_count():
    """At least one CUDA device must be accessible."""
    assert torch.cuda.device_count() > 0, (
        "torch.cuda.device_count() returned 0 even though nvidia-smi reports GPUs."
    )


def test_cuda_device_usable():
    """A small tensor operation on the GPU must succeed."""
    device = torch.device("cuda:0")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert x.sum().item() == pytest.approx(6.0)

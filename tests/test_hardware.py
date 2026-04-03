"""Tests for hardware compatibility between torch and the NVIDIA driver.

GPU tests skip silently on machines without a GPU (e.g. CI runners) but
fail loudly on machines that *have* a GPU where torch cannot use it — which
indicates a driver/torch-build version mismatch (e.g. torch compiled for
CUDA 13.0 on a server that only supports CUDA 12.x).

The torchvision compatibility test always runs (no GPU needed).
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


_requires_gpu = pytest.mark.skipif(
    not _nvidia_smi_available(),
    reason="No GPU detected by nvidia-smi — skipping GPU hardware tests",
)


def test_torchvision_importable():
    """torchvision must be importable with the installed torch version.

    torchvision releases are tightly coupled to torch: torch 2.x.y requires
    torchvision 0.(x+15).y.  A mismatch (e.g. torchvision 0.26 with torch
    2.10) raises RuntimeError at import time with "operator does not exist".
    Pin both torch and torchvision in pyproject.toml override-dependencies.
    """
    try:
        import torchvision  # noqa: F401
    except RuntimeError as exc:
        pytest.fail(
            f"torchvision import failed — likely a version mismatch with torch. "
            f"torch={torch.__version__!r}. Error: {exc}"
        )


@_requires_gpu
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


@_requires_gpu
def test_cuda_device_count():
    """At least one CUDA device must be accessible."""
    assert torch.cuda.device_count() > 0, (
        "torch.cuda.device_count() returned 0 even though nvidia-smi reports GPUs."
    )


@_requires_gpu
def test_cuda_device_usable():
    """A small tensor operation on the GPU must succeed."""
    device = torch.device("cuda:0")
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert x.sum().item() == pytest.approx(6.0)

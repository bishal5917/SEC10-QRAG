"""
Device Detection Module.

Provides cross-platform GPU/accelerator detection for PyTorch workloads.
Supports CUDA (NVIDIA GPUs on Windows/Linux), MPS (Apple Silicon on macOS),
and falls back to CPU when no accelerator is available.

Priority order:
    1. CUDA — NVIDIA GPU (Windows, Linux)
    2. MPS  — Apple Silicon GPU (macOS M1/M2/M3+)
    3. CPU  — Universal fallback (always available)

Usage:
    from app.core.device import get_device, get_device_info
    device = get_device()          # → "cuda", "mps", or "cpu"
    info = get_device_info()       # → detailed dict with device metadata
"""

import platform

import torch

from app.core.logging import get_logger

logger = get_logger(__name__)


def get_device() -> str:
    """
    Detect the best available compute device for PyTorch.

    Checks for GPU availability in order of preference:
        1. CUDA (NVIDIA) — best for Windows/Linux with discrete GPUs
        2. MPS (Metal)   — Apple Silicon acceleration on macOS
        3. CPU           — always available, slowest

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Compute device selected: {device}")
    return device


def get_device_info() -> dict:
    """
    Get detailed information about the detected compute device.

    Returns a dictionary with:
        - device: The device string ("cuda", "mps", "cpu")
        - platform: Operating system (Darwin, Windows, Linux)
        - gpu_name: Name of the GPU (if CUDA) or chip info
        - gpu_memory: Available GPU memory in GB (CUDA only)
        - cuda_version: CUDA version string (if applicable)

    Returns:
        Dictionary with device metadata for logging/debugging.
    """
    device = get_device()

    info = {
        "device": device,
        "platform": platform.system(),
        "python_arch": platform.machine(),
    }

    if device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
        )
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()

    elif device == "mps":
        info["gpu_name"] = "Apple Silicon (Metal Performance Shaders)"
        # MPS doesn't expose memory info easily
        info["chip"] = platform.processor() or "Apple M-series"

    else:
        info["gpu_name"] = "None (CPU only)"
        info["note"] = (
            "No GPU detected. Install CUDA toolkit (NVIDIA) or run on "
            "Apple Silicon Mac for hardware acceleration."
        )

    return info

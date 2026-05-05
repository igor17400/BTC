"""Device setup utilities for PyTorch training."""

import torch

from src.core.io.logging import console


def setup_device(
    gpu_ids: list[int] | None = None,
    memory_limit: float = 0.9,
) -> torch.device:
    """Configure device and memory for PyTorch.

    Priority order: CUDA (with specific GPU) > MPS (Apple Silicon) > CPU.

    Args:
        gpu_ids: GPU IDs to use. Empty list or None for auto-detect.
        memory_limit: Fraction of GPU memory to allocate (0 to 1).

    Returns:
        torch.device configured for the best available hardware.
    """
    if gpu_ids and torch.cuda.is_available():
        device_id = gpu_ids[0]
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        if memory_limit < 1.0:
            torch.cuda.set_per_process_memory_fraction(memory_limit, device_id)
        console.log(
            f"PyTorch: using CUDA device {torch.cuda.get_device_name(device_id)}"
        )
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        console.log(f"PyTorch: using CUDA device {torch.cuda.get_device_name(0)}")
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        console.log("PyTorch: using Apple MPS device")
        return device

    console.log("PyTorch: using CPU device")
    return torch.device("cpu")

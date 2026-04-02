"""Device setup utilities for PyTorch training."""

import torch


def setup_device(gpu_ids: str = "") -> torch.device:
    """Auto-detect and setup the best available compute device.

    Priority order: CUDA (with specific GPU) > MPS (Apple Silicon) > CPU.

    Args:
        gpu_ids: Comma-separated GPU IDs (e.g. "0", "0,1"). Empty string
                 means auto-detect the best single device.

    Returns:
        torch.device configured for the best available hardware.
    """
    if gpu_ids and torch.cuda.is_available():
        # Use the first specified GPU
        device_id = int(gpu_ids.split(",")[0])
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        print(f"Using CUDA device: {torch.cuda.get_device_name(device_id)}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
        return device

    print("Using CPU device")
    return torch.device("cpu")

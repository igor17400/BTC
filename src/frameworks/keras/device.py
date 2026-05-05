"""Backend-aware device setup for Keras 3.

Supports JAX, PyTorch, and TensorFlow backends.
"""

import os

import keras

from src.core.io.logging import console


def _setup_jax_device(gpu_ids: list[int], memory_limit: float) -> None:
    import jax

    if not gpu_ids:
        console.log("Using CPU (JAX backend)")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        return

    devices = jax.devices()
    gpu_devices = [
        d
        for d in devices
        if "cuda" in str(type(d)).lower()
        or getattr(d, "platform", "") in ("cuda", "gpu")
    ]

    if not gpu_devices:
        console.log("No GPU found, falling back to CPU (JAX)")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        return

    if memory_limit < 1.0:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_limit)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    console.log(f"JAX: using GPU(s) {gpu_ids}, {len(gpu_devices)} available")


def _setup_torch_device(gpu_ids: list[int], memory_limit: float) -> None:
    import torch

    if not gpu_ids or not torch.cuda.is_available():
        console.log(
            f"Using CPU (PyTorch backend, CUDA available: {torch.cuda.is_available()})"
        )
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    if memory_limit < 1.0:
        for gid in gpu_ids:
            torch.cuda.set_per_process_memory_fraction(memory_limit, gid)

    console.log(
        f"PyTorch: using GPU(s) {gpu_ids}, {torch.cuda.device_count()} available"
    )


_BACKEND_SETUP = {
    "jax": _setup_jax_device,
    "torch": _setup_torch_device,
}


def setup_device(gpu_ids: list[int], memory_limit: float = 0.9) -> None:
    """Configure device and memory for the active Keras backend.

    Args:
        gpu_ids: GPU IDs to use. Empty list for CPU.
        memory_limit: Fraction of GPU memory to allocate (0 to 1).
    """
    backend = keras.backend.backend()
    setup_fn = _BACKEND_SETUP.get(backend)

    if setup_fn is None:
        console.log(f"Unknown Keras backend '{backend}', skipping device setup")
        return

    try:
        setup_fn(gpu_ids, memory_limit)
    except Exception as e:
        console.log(f"Device setup error ({backend}): {e} — falling back to defaults")

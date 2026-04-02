import os

import jax

from typing import List
from src.core.io.logging import console


def setup_device(gpu_ids: List[int], memory_limit: float = 0.9) -> None:
    """Configure JAX device and memory settings for Keras 3 + JAX backend.

    Args:
        gpu_ids: List of GPU IDs to use. Empty list for CPU.
        memory_limit: Fraction of GPU memory to allocate (0 to 1)
    """
    console.log("🚀 Setting up JAX devices...")

    if not gpu_ids:
        console.log("💻 Using CPU for training")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        return

    try:
        # Configure JAX for GPU usage
        devices = jax.devices()

        # Check for CUDA/GPU devices using multiple methods for robustness
        gpu_devices = []
        for d in devices:
            # Method 1: Check device type string (most reliable for CudaDevice)
            if 'cuda' in str(type(d)).lower():
                gpu_devices.append(d)
            # Method 2: Check device_kind attribute if it exists
            elif hasattr(d, 'device_kind') and d.device_kind in ['cuda', 'gpu']:
                gpu_devices.append(d)
            # Method 3: Check platform attribute if it exists
            elif hasattr(d, 'platform') and d.platform in ['cuda', 'gpu']:
                gpu_devices.append(d)

        if not gpu_devices:
            console.log("⚠️ No GPU found. Using CPU instead.")
            console.log(f"Available devices: {[str(d) for d in devices]}")
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            return

        # Set memory fraction for JAX
        if memory_limit < 1.0:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_limit)
            console.log(f"🎮 JAX GPU memory limit set to: {memory_limit * 100:.0f}%")

        # Configure visible GPUs
        if gpu_ids:
            gpu_ids_str = ",".join(map(str, gpu_ids))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
            console.log(f"🎮 Using GPUs: {gpu_ids_str}")

        console.log(f"Found {len(gpu_devices)} GPU devices")
        for i, device in enumerate(gpu_devices[:len(gpu_ids) if gpu_ids else len(gpu_devices)]):
            console.log(f"🔍 GPU {i}: {device}")

    except Exception as e:
        console.log(f"❌ Error setting up JAX GPU: {e}")
        console.log("⚠️ Falling back to CPU")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    # Log final JAX backend info
    console.log(f"🔍 JAX backend: {jax.default_backend()}")
    console.log(f"🔍 Available JAX devices: {len(jax.devices())}")

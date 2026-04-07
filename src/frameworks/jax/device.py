"""JAX device configuration utilities."""

import os

import jax

from src.core.io.logging import console


def setup_device(
    gpu_ids: list[int] | None = None,
    memory_limit: float = 0.9,
) -> None:
    """Configure JAX devices and GPU memory.

    Args:
        gpu_ids: List of GPU device IDs to use. Pass an empty list or
            ``None`` to fall back to CPU.
        memory_limit: Fraction of GPU memory to pre-allocate (0.0 to 1.0).
    """
    if not gpu_ids:
        console.log("JAX: no GPU IDs specified -- using CPU.")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        return

    try:
        devices = jax.devices()

        # Detect GPU devices
        gpu_devices = []
        for d in devices:
            dtype_str = str(type(d)).lower()
            if (
                "cuda" in dtype_str
                or "gpu" in dtype_str
                or hasattr(d, "device_kind")
                and d.device_kind in ("cuda", "gpu")
                or hasattr(d, "platform")
                and d.platform in ("cuda", "gpu")
            ):
                gpu_devices.append(d)

        if not gpu_devices:
            console.log(
                f"JAX: no GPU found among {[str(d) for d in devices]}. Falling back to CPU."
            )
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            return

        if memory_limit < 1.0:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_limit)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        console.log(f"JAX: using GPU(s) {gpu_ids}, {len(gpu_devices)} available")

    except Exception as exc:
        console.log(f"JAX: device setup error: {exc} -- falling back to CPU")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

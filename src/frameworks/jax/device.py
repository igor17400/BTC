"""JAX device configuration utilities."""

import logging
import os

import jax

logger = logging.getLogger(__name__)


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
    logger.info("Setting up JAX devices...")

    if not gpu_ids:
        logger.info("No GPU IDs specified -- using CPU.")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        return

    try:
        devices = jax.devices()

        # Detect GPU devices
        gpu_devices = []
        for d in devices:
            dtype_str = str(type(d)).lower()
            if "cuda" in dtype_str or "gpu" in dtype_str:
                gpu_devices.append(d)
            elif hasattr(d, "device_kind") and d.device_kind in ("cuda", "gpu"):
                gpu_devices.append(d)
            elif hasattr(d, "platform") and d.platform in ("cuda", "gpu"):
                gpu_devices.append(d)

        if not gpu_devices:
            logger.warning(
                "No GPU found among available devices: %s. Falling back to CPU.",
                [str(d) for d in devices],
            )
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            return

        # Memory fraction
        if memory_limit < 1.0:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_limit)
            logger.info("JAX GPU memory limit set to %.0f%%", memory_limit * 100)

        # Visible GPUs
        gpu_ids_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        logger.info("Using GPUs: %s", gpu_ids_str)

        for i, device in enumerate(gpu_devices[: len(gpu_ids)]):
            logger.info("GPU %d: %s", i, device)

    except Exception as exc:
        logger.error("Error setting up JAX GPU: %s -- falling back to CPU", exc)
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    logger.info("JAX backend: %s", jax.default_backend())
    logger.info("Available JAX devices: %d", len(jax.devices()))

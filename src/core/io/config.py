"""Utilities for saving and loading model configurations alongside weights."""

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

console = Console()


def save_model_config(config: DictConfig, model_weights_path: Path) -> Path:
    """Save model configuration alongside model weights.

    Args:
        config: The full configuration used for training
        model_weights_path: Path where model weights are saved

    Returns:
        Path to the saved configuration file
    """
    # Create config path next to weights file
    config_path = model_weights_path.parent / f"{model_weights_path.stem}_config.yaml"

    # Convert to container for saving
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Save configuration
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Silent — logged at end of training only
    return config_path


def load_model_config(model_weights_path: Path) -> dict[str, Any]:
    """Load model configuration from alongside model weights.

    Args:
        model_weights_path: Path where model weights are saved

    Returns:
        Dictionary containing the model configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
    """
    # Look for config file next to weights
    config_path = model_weights_path.parent / f"{model_weights_path.stem}_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model configuration not found at {config_path}. "
            "This model may have been saved without configuration."
        )

    # Load configuration
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    console.log(f"Loaded model configuration from: {config_path}")
    return config_dict


def verify_model_compatibility(
    model_config: dict[str, Any], weights_path: Path
) -> bool:
    """Verify that model configuration is compatible with saved weights.

    Args:
        model_config: Model configuration dictionary
        weights_path: Path to model weights

    Returns:
        True if compatible, False otherwise
    """
    # For now, just check that the config exists and has model section
    # In the future, could add more sophisticated checks
    if "model" not in model_config:
        console.log("[red]Model configuration missing 'model' section[/red]")
        return False

    console.log("[green]Model configuration verified successfully[/green]")
    return True

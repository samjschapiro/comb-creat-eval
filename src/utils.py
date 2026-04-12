"""Cross-track utilities with stable API."""

import shutil
import yaml
from pathlib import Path


def init_directory(output_dir: str | Path, overwrite: bool = False) -> Path:
    """Initialize an output directory with safety checks.

    Args:
        output_dir: Path to the output directory.
        overwrite: If True, remove existing directory first.

    Returns:
        Path object for the created directory.

    Raises:
        ValueError: If directory exists and overwrite is False.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise ValueError(
                f"Output directory {output_dir} already exists. "
                "Use --overwrite to replace."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_config(config_path: str | Path) -> dict:
    """Load and validate a YAML config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is missing required 'output_dir' field.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    return config


def save_config(config: dict, output_dir: str | Path, filename: str = "config.yaml") -> Path:
    """Save a copy of the config to the output directory.

    Args:
        config: Config dictionary to save.
        output_dir: Directory to save the config in.
        filename: Name for the saved config file.

    Returns:
        Path to the saved config file.
    """
    output_path = Path(output_dir) / filename
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return output_path

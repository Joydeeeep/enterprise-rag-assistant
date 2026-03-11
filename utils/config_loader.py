"""
Load and validate config from config.yaml.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load YAML config. Default path: configs/config.yaml relative to project root.

    Args:
        config_path: Override path to config file.

    Returns:
        Config dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

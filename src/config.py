"""Configuration loader for SemSearch."""

import logging
from pathlib import Path
import yaml
import os


def load_config():
    """Load configuration from config.yaml with environment variable expansion."""
    config_path = Path("config.yaml")
    try:
        with open(config_path, 'r') as file:
            content = file.read()
            content = os.path.expandvars(content)
            return yaml.safe_load(content)
    except FileNotFoundError:
        logging.getLogger(__name__).error("config.yaml file not found")
        raise
    except yaml.YAMLError as e:
        logging.getLogger(__name__).error(f"Error parsing config.yaml: {str(e)}")
        raise
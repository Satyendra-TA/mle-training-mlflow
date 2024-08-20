import logging
import logging.config
import os

import yaml


def setup_logging():
    """
    Configures the root logger

    """
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "logging_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

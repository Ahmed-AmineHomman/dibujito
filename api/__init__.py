import logging
from logging import getLogger
from typing import Optional

from .config import AppConfig
from .diffuser import Diffuser
from .llm import LLM

DEFAULT_CHECKPOINT_NAME = "dreamshaper.safetensors"


def configure_logger(logpath: Optional[str] = None) -> None:
    """Configures the logger."""
    logger = getLogger()
    logger.setLevel(logging.INFO)
    if logpath:
        file_handler = logging.FileHandler(logpath)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def configure_environment(config: AppConfig) -> None:
    """Configures the environment."""
    configure_logger(config.log_path)

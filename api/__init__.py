import logging
import os
from logging import getLogger

from .llm import LLM
from .diffuser import Diffuser
from typing import Optional

DEFAULT_MODEL_DIR = os.path.join(".", "models")
DEFAULT_CHECKPOINT_NAME = "aom3.safetensors"

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


def configure_api_keys(api_key: Optional[str] = None) -> None:
    """Configure the environment variables corresponding to the provided API keys."""
    if api_key:
        os.environ[LLM._environment_key] = api_key
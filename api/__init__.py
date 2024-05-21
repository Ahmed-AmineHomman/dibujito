import logging
import os
from logging import getLogger

DEFAULT_MODEL_DIR = os.path.join(".", "models")
DEFAULT_CHECKPOINT_NAME = "dreamshaper.safetensors"
DEFAULT_DEPOSIT = "Lykon/DreamShaper"


def configure_logger(logpath: str) -> None:
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

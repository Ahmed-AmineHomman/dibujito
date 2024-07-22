import logging
import os
import tomllib
from logging import getLogger
from typing import Optional

from .diffuser import Diffuser


def configure_logger(filepath: Optional[str] = None) -> None:
    """Configures the logger."""
    logger = getLogger()
    logger.setLevel(logging.INFO)
    if filepath:
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def get_ui_doc(language: str) -> dict[str, str]:
    output = {}
    directory = os.path.join(os.path.dirname(__file__), "data", "locales")
    available_languages = [f.split(".")[0] for f in os.listdir(directory) if f.endswith(".toml")]
    if language not in available_languages:
        logging.warning(f"unsupported language: {language} -> defaulting to English")
        with open(os.path.join(directory, f"en.toml"), "rb") as fp:
            output = tomllib.load(fp)
    else:
        with open(os.path.join(directory, f"{language}.toml"), "rb") as fp:
            output = tomllib.load(fp)
    return output

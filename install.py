import logging
import os
from argparse import ArgumentParser, Namespace

from api import configure_logger, AppConfig
from api.config import COHERE_API_KEY_ENV_NAME, OLLAMA_HOST_ENV_NAME
from api.factory import ModelFactory


def load_parameters() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--hf-api-key",
        type=str,
        required=False,
        help="Hugging Face Hub API key."
    )
    parser.add_argument(
        "--cohere-api-key",
        type=str,
        required=False,
        help="Cohere API key."
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        required=False,
        help="Ollama API host (i.e. base url)."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=False,
        help="path to the directory containing the checkpoints."
    )
    parser.add_argument(
        "--loras-dir",
        type=str,
        required=False,
        help="path to the directory containing the embeddings."
    )
    parser.add_argument(
        "--logpath",
        type=str,
        required=False,
        help="path to the directory containing the logs. If not provided, logs will be outputted in default stream."
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameters = load_parameters()
    configure_logger(logpath=parameters.logpath if parameters.logpath else None)

    logging.info("generating config file")

    # custom paths
    paths = {}
    if parameters.checkpoints_dir:
        paths["checkpoints"] = parameters.checkpoints_dir
    if parameters.loras_dir:
        paths["loras"] = parameters.loras_dir

    # custom environment variables
    if parameters.cohere_api_key:
        os.environ[COHERE_API_KEY_ENV_NAME] = parameters.cohere_api_key
    if parameters.ollama_host:
        os.environ[OLLAMA_HOST_ENV_NAME] = parameters.ollama_host

    # generate config
    config = AppConfig(paths=paths)

    logging.info("saving config file")
    config.dump("config.json")

    logging.info("preparing database")
    logging.info("downloading checkpoints")
    ModelFactory(paths=config.paths).download_all()

    logging.info("done")

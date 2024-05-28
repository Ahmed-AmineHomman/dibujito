import logging
import os
from argparse import ArgumentParser, Namespace

from api import configure_logger, AppConfig
from api.factory import ModelFactory


def load_parameters() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--api-key",
        type=str,
        required=False,
        help="Hugging Face Hub API key."
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

    logging.info("configuring model paths")
    if os.path.exists("config.json"):
        config = AppConfig.load(
            filepath="config.json",
            checkpoints=parameters.checkpoints_dir,
            loras=parameters.loras_dir,
        )
    else:
        config = AppConfig(
            checkpoints=parameters.checkpoints_dir,
            loras=parameters.loras_dir,
        )

    logging.info("dumping config")
    config.dump("config.json")

    logging.info("preparing database")
    factory = ModelFactory(config=config, api_key=parameters.api_key)

    logging.info("downloading checkpoints")
    factory.download_all()

    logging.info("done")

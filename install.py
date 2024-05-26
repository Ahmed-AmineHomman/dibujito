import logging
import os
from argparse import ArgumentParser, Namespace

from huggingface_hub import hf_hub_download

from api import configure_logger, AppConfig, DEFAULT_CHECKPOINT_NAME


def load_parameters() -> Namespace:
    parser = ArgumentParser()
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
        "--embeddings-dir",
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
            embeddings=parameters.embeddings_dir,
        )
    else:
        config = AppConfig(
            checkpoints=parameters.checkpoints_dir,
            loras=parameters.loras_dir,
            embeddings=parameters.embeddings_dir,
        )

    logging.info("dumping config")
    config.dump("config.json")

    logging.info("creating directories")
    os.makedirs(config.checkpoints, exist_ok=True)
    os.makedirs(config.loras, exist_ok=True)
    os.makedirs(config.embeddings, exist_ok=True)

    logging.info("downloading checkpoints")
    filepath = os.path.join(config.checkpoints, DEFAULT_CHECKPOINT_NAME)
    if os.path.exists(filepath):
        logging.info("model already downloaded")
    else:
        hf_hub_download(
            repo_id="WarriorMama777/OrangeMixs",
            repo_type="model",
            subfolder="Models/AbyssOrangeMix3",
            filename="AOM3_orangemixs.safetensors",
            local_dir=config.checkpoints
        )
        os.rename(
            src=os.path.join(config.checkpoints, "Models", "AbyssOrangeMix3", "AOM3_orangemixs.safetensors"),
            dst=os.path.join(config.checkpoints, DEFAULT_CHECKPOINT_NAME)
        )

    logging.info("done")

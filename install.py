from huggingface_hub import hf_hub_download
from argparse import ArgumentParser, Namespace
import os
import json

from api import configure_logger, DEFAULT_CHECKPOINT_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_DEPOSIT

import logging


def load_parameters() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="path to the directory containing the checkpoints."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=DEFAULT_EMBEDDINGS_DIR,
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

    logging.info("preparing folders")
    os.makedirs(parameters.checkpoints_dir, exist_ok=True)
    os.makedirs(parameters.embeddings_dir, exist_ok=True)

    logging.info("downloading checkpoints")
    filepath = os.path.join(parameters.checkpoints_dir, "DreamShaper8_LCM.safetensors")
    if os.path.exists(filepath):
        logging.info("model already downloaded")
    else:
        hf_hub_download(
            repo_id=DEFAULT_DEPOSIT,
            repo_type="model",
            filename="DreamShaper8_LCM.safetensors",
            local_dir=parameters.checkpoints_dir
        )

    logging.info("downloading embeddings")
    filepath = os.path.join(parameters.checkpoints_dir, "BadDream.pt")
    if os.path.exists(filepath):
        logging.info("model already downloaded")
    else:
        hf_hub_download(
            repo_id=DEFAULT_DEPOSIT,
            repo_type="model",
            filename="BadDream.pt",
            local_dir=parameters.embeddings_dir
        )

    logging.info("preparing environment")
    with open("config.json", "w") as fh:
        json.dump(
            obj={
                "checkpoints-dir": parameters.checkpoints_dir,
                "embeddings-dir": parameters.embeddings_dir,
            },
            fp=fh
        )

    logging.info("done")
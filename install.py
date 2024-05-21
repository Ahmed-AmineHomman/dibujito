import json
import logging
import os
from argparse import ArgumentParser, Namespace

from huggingface_hub import hf_hub_download

from api import configure_logger, DEFAULT_MODEL_DIR, DEFAULT_CHECKPOINT_NAME


def load_parameters() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=os.path.join(DEFAULT_MODEL_DIR, "checkpoints"),
        help="path to the directory containing the checkpoints."
    )
    parser.add_argument(
        "--loras-dir",
        type=str,
        default=os.path.join(DEFAULT_MODEL_DIR, "loras"),
        help="path to the directory containing the embeddings."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=os.path.join(DEFAULT_MODEL_DIR, "embeddings"),
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

    logging.info("preparing environment")
    with open("config.json", "w") as fh:
        json.dump(
            obj={
                "checkpoints-dir": parameters.checkpoints_dir,
                "loras-dir": parameters.loras_dir,
                "embeddings-dir": parameters.embeddings_dir,
            },
            fp=fh
        )
    os.makedirs(parameters.checkpoints_dir, exist_ok=True)
    os.makedirs(parameters.loras_dir, exist_ok=True)
    os.makedirs(parameters.embeddings_dir, exist_ok=True)

    logging.info("downloading checkpoints")
    filepath = os.path.join(parameters.checkpoints_dir, DEFAULT_CHECKPOINT_NAME)
    if os.path.exists(filepath):
        logging.info("model already downloaded")
    else:
        hf_hub_download(
            repo_id="WarriorMama777/OrangeMixs",
            repo_type="model",
            subfolder="Models/AbyssOrangeMix3",
            filename="AOM3_orangemixs.safetensors",
            local_dir=parameters.checkpoints_dir
        )
        os.rename(
            src=os.path.join(parameters.checkpoints_dir, "Models", "AbyssOrangeMix3", "AOM3_orangemixs.safetensors"),
            dst=os.path.join(parameters.checkpoints_dir, DEFAULT_CHECKPOINT_NAME)
        )

    logging.info("done")

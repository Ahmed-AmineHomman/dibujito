import logging
import os
from argparse import ArgumentParser, Namespace
from typing import List

import gradio as gr
from PIL import Image

from api import configure_logger, configure_api_keys, AppConfig
from api.diffuser import Diffuser
from api.optimizers import optimize

APP_NAME = "Dibujito"
APP_DESCRIPTION = """
A simple web app allowing to turn your creative ideas into images.
Just write what you want to draw and the app will take care of the rest.
"""
PROMPT_DESCRIPTION = """
Write anything you would like the model to draw.
Your description will be automatically improved by the LLM powering the app.

Some tips:
-   **Use details**: the more details you give to the AI, the better it will be at creating an image corresponding to what you have in mind.
-   **Be verbose**: don't bother with prompt engineering, just write what you want for your image to display.
    The LLM will take care of converting it into an optimal prompt.
"""

DIFFUSER: Diffuser
CONFIG: AppConfig


def load_parameters() -> Namespace:
    """Loads the parameters from the environment."""
    parser = ArgumentParser()
    parser.add_argument(
        "--api-key",
        type=str,
        required=False,
        help="The Cohere API key. If not provided, its value will be searched in the COHERE_API_KEY environment variable."
    )
    parser.add_argument(
        "--logpath",
        type=str,
        required=False,
        help="path to the logfile"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=False,
        help="path to the directory containing the model checkpoints."
    )
    parser.add_argument(
        "--loras-dir",
        type=str,
        required=False,
        help="path to the directory containing the loras."
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=False,
        help="path to the directory containing the embeddings."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="name of the checkpoint file to load during app start."
    )
    return parser.parse_args()


def load_pipeline(
        checkpoint: str,
        lora: List[str],
        embeddings: List[str],
) -> None:
    logging.info(f"loading checkpoint")
    DIFFUSER.load_checkpoint(checkpoint)

    logging.info("loading loras")
    for l in lora:
        DIFFUSER.load_lora(l)

    logging.info("loading embeddings")
    for e in embeddings:
        DIFFUSER.load_embeddings(e)

    logging.info("done")


def generate(
        prompt: str,
        negative_prompt: str,
        project: str,
        steps: int,
        guidance: float,
        aspect: str,
) -> Image:
    """Generates the image corresponding to the provided prompt."""
    logging.info(f"optimizing prompt")
    try:
        optimized_prompt = optimize(prompt, model="sd1", project=project)
    except Exception as error:
        message = f"Error (prompt optim): {error}"
        logging.error(message)
        raise gr.Error(message)
    logging.info(f"optimized prompt: {optimized_prompt}")

    logging.info(f"generating image")
    try:
        image = DIFFUSER.imagine(
            prompt=optimized_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            aspect=aspect
        )
    except Exception as error:
        message = f"Error (image gen): {error}"
        logging.error(message)
        raise gr.Error(message)
    logging.info(f"image generated")

    return image


def build_ui(
        checkpoint: str
) -> gr.Blocks:
    """Builds the UI."""
    # list models
    models = {}
    for subtype in ["checkpoints", "loras", "embeddings"]:
        base_dir = CONFIG.get(subtype)
        models[subtype] = [
            f for f in os.listdir(base_dir) if
            (os.path.isfile(os.path.join(base_dir, f)) and (f.split(".")[-1] in ["safetensors", "ckpt", "bin", "pt"]))
        ]

    with gr.Blocks() as app:
        gr.Markdown(f"# {APP_NAME}\n\n{APP_DESCRIPTION}")

        # prompting section
        with gr.Row():
            image = gr.Image(label="Image", format="png", type="pil", height=1024, width=1024, scale=2)
            with gr.Column(scale=1):
                with gr.Accordion(label="Models", open=False):
                    checkpoint = gr.Dropdown(label="Diffuser", choices=models.get("checkpoints"), value=checkpoint,
                                             multiselect=False)
                    loras = gr.Dropdown(label="LoRAs", choices=models.get("loras"), value=[], multiselect=True)
                    embeddings = gr.Dropdown(label="Embeddings", choices=models.get("embeddings"), value=[],
                                             multiselect=True)
                    pipeline_btn = gr.Button(value="Load pipeline", variant="secondary")
                with gr.Accordion(label="Parameters", open=False):
                    steps = gr.Slider(label="# steps", minimum=1, maximum=50, value=15, step=1)
                    guidance = gr.Slider(label="guidance", minimum=1, maximum=20, value=7, step=0.5)
                    aspect = gr.Dropdown(label="Aspect", choices=["square", "portrait", "landscape"], value="square")
                with gr.Accordion(label="Prompting", open=True):
                    negative_prompt = gr.TextArea(
                        label="Negative prompt",
                        placeholder="describe what you don't want to draw",
                        interactive=True
                    )
                    project = gr.Text(label="Project", placeholder="describe your project here", interactive=True)
        with gr.Row():
            prompt = gr.TextArea(
                label="Prompt",
                placeholder="describe what you want to draw",
                scale=2
            )
            generate_btn = gr.Button(value="Run", variant="primary", scale=1)

        # UI logic
        pipeline_btn.click(
            fn=load_pipeline,
            inputs=[checkpoint, loras, embeddings],
            outputs=[]
        )
        generate_btn.click(
            fn=generate,
            inputs=[prompt, negative_prompt, project, steps, guidance, aspect],
            outputs=[image])
    return app


if __name__ == "__main__":
    parameters = load_parameters()

    # configure app
    configure_logger(logpath=parameters.logpath if parameters.logpath else None)
    configure_api_keys(api_key=parameters.api_key if parameters.api_key else None)

    # generate default config file if not already existing
    logging.info("loading app config")
    if os.path.exists("./config.json"):
        CONFIG = AppConfig.load(
            filepath="config.json",
            checkpoints=parameters.checkpoints_dir,
            loras=parameters.loras_dir,
            embeddings=parameters.embeddings_dir,
        )
    else:
        logging.warning("no config file found -> using defaults paths")
        CONFIG = AppConfig(
            checkpoints=parameters.checkpoints_dir,
            loras=parameters.loras_dir,
            embeddings=parameters.embeddings_dir,
        )

    logging.info(f"loading diffusion pipeline")
    DIFFUSER = Diffuser(
        checkpoint_dir=CONFIG.checkpoints,
        lora_dir=CONFIG.loras,
        embeddings_dir=CONFIG.embeddings,
        checkpoint=parameters.checkpoint
    )

    logging.info(f"building UI")
    app = build_ui(
        checkpoint=parameters.checkpoint
    )

    logging.info("running app")
    app.launch()

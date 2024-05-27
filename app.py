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


def generate(
        checkpoint: str,
        lora: List[str],
        embeddings: List[str],
        prompt: str,
        negative_prompt: str,
        project: str,
        steps: int,
        guidance: float,
        aspect: str,
        scheduler: str,
        progress: gr.Progress = gr.Progress()
) -> Image:
    """Generates the image corresponding to the provided prompt."""
    progress(progress=0, desc="loading checkpoint")
    if (not DIFFUSER.ready) or (DIFFUSER.checkpoint != checkpoint):
        logging.info(f"loading checkpoint")
        DIFFUSER.load_checkpoint(checkpoint)
    else:
        logging.info("reusing pipeline")

    progress(progress=0, desc="loading loras")
    for i, l in enumerate(lora):
        progress(progress=0.2 + 0.1 * i / len(lora), desc=f"loading lora '{l}'")
        DIFFUSER.load_lora(l)

    logging.info("loading embeddings")
    for i, e in enumerate(embeddings):
        progress(progress=0.3 + 0.05 * i / len(embeddings), desc=f"loading embeddings '{e}'")
        DIFFUSER.load_embeddings(e)

    logging.info(f"defining scheduler")
    progress(progress=0.35, desc=f"loading scheduler")
    DIFFUSER.set_scheduler(scheduler)

    logging.info(f"optimizing prompt")
    progress(progress=0.4, desc="optimizing prompt")
    try:
        optimized_prompt = optimize(prompt, model="sd1", project=project)
    except Exception as error:
        message = f"Error (prompt optim): {error}"
        logging.error(message)
        raise gr.Error(message)
    logging.info(f"optimized prompt: {optimized_prompt}")

    logging.info(f"generating image")
    progress(progress=0.6, desc="generating image")
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

        with gr.Accordion(label="Parameters", open=False):
            with gr.Row():
                checkpoint = gr.Dropdown(label="Diffuser", choices=models.get("checkpoints"), value=checkpoint,
                                         multiselect=False, scale=2)
                loras = gr.Dropdown(label="LoRAs", choices=models.get("loras"), value=[], multiselect=True, scale=2)
                embeddings = gr.Dropdown(label="Embeddings", choices=models.get("embeddings"), value=[],
                                         multiselect=True, scale=2)
            project = gr.Text(
                label="Project",
                placeholder="describe your project here",
                interactive=True,
                container=True,
                lines=2
            )
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                image = gr.Image(
                    label="Image",
                    format="png",
                    type="pil",
                    container=True,
                    height=1024,
                    width=1024
                )
                with gr.Row(equal_height=True):
                    prompt = gr.Text(
                        label="Prompt",
                        placeholder="describe what you want to draw",
                        lines=3,
                        interactive=True,
                        container=True,
                        scale=3
                    )
                    generate_btn = gr.Button(value="Run", variant="primary", scale=1)
            with gr.Column(scale=1, variant="panel"):
                steps = gr.Slider(label="# steps", minimum=1, maximum=50, value=15, step=1)
                guidance = gr.Slider(label="guidance", minimum=1, maximum=20, value=7, step=0.5)
                aspect = gr.Dropdown(label="Aspect", choices=["square", "portrait", "landscape"], value="square")
                scheduler = gr.Dropdown(label="Scheduler", choices=DIFFUSER.get_supported_schedulers(), value="euler")
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    placeholder="describe what you don't want to draw",
                    interactive=True,
                    container=True,
                    lines=5,
                )

        # UI logic
        generate_btn.click(
            fn=generate,
            inputs=[checkpoint, loras, embeddings, prompt, negative_prompt, project, steps, guidance, aspect,
                    scheduler],
            outputs=[image]
        )
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

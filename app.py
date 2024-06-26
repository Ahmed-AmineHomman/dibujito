import logging
import os
from argparse import ArgumentParser, Namespace
from typing import List, Dict

import gradio as gr
from PIL import Image

from api import configure_logger, AppConfig, DEFAULT_CHECKPOINT_NAME
from api.diffuser import Diffuser
from api.optimizers import PromptOptimizer

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

PATHS: Dict[str, str]
DIFFUSER: Diffuser
OPTIMIZER: PromptOptimizer


def load_parameters() -> Namespace:
    """Loads the parameters from the environment."""
    parser = ArgumentParser()
    parser.add_argument(
        "--api",
        type=str,
        required=False,
        choices=["cohere", "ollama"],
        default="cohere",
        help="API providing the LLM used in the app"
    )
    parser.add_argument(
        "--api-model",
        type=str,
        required=False,
        default="command-r-plus",
        help="Model requested when calling the API (note: each API uses its own labels for its underlying models)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=DEFAULT_CHECKPOINT_NAME,
        help="name of the checkpoint file to load during app start."
    )
    parser.add_argument(
        "--logpath",
        type=str,
        required=False,
        help="path to the directory containing the logs. If not provided, logs will be outputted in default stream."
    )
    return parser.parse_args()


def generate(
        checkpoint: str,
        lora: List[str],
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
    if DIFFUSER.get_checkpoint() != checkpoint.split(".")[0]:
        logging.info(f"loading checkpoint")
        DIFFUSER.load_checkpoint(filepath=os.path.join(PATHS.get("checkpoints"), checkpoint))
    else:
        logging.info("reusing pipeline")

    progress(progress=0, desc="loading loras")
    for i, l in enumerate(lora):
        progress(progress=0.2 + 0.1 * i / len(lora), desc=f"loading lora '{l}'")
        DIFFUSER.load_lora(filepath=os.path.join(PATHS.get("loras"), l))

    logging.info(f"defining scheduler")
    progress(progress=0.35, desc=f"loading scheduler")
    DIFFUSER.set_scheduler(scheduler)

    logging.info(f"optimizing prompt")
    progress(progress=0.4, desc="optimizing prompt")
    try:
        optimized_prompt = OPTIMIZER.optimize(description=prompt, model="sd1", project=project)
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


def build_ui() -> gr.Blocks:
    """Builds the UI."""
    # list models
    models = {}
    for subtype in ["checkpoints", "loras"]:
        base_dir = PATHS.get(subtype)
        models[subtype] = [
            f for f in os.listdir(base_dir) if
            (os.path.isfile(os.path.join(base_dir, f)) and (f.split(".")[-1] in ["safetensors", "ckpt", "bin", "pt"]))
        ]
    checkpoint = DIFFUSER.get_checkpoint()

    with gr.Blocks() as ui:
        gr.Markdown(f"# {APP_NAME}\n\n{APP_DESCRIPTION}")

        with gr.Accordion(label="Project", open=False):
            project = gr.Text(
                label="Project",
                placeholder="describe your project here",
                interactive=True,
                container=True,
                lines=2,
                scale=3
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
                checkpoint = gr.Dropdown(label="Diffuser", choices=models.get("checkpoints"), value=checkpoint,
                                         multiselect=False)
                loras = gr.Dropdown(label="LoRAs", choices=models.get("loras"), value=[], multiselect=True)
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
            inputs=[checkpoint, loras, prompt, negative_prompt, project, steps, guidance, aspect, scheduler],
            outputs=[image]
        )
    return ui


if __name__ == "__main__":
    parameters = load_parameters()
    configure_logger(logpath=parameters.logpath if parameters.logpath else None)

    # load configuration
    logging.info("loading config")
    if not os.path.exists("config.json"):
        logging.warning("cannot find configuration file -> continuing with default values")
        config = AppConfig()
    else:
        config = AppConfig.load(filepath="config.json")

    # setting API keys & Hosts
    logging.info("setting up environment variables")
    for key, value in config.environment.items():
        os.environ[key] = value

    logging.info("setting up model paths")
    PATHS = {**config.paths}

    logging.info(f"loading diffuser")
    DIFFUSER = Diffuser(checkpoint_path=os.path.join(config.paths.get("checkpoints"), parameters.checkpoint))

    logging.info("loading prompt optimizer")
    OPTIMIZER = PromptOptimizer(api=parameters.api, api_model=parameters.api_model)

    logging.info(f"building UI")
    app = build_ui()

    logging.info("running app")
    app.launch()

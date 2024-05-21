import logging
from argparse import ArgumentParser, Namespace

import gradio as gr
from PIL import Image

from api import configure_logger
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
        help="path to the logfile. If not provided, logs will only be written in the default stream."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=False,
        help="path to the directory containing the checkpoints."
    )
    return parser.parse_args()


def generate(
        prompt: str,
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
        image = DIFFUSER.imagine(prompt=optimized_prompt, steps=steps, guidance=guidance, aspect=aspect)
    except Exception as error:
        message = f"Error (image gen): {error}"
        logging.error(message)
        raise gr.Error(message)
    logging.info(f"image generated")

    return image


def build_ui() -> gr.Blocks:
    """Builds the UI."""
    with gr.Blocks() as app:
        gr.Markdown(f"# {APP_NAME}\n\n{APP_DESCRIPTION}")

        # parameter section
        with gr.Accordion(label="Parameters", open=False):
            with gr.Row():
                steps = gr.Slider(label="# steps", minimum=1, maximum=50, value=15, step=1)
                guidance = gr.Slider(label="guidance", minimum=1, maximum=20, value=7, step=0.5)
                aspect = gr.Dropdown(label="Aspect", choices=["square", "portrait", "landscape"], value="square")
        project = gr.Text(label="Project", placeholder="describe your project here", interactive=True)

        # generated image
        image = gr.Image(label="Image", format="png")

        # prompting section
        with gr.Row():
            prompt = gr.TextArea(
                label="Prompt",
                placeholder="describe whatever you can think of here",
                scale=4,
            )
            button = gr.Button(value="Run", variant="primary", scale=1)
        button.click(fn=generate, inputs=[prompt, project, steps, guidance, aspect], outputs=[image])
    return app


if __name__ == "__main__":
    parameters = load_parameters()
    configure_logger(logpath=parameters.logpath if parameters.logpath else None)

    logging.info(f"loading diffuser model")
    DIFFUSER = Diffuser()

    logging.info(f"building UI")
    app = build_ui()

    logging.info("running app")
    app.launch()

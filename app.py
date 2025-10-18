import logging
import os
import os.path
import tomllib
from argparse import ArgumentParser, Namespace
from logging import getLogger
from tomllib import load
from typing import Optional

import gradio as gr
from click import MissingParameter

from api import get_supported_image_ratios, LLM, Diffuser
from app_api import generate_image, generate_prompt, load_model, get_model_list


def load_parameters() -> Namespace:
    """Loads the parameters from the environment."""
    parser = ArgumentParser()
    parser.add_argument(
        "--language",
        choices=["en"],
        required=False,
        default="en",
        help="Language of the app."
    )
    parser.add_argument(
        "--logpath",
        type=str,
        required=False,
        help="path to the directory containing the logs. If not provided, logs will be outputted in default stream."
    )
    return parser.parse_args()


def build_ui(
        doc: dict,
        diffuser_directory: str,
        llm_directory: str,
        rules_directory: str,
) -> gr.Blocks:
    """Builds the UI."""
    available_llms = get_model_list(directory=llm_directory, model_type="llm")
    available_diffusers = get_model_list(directory=diffuser_directory, model_type="diffuser")
    available_optimizers = get_model_list(directory=rules_directory, model_type="optimizer")

    with gr.Blocks() as app:
        gr.Markdown(f"# {doc.get('title')}\n\n{doc.get('description')}")

        gr.Markdown(f"## {doc.get('project_title')}\n\n{doc.get('project_description')}")
        project = gr.Text(
            label=None,
            interactive=True,
            container=False,
            placeholder=doc.get("project_placeholder"),
        )

        gr.Markdown(f"## {doc.get('generation_title')}\n\n{doc.get('generation_description')}")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, variant="default"):
                image = gr.Image(
                    label=doc.get("image_label"),
                    format="png",
                    type="pil",
                    container=False,
                )
                with gr.Row(equal_height=True):
                    prompt = gr.Text(
                        label=doc.get("diffuser_prompt_label"),
                        placeholder=doc.get("diffuser_prompt_placeholder"),
                        interactive=True,
                        container=False,
                        lines=5,
                        scale=3
                    )
                    with gr.Column(scale=1):
                        improve_btn = gr.Button(
                            value=doc.get("improve_button"),
                            variant="secondary",
                            scale=1,
                        )
                        generate_btn = gr.Button(
                            value=doc.get("generate_button"),
                            variant="primary",
                            scale=1,
                        )
            with gr.Column(scale=2, variant="panel"):
                with gr.Tab("LLM"):
                    llm = gr.Dropdown(
                        label=doc.get("parameter_llm_label"),
                        info=doc.get("parameter_llm_description"),
                        choices=available_llms,
                        value=available_llms[0],
                        multiselect=False
                    )
                    prompting_rules = gr.Dropdown(
                        label=doc.get("parameter_rules_label"),
                        info=doc.get("parameter_rules_description"),
                        choices=available_optimizers,
                        value=available_optimizers[0],
                        multiselect=False
                    )
                    llm_temperature = gr.Slider(
                        label=doc.get("parameter_llm_temperature_label"),
                        info=doc.get("parameter_llm_temperature_description"),
                        value=0.2,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                    )
                    llm_seed = gr.Number(
                        label=doc.get("parameter_llm_seed_label"),
                        info=doc.get("parameter_llm_seed_description"),
                        value=-1,
                        minimum=-1,
                        maximum=1000000000,
                        step=1,
                    )
                with gr.Tab("Diffuser"):
                    diffuser = gr.Dropdown(
                        label=doc.get("parameter_diffuser_label"),
                        info=doc.get("parameter_diffuser_description"),
                        choices=available_diffusers,
                        value=available_diffusers[0],
                        multiselect=False
                    )
                    negative_prompt = gr.TextArea(
                        label=doc.get("parameter_negative_prompt_label"),
                        info=doc.get("parameter_negative_prompt_description"),
                        interactive=True,
                        container=True,
                        lines=1,
                        max_lines=3
                    )
                    steps = gr.Slider(
                        label=doc.get("parameter_steps_label"),
                        info=doc.get("parameter_steps_description"),
                        minimum=1,
                        maximum=50,
                        value=25,
                        step=1
                    )
                    preview_frequency = gr.Slider(
                        label="Preview every N steps",
                        info="Set to 0 to disable intermediate preview images.",
                        minimum=0,
                        maximum=50,
                        value=0,
                        step=1,
                    )
                    guidance = gr.Slider(
                        label=doc.get("parameter_guidance_label"),
                        info=doc.get("parameter_guidance_description"),
                        minimum=1,
                        maximum=15,
                        value=7,
                        step=0.5
                    )
                    aspect = gr.Dropdown(
                        label=doc.get("parameter_aspect_label"),
                        info=doc.get("parameter_aspect_description"),
                        choices=get_supported_image_ratios(),
                        value=get_supported_image_ratios()[0],
                    )
                    diffuser_seed = gr.Number(
                        label=doc.get("parameter_seed_label"),
                        info=doc.get("parameter_seed_description"),
                        value=-1,
                        minimum=-1,
                        maximum=1000000000,
                        step=1,
                    )
        diffuser_dir = gr.State(diffuser_directory)
        llm_dir = gr.State(llm_directory)
        rules_dir = gr.State(rules_directory)

        # UI logic
        improve_btn.click(
            fn=generate_prompt,
            inputs=[prompt, llm, llm_dir, prompting_rules, rules_dir, project, llm_temperature, llm_seed],
            outputs=[prompt]
        )
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, diffuser, diffuser_dir, negative_prompt, steps, guidance, preview_frequency, aspect,
                    diffuser_seed],
            outputs=[image]
        )
    return app


def check_configuration(config: dict) -> None:
    if not config.get("llm").get("directory"):
        raise MissingParameter("cannot find llm model directory in the configuration file")
    if not config.get("diffuser").get("directory"):
        raise MissingParameter("cannot find diffusers model directory in the configuration file")


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


def get_ui_config() -> dict:
    directory = "."
    config_path = "config.toml"
    default_path = "config_example.toml"
    try:
        if not os.path.exists(os.path.join(directory, config_path)):
            raise FileNotFoundError("cannot find 'config.toml' in the root deposit")
        with open(os.path.join(directory, config_path), "rb") as fh:
            config = load(fh)
        check_configuration(config)
    except Exception as e:
        logging.error(e)
        logging.warning("falling back to default configuration")
        with open(os.path.join(directory, default_path), "rb") as fh:
            config = load(fh)
    return config


if __name__ == "__main__":
    parameters = load_parameters()
    configure_logger(filepath=parameters.logpath if parameters.logpath else None)

    logging.info(f"loading UI locale")
    doc = get_ui_doc(language=parameters.language)

    logging.info("loading config")
    config = get_ui_config()

    logging.info("loading models")
    load_model(llm=LLM(), diffuser=Diffuser())

    logging.info(f"building UI")
    app = build_ui(
        doc=doc,
        llm_directory=config.get("llm").get("directory"),
        diffuser_directory=config.get("diffuser").get("directory"),
        rules_directory=config.get("prompting_rules").get("directory")
    )

    logging.info("running app")
    app.launch()

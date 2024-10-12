import logging
import os.path
from argparse import ArgumentParser, Namespace
from tomllib import load
from warnings import warn

import gradio as gr
from click import MissingParameter

from api import configure_logger, get_ui_doc, get_supported_optimizers, get_supported_image_ratios, LLM, Diffuser
from api.clients import APIClientFactory
from app_api import generate_image, load_model, get_model_list


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


def check_configuration(config: dict) -> None:
    if not config.get("llm").get("model_dir"):
        raise MissingParameter("cannot find llm model directory in the configuration file")
    if not config.get("diffuser").get("model_dir"):
        raise MissingParameter("cannot find diffusers model directory in the configuration file")


def build_ui(
        doc: dict,
        diffuser_directory: str,
        llm_directory: str
) -> gr.Blocks:
    """Builds the UI."""
    available_llms = get_model_list(directory=llm_directory, model_type="llm")
    available_diffusers = get_model_list(directory=diffuser_directory, model_type="diffusers")

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
        with gr.Row(equal_height=True):
            with gr.Column(scale=3, variant="default"):
                with gr.Row(equal_height=True):
                    prompt = gr.TextArea(
                        label=doc.get('prompt_label'),
                        placeholder=doc.get("prompt_placeholder"),
                        interactive=True,
                        container=False,
                        lines=2,
                        max_lines=3,
                        scale=3
                    )
                    generate_image_btn = gr.Button(
                        value=doc.get("generate_image_button"),
                        variant="primary",
                        scale=1,
                    )
                optimized_prompt = gr.TextArea(
                    label=doc.get("optimized_prompt_label"),
                    placeholder=doc.get("optimized_prompt_placeholder"),
                    interactive=False,
                    container=False,
                    lines=2,
                    max_lines=3,
                )
                image = gr.Image(
                    label=doc.get("image_label"),
                    format="png",
                    type="pil",
                    container=True,
                    height=1024,
                    width=1024,
                )
            with gr.Column(scale=1, variant="default"):
                llm = gr.Dropdown(
                    label=doc.get("parameter_llm_label"),
                    info=doc.get("parameter_llm_description"),
                    choices=available_llms,
                    multiselect=False
                )
                diffuser = gr.Dropdown(
                    label=doc.get("parameter_diffuser_label"),
                    info=doc.get("parameter_diffuser_description"),
                    choices=available_diffusers,
                    multiselect=False
                )
                expand_prompt = gr.Checkbox(
                    value=True,
                    label=doc.get("prompt_expansion_label"),
                    info=doc.get("prompt_expansion_description")
                )
                optimize_prompt = gr.Checkbox(
                    value=True,
                    label=doc.get("prompt_optimization_label"),
                    info=doc.get("prompt_optimization_description")
                )
                optimization_target = gr.Dropdown(
                    label=doc.get("parameter_optimization_target_label"),
                    info=doc.get("parameter_optimization_target_description"),
                    choices=get_supported_optimizers(),
                    value=get_supported_optimizers()[0],
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
                seed = gr.Number(
                    label=doc.get("parameter_seed_label"),
                    info=doc.get("parameter_seed_description"),
                    value=None,
                    minimum=0,
                    maximum=1000000000,
                    step=1,
                )
        diffuser_dir = gr.State(diffuser_directory)
        llm_dir = gr.State(llm_directory)

        # UI logic
        generate_image_btn.click(
            fn=generate_image,
            inputs=[
                diffuser, diffuser_dir, prompt, negative_prompt, steps, guidance, aspect, seed,
                llm, llm_dir, expand_prompt, optimize_prompt, optimization_target, project
            ],
            outputs=[image, optimized_prompt]
        )
    return app


if __name__ == "__main__":
    parameters = load_parameters()
    configure_logger(filepath=parameters.logpath if parameters.logpath else None)

    logging.info(f"loading UI documentation")
    ui_doc = get_ui_doc(language=parameters.language)

    logging.info("loading config")
    if not os.path.exists("config.toml"):
        message = "cannot find 'config.toml' configuration file in the root deposit"
        logging.error(message)
        raise Exception(message)
    with open("config.toml", "rb") as fh:
        config = load(fh)
    try:
        check_configuration(config)
    except Exception as e:
        logging.error(e)
        raise e

    logging.info("loading models")
    load_model(llm=LLM(), diffuser=Diffuser())

    logging.info(f"building UI")
    app = build_ui(
        doc=ui_doc,
        llm_directory=config.get("llm").get("model_dir"),
        diffuser_directory=config.get("diffuser").get("model_dir"),
    )

    logging.info("running app")
    app.launch()

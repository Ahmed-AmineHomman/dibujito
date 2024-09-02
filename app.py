import logging
from argparse import ArgumentParser, Namespace

import gradio as gr

from api import configure_logger, get_ui_doc, get_supported_llms, get_supported_diffusers, get_supported_optimizers, \
    get_supported_image_ratios
from app_api import generate_image


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
) -> gr.Blocks:
    """Builds the UI."""
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
                    choices=get_supported_llms(),
                    value=get_supported_llms()[0],
                    multiselect=False
                )
                diffuser = gr.Dropdown(
                    label=doc.get("parameter_diffuser_label"),
                    info=doc.get("parameter_diffuser_description"),
                    choices=get_supported_diffusers(),
                    value=get_supported_diffusers()[0],
                    multiselect=False
                )
                optimization_level = gr.Dropdown(
                    label=doc.get("parameter_optimization_level_label"),
                    info=doc.get("parameter_optimization_level_description"),
                    choices=["none", "light", "strong"],
                    value="strong",
                    multiselect=False
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

        # UI logic
        generate_image_btn.click(
            fn=generate_image,
            inputs=[
                diffuser, prompt, negative_prompt, steps, guidance, aspect, seed,
                llm, optimization_level, optimization_target, project
            ],
            outputs=[image, optimized_prompt]
        )
    return app


if __name__ == "__main__":
    parameters = load_parameters()
    configure_logger(filepath=parameters.logpath if parameters.logpath else None)

    logging.info(f"loading UI documentation")
    ui_doc = get_ui_doc(language=parameters.language)

    logging.info(f"building UI")
    app = build_ui(
        doc=ui_doc,
    )

    logging.info("running app")
    app.launch()

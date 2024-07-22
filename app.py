import logging
from argparse import ArgumentParser, Namespace

import gradio as gr

from api import configure_logger, get_ui_doc
from api.diffuser import Diffuser
from api.optimizer import PromptOptimizer
from app_api import generate

DIFFUSER: Diffuser


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
        "--api",
        type=str,
        required=False,
        choices=["cohere", "ollama"],
        default="cohere",
        help="API providing the LLM used in the app"
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

        with gr.Accordion(label=doc.get("project_label"), open=False):
            gr.Markdown(f"## {doc.get('project_title')}\n\n{doc.get('project_description')}")
            project = gr.Text(
                interactive=True,
                container=True,
                lines=2,
            )
        with gr.Row():
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
                    with gr.Column(scale=3):
                        prompt = gr.Text(
                            label=doc.get("negative_prompt_label"),
                            interactive=True,
                            container=True,
                        )
                        negative_prompt = gr.Text(
                            label=doc.get("negative_prompt_label"),
                            interactive=True,
                            container=True,
                        )
                    with gr.Column(scale=1):
                        with gr.Row():
                            gr.Markdown(doc.get("parameter_optimize_prompt_description"))
                            optimize_prompt = gr.Checkbox(
                                label=doc.get("parameter_optimize_prompt_label"),
                                value=True,
                                interactive=True,
                            )
                        generate_btn = gr.Button(
                            value=doc.get("generate_button_label"),
                            variant="primary",
                        )
            with gr.Column(scale=1, variant="panel"):
                with gr.Row():
                    gr.Markdown(doc.get("parameter_diffuser_description"))
                    model = gr.Dropdown(
                        label=doc.get("parameter_diffuser_label"),
                        choices=Diffuser.get_supported_models(),
                        value=Diffuser.get_supported_models()[0],
                        multiselect=False
                    )
                with gr.Row():
                    gr.Markdown(doc.get("parameter_optimizer_description"))
                    target_model = gr.Dropdown(
                        label=doc.get("parameter_optimizer_label"),
                        choices=PromptOptimizer.get_supported_rules(),
                        value=PromptOptimizer.get_supported_rules()[0],
                        multiselect=False
                    )
                with gr.Row():
                    gr.Markdown(doc.get("parameter_steps_description"))
                    steps = gr.Slider(
                        label=doc.get("parameter_steps_label"),
                        minimum=1,
                        maximum=50,
                        value=25,
                        step=1
                    )
                with gr.Row():
                    gr.Markdown(doc.get("parameter_guidance_description"))
                    guidance = gr.Slider(
                        label=doc.get("parameter_guidance_label"),
                        minimum=1,
                        maximum=15,
                        value=7,
                        step=0.5
                    )
                with gr.Row():
                    gr.Markdown(doc.get("parameter_aspect_description"))
                    aspect = gr.Dropdown(
                        label=doc.get("parameter_aspect_label"),
                        choices=Diffuser.get_supported_aspects(),
                        value=Diffuser.get_supported_aspects()[0],
                    )

        diffuser = gr.State(value=DIFFUSER)

        # UI logic
        generate_btn.click(
            fn=generate,
            inputs=[model, prompt, negative_prompt, steps, guidance, aspect, project, optimize_prompt, target_model,
                    diffuser],
            outputs=[image, diffuser]
        )
    return app


if __name__ == "__main__":
    parameters = load_parameters()
    configure_logger(filepath=parameters.logpath if parameters.logpath else None)

    logging.info(f"loading models")
    DIFFUSER = Diffuser()

    logging.info(f"loading UI documentation")
    ui_doc = get_ui_doc(language=parameters.language)

    logging.info(f"building UI")
    app = build_ui(doc=ui_doc)

    logging.info("running app")
    app.launch()

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
from click import MissingParameter
from tomllib import load as load_toml

from api import Diffuser, LLM, get_supported_image_ratios
from app_api import generate_image, generate_prompt, get_model_list, load_model

LANGUAGES = ("en",)
DEFAULT_LANGUAGE = "en"
CONFIG_PATH = Path("config.toml")
EXAMPLE_CONFIG_PATH = Path("config_example.toml")
LOCALES_DIRECTORY = Path(__file__).parent / "data" / "locales"


@dataclass
class PromptSection:
    image: gr.Image
    prompt: gr.Textbox
    negative_prompt: gr.Textbox
    improve_button: gr.Button
    generate_button: gr.Button


@dataclass
class ModelSection:
    llm: gr.Dropdown
    rules: gr.Dropdown
    temperature: gr.Slider
    seed: gr.Number
    diffuser: gr.Dropdown
    steps: gr.Slider
    guidance: gr.Slider
    diffuser_seed: gr.Number
    preview_frequency: gr.Slider
    preview_method: gr.Radio
    aspect: gr.Dropdown


def parse_cli_args() -> Namespace:
    """Parse command-line arguments supplied by the user."""
    parser = ArgumentParser()
    parser.add_argument(
        "--language",
        choices=LANGUAGES,
        default=DEFAULT_LANGUAGE,
        help="Language of the UI copy.",
    )
    parser.add_argument(
        "--logpath",
        type=str,
        help="Optional path to a log file. Defaults to stderr.",
    )
    return parser.parse_args()


def build_ui(doc: Dict[str, Any], diffuser_directory: str, llm_directory: str, rules_directory: str) -> gr.Blocks:
    """Construct and wire the Gradio UI."""
    available_llms = get_model_list(directory=llm_directory, model_type="llm")
    available_diffusers = get_model_list(directory=diffuser_directory, model_type="diffuser")
    available_optimizers = get_model_list(directory=rules_directory, model_type="optimizer")

    with gr.Blocks() as app:
        _build_header(doc)
        with gr.Row(equal_height=False):
            prompt_section = _build_prompt_section(doc)
            model_section = _build_model_section(doc, available_llms, available_diffusers, available_optimizers)

        diffuser_state = gr.State(diffuser_directory)
        llm_state = gr.State(llm_directory)
        rules_state = gr.State(rules_directory)
        project_state = gr.State("")

        prompt_section.improve_button.click(
            fn=generate_prompt,
            inputs=[
                prompt_section.prompt,
                model_section.llm,
                llm_state,
                model_section.rules,
                rules_state,
                project_state,
                model_section.temperature,
                model_section.seed,
            ],
            outputs=[prompt_section.prompt],
        )
        prompt_section.generate_button.click(
            fn=generate_image,
            inputs=[
                prompt_section.prompt,
                model_section.diffuser,
                diffuser_state,
                prompt_section.negative_prompt,
                model_section.steps,
                model_section.guidance,
                model_section.preview_frequency,
                model_section.preview_method,
                model_section.aspect,
                model_section.diffuser_seed,
            ],
            outputs=[prompt_section.image],
        )
    return app


def _build_header(doc: Dict[str, Any]) -> None:
    gr.Markdown(f"# {doc.get('title', '')}")
    tagline = doc.get("tagline")
    if tagline:
        gr.Markdown(tagline)


def _build_prompt_section(doc: Dict[str, Any]) -> PromptSection:
    with gr.Column(scale=4, min_width=640):
        image = gr.Image(
            label=doc.get("image_label"),
            format="png",
            type="pil",
            container=False,
        )
        prompt = gr.Textbox(
            label=doc.get("prompt_label"),
            placeholder=doc.get("prompt_placeholder"),
            info=doc.get("prompt_info"),
            interactive=True,
            lines=8,
        )
        negative_prompt = gr.Textbox(
            label=doc.get("negative_prompt_label"),
            placeholder=doc.get("negative_prompt_placeholder"),
            info=doc.get("negative_prompt_info"),
            interactive=True,
            lines=3,
        )
        with gr.Row():
            improve_button = gr.Button(
                value=doc.get("improve_button"),
                variant="secondary",
            )
            generate_button = gr.Button(
                value=doc.get("generate_button"),
                variant="primary",
            )
    return PromptSection(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        improve_button=improve_button,
        generate_button=generate_button,
    )


def _build_model_section(
    doc: Dict[str, Any],
    available_llms: list[str],
    available_diffusers: list[str],
    available_optimizers: list[str],
) -> ModelSection:
    aspect_choices = get_supported_image_ratios()
    with gr.Column(scale=2, min_width=320, variant="panel"):
        gr.Markdown(f"### {doc.get('control_panel_title')}")
        with gr.Accordion(doc.get("optimizer_section_title"), open=True):
            llm = gr.Dropdown(
                label=doc.get("parameter_llm_label"),
                info=doc.get("parameter_llm_description"),
                choices=available_llms,
                value=_default_choice(available_llms),
                multiselect=False,
            )
            prompting_rules = gr.Dropdown(
                label=doc.get("parameter_rules_label"),
                info=doc.get("parameter_rules_description"),
                choices=available_optimizers,
                value=_default_choice(available_optimizers),
                multiselect=False,
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
                maximum=1_000_000_000,
                step=1,
            )
        with gr.Accordion(doc.get("diffuser_section_title"), open=True):
            diffuser = gr.Dropdown(
                label=doc.get("parameter_diffuser_label"),
                info=doc.get("parameter_diffuser_description"),
                choices=available_diffusers,
                value=_default_choice(available_diffusers),
                multiselect=False,
            )
            steps = gr.Slider(
                label=doc.get("parameter_steps_label"),
                info=doc.get("parameter_steps_description"),
                minimum=1,
                maximum=50,
                value=25,
                step=1,
            )
            guidance = gr.Slider(
                label=doc.get("parameter_guidance_label"),
                info=doc.get("parameter_guidance_description"),
                minimum=1,
                maximum=15,
                value=7,
                step=0.5,
            )
            diffuser_seed = gr.Number(
                label=doc.get("parameter_seed_label"),
                info=doc.get("parameter_seed_description"),
                value=-1,
                minimum=-1,
                maximum=1_000_000_000,
                step=1,
            )
        with gr.Accordion(doc.get("app_section_title"), open=True):
            preview_frequency = gr.Slider(
                label=doc.get("preview_frequency_label"),
                info=doc.get("preview_frequency_info"),
                minimum=0,
                maximum=50,
                value=doc.get("preview_frequency_default", 0),
                step=1,
            )
            preview_method = gr.Radio(
                label=doc.get("preview_method_label"),
                info=doc.get("preview_method_info"),
                choices=["fast", "medium", "full"],
                value="fast",
            )
            aspect = gr.Dropdown(
                label=doc.get("parameter_aspect_label"),
                info=doc.get("parameter_aspect_description"),
                choices=aspect_choices,
                value=_default_choice(aspect_choices),
            )
    return ModelSection(
        llm=llm,
        rules=prompting_rules,
        temperature=llm_temperature,
        seed=llm_seed,
        diffuser=diffuser,
        steps=steps,
        guidance=guidance,
        diffuser_seed=diffuser_seed,
        preview_frequency=preview_frequency,
        preview_method=preview_method,
        aspect=aspect,
    )


def _default_choice(options: list[str]) -> Optional[str]:
    return options[0] if options else None


def validate_configuration(config: Dict[str, Any]) -> None:
    """Ensure the runtime configuration contains the directories we need."""
    required_sections = ("llm", "diffuser", "prompting_rules")
    for section in required_sections:
        directory = config.get(section, {}).get("directory")
        if not directory:
            raise MissingParameter(f"cannot find {section} directory in the configuration file")


def configure_logger(filepath: Optional[str] = None) -> None:
    """Configure the root logger for both console and optional file targets."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    if filepath:
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)


def load_ui_copy(language: str) -> Dict[str, Any]:
    """Load the UI translation corresponding to ``language``."""
    locale_path = LOCALES_DIRECTORY / f"{language}.toml"
    if not locale_path.exists():
        logging.warning("Unsupported language '%s'; defaulting to '%s'.", language, DEFAULT_LANGUAGE)
        locale_path = LOCALES_DIRECTORY / f"{DEFAULT_LANGUAGE}.toml"
    return _read_toml(locale_path)


def load_configuration() -> Dict[str, Any]:
    """Load the runtime configuration, falling back to the example file if needed."""
    for candidate, is_fallback in ((CONFIG_PATH, False), (EXAMPLE_CONFIG_PATH, True)):
        try:
            config = _read_toml(candidate)
            validate_configuration(config)
            if is_fallback:
                logging.warning("Using fallback configuration from '%s'.", candidate)
            return config
        except FileNotFoundError:
            if not is_fallback:
                logging.error("Cannot find configuration file at '%s'.", candidate)
        except Exception as error:
            logging.error("Unable to load configuration from '%s': %s", candidate, error)
            if not is_fallback:
                logging.warning("Falling back to example configuration.")
    raise RuntimeError("Failed to load any application configuration.")


def _read_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return load_toml(handle)


def main() -> None:
    parameters = parse_cli_args()
    configure_logger(parameters.logpath)

    logging.info("loading UI locale")
    doc = load_ui_copy(language=parameters.language)

    logging.info("loading configuration")
    config = load_configuration()

    logging.info("loading models")
    load_model(llm=LLM(), diffuser=Diffuser())

    logging.info("building UI")
    app = build_ui(
        doc=doc,
        llm_directory=config["llm"]["directory"],
        diffuser_directory=config["diffuser"]["directory"],
        rules_directory=config["prompting_rules"]["directory"],
    )

    logging.info("running app")
    app.launch()


if __name__ == "__main__":
    main()

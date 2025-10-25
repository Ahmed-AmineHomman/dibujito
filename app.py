from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from tomllib import load as load_toml
from typing import Any, Dict, Iterator, Optional

import gradio as gr
from click import MissingParameter

from api import Diffuser, LLM, get_supported_image_ratios
from app_api import generate_image, get_model_list, load_model, optimize_prompt

LANGUAGES = ("en",)
DEFAULT_LANGUAGE = "en"
CONFIG_PATH = Path("config.toml")
EXAMPLE_CONFIG_PATH = Path("config_example.toml")
LOCALES_DIRECTORY = Path(__file__).parent / "data" / "locales"


@dataclass
class GenerationPanel:
    image: gr.Image
    positive_prompt: gr.Textbox
    negative_prompt: gr.Textbox


@dataclass
class ControlPanel:
    llm: gr.Dropdown
    optimizer: gr.Dropdown
    llm_temperature: gr.Slider
    llm_seed: gr.Number
    diffuser: gr.Dropdown
    diffuser_steps: gr.Slider
    diffuser_guidance: gr.Slider
    diffuser_seed: gr.Number
    preview_frequency: gr.Slider
    preview_method: gr.Radio
    aspect: gr.Dropdown
    chatbot: gr.Chatbot
    chatbot_input: gr.Textbox


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


def build_ui(
        doc: Dict[str, Any],
        diffuser_directory: str,
        llm_directory: str,
        rules_directory: str
) -> gr.Blocks:
    """Construct and wire the Gradio UI."""
    available_llms = get_model_list(directory=llm_directory, model_type="llm")
    available_diffusers = get_model_list(directory=diffuser_directory, model_type="diffuser")
    available_optimizers = get_model_list(directory=rules_directory, model_type="optimizer")

    with gr.Blocks() as app:
        _build_header(doc)
        with gr.Row(elem_id="main-layout", equal_height=True):
            generation_panel = _build_generation_panel(doc)
            control_panel = _build_control_panel(doc, available_llms, available_diffusers, available_optimizers)

        diffuser_state = gr.State(diffuser_directory)
        llm_state = gr.State(llm_directory)
        rules_state = gr.State(rules_directory)

        image_inputs = [
            generation_panel.positive_prompt,
            control_panel.diffuser,
            diffuser_state,
            generation_panel.negative_prompt,
            control_panel.diffuser_steps,
            control_panel.diffuser_guidance,
            control_panel.preview_frequency,
            control_panel.preview_method,
            control_panel.aspect,
            control_panel.diffuser_seed,
        ]
        generation_panel.positive_prompt.submit(
            fn=generate_image,
            inputs=image_inputs,
            outputs=[generation_panel.image],
        )
        generation_panel.negative_prompt.submit(
            fn=generate_image,
            inputs=image_inputs,
            outputs=[generation_panel.image],
        )
        control_panel.chatbot_input.submit(
            fn=_assistant_chat_response,
            inputs=[
                control_panel.chatbot,
                control_panel.chatbot_input,
                control_panel.llm,
                llm_state,
                control_panel.llm_temperature,
                control_panel.llm_seed,
                control_panel.optimizer,
                rules_state,
            ],
            outputs=[control_panel.chatbot, control_panel.chatbot_input],
        )
    return app


def _build_header(doc: Dict[str, Any]) -> None:
    gr.Markdown(f"# {doc.get("header").get('title')}")
    if doc.get("header").get("tagline"):
        gr.Markdown(doc.get("header").get("tagline"))


def _build_generation_panel(doc: Dict[str, Any]) -> GenerationPanel:
    with gr.Column(scale=3, min_width=520):
        image = gr.Image(
            label=doc.get("generation").get("image").get("label"),
            format="png",
            type="pil",
            container=False,
        )
        with gr.Group():
            positive_prompt = gr.Textbox(
                label=doc.get("generation").get("positive_prompt").get("label"),
                placeholder=doc.get("generation").get("positive_prompt").get("placeholder"),
                info=doc.get("generation").get("positive_prompt").get("info"),
                interactive=True,
                lines=6,
            )
            negative_prompt = gr.Textbox(
                label=doc.get("generation").get("negative_prompt").get("label"),
                placeholder=doc.get("generation").get("negative_prompt").get("placeholder"),
                info=doc.get("generation").get("negative_prompt").get("info"),
                interactive=True,
                lines=2,
            )
    return GenerationPanel(
        image=image,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
    )


def _build_control_panel(
        doc: Dict[str, Any],
        available_llms: list[str],
        available_diffusers: list[str],
        available_optimizers: list[str],
) -> ControlPanel:
    aspect_choices = get_supported_image_ratios()
    with gr.Column(min_width=0, scale=1):
        with gr.Tabs():
            with gr.Tab(doc.get("control").get("llm").get("title")):
                doc_current = doc.get("control").get("llm")
                if doc_current.get("tagline"):
                    gr.Markdown(doc_current.get("tagline"))
                llm = gr.Dropdown(
                    label=doc_current.get("llm").get("label"),
                    info=doc_current.get("llm").get("info"),
                    choices=available_llms,
                    value=_default_choice(available_llms),
                    multiselect=False,
                )
                llm_temperature = gr.Slider(
                    label=doc_current.get("temperature").get("label"),
                    info=doc_current.get("temperature").get("info"),
                    value=0.2,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                )
                llm_seed = gr.Number(
                    label=doc_current.get("seed").get("label"),
                    info=doc_current.get("seed").get("info"),
                    value=-1,
                    minimum=-1,
                    maximum=1_000_000_000,
                    step=1,
                )
            with gr.Tab(doc.get("control").get("diffuser").get("title")):
                doc_current = doc.get("control").get("diffuser")
                if doc_current.get("tagline"):
                    gr.Markdown(doc_current.get("tagline"))
                diffuser = gr.Dropdown(
                    label=doc_current.get("diffuser").get("label"),
                    info=doc_current.get("diffuser").get("info"),
                    choices=available_diffusers,
                    value=_default_choice(available_diffusers),
                    multiselect=False,
                )
                steps = gr.Slider(
                    label=doc_current.get("steps").get("label"),
                    info=doc_current.get("steps").get("info"),
                    minimum=1,
                    maximum=50,
                    value=28,
                    step=1,
                )
                guidance = gr.Slider(
                    label=doc_current.get("guidance").get("label"),
                    info=doc_current.get("guidance").get("info"),
                    minimum=1,
                    maximum=15,
                    value=5.0,
                    step=0.1,
                )
                diffuser_seed = gr.Number(
                    label=doc_current.get("seed").get("label"),
                    info=doc_current.get("seed").get("info"),
                    value=-1,
                    minimum=-1,
                    step=1,
                )
            with gr.Tab(doc.get("control").get("app").get("title")):
                doc_current = doc.get("control").get("app")
                if doc_current.get("tagline"):
                    gr.Markdown(doc_current.get("tagline"))
                aspect = gr.Dropdown(
                    label=doc_current.get("aspect").get("label"),
                    info=doc_current.get("aspect").get("info"),
                    choices=aspect_choices,
                    value=_default_choice(aspect_choices),
                )
                optimizer = gr.Dropdown(
                    label=doc_current.get("optimizer").get("label"),
                    info=doc_current.get("optimizer").get("info"),
                    choices=available_optimizers,
                    value=_default_choice(available_optimizers),
                    multiselect=False,
                )
                preview_frequency = gr.Slider(
                    label=doc_current.get("preview_frequency").get("label"),
                    info=doc_current.get("preview_frequency").get("info"),
                    minimum=0,
                    maximum=50,
                    value=0,
                    step=1,
                )
                preview_method = gr.Radio(
                    label=doc_current.get("preview_method").get("label"),
                    info=doc_current.get("preview_method").get("info"),
                    choices=["fast", "medium", "full"],
                    value="fast",
                )
            with gr.Tab(doc.get("control").get("chatbot").get("title")):
                doc_current = doc.get("control").get("chatbot")
                if doc_current.get("tagline"):
                    gr.Markdown(doc_current.get("tagline"))
                chatbot = gr.Chatbot(
                    label=doc_current.get("chatbot").get("label"),
                    value=[],
                    height=440,
                    type="messages",
                )
                input_box = gr.Textbox(
                    show_label=False,
                    placeholder=doc_current.get("input").get("placeholder"),
                    interactive=True,
                    lines=1,
                    scale=4,
                )

    return ControlPanel(
        llm=llm,
        llm_temperature=llm_temperature,
        llm_seed=llm_seed,
        diffuser=diffuser,
        diffuser_steps=steps,
        diffuser_guidance=guidance,
        diffuser_seed=diffuser_seed,
        aspect=aspect,
        optimizer=optimizer,
        preview_frequency=preview_frequency,
        preview_method=preview_method,
        chatbot=chatbot,
        chatbot_input=input_box,
    )


def _assistant_chat_response(
        history: list[dict[str, str]] | None,
        message: str,
        llm_name: Optional[str],
        llm_dir: str,
        temperature: Optional[float],
        seed: Optional[float],
        optimizer_name: Optional[str],
        rules_dir: str,
) -> Iterator[tuple[list[dict[str, str]], str]]:
    """Generate a conversational response containing an optimised prompt suggestion."""
    # consistency checks
    if not message or not message.strip():
        gr.Warning("Received empty message for assistant chat; no action taken.")
        yield history or [], ""
        return
    if not llm_name:
        gr.Error("Select an LLM before requesting prompt help.")
        yield history or [], ""
        return
    if not optimizer_name:
        gr.Error("Select a prompt optimizer to continue.")
        yield history or [], ""
        return

    # preparing inference parameters
    temperature_value = float(temperature) if temperature is not None else 0.5
    seed_value: Optional[int] = None
    if seed is not None:
        try:
            seed_candidate = int(seed)
        except (TypeError, ValueError):
            seed_candidate = -1
        if seed_candidate >= 0:
            seed_value = seed_candidate

    # append user message to conversation history
    conversation: list[dict[str, str]] = list(history or [])
    conversation.append({"role": "user", "content": message})
    yield conversation, ""

    # create snapshot for the model before appending the streaming placeholder
    conversation_for_llm = list(conversation)

    # stream assistant response
    conversation.append({"role": "assistant", "content": ""})
    try:
        stream = optimize_prompt(
            conversation=conversation_for_llm,
            llm=llm_name,
            llm_dir=llm_dir,
            rules=optimizer_name,
            rules_dir=rules_dir,
            temperature=temperature_value,
            seed=seed_value,
            creative_mode=True,
        )

        produced_content = False
        for chunk in stream:
            produced_content = True
            conversation[-1]["content"] += chunk
            yield conversation, ""

        if not produced_content:
            gr.Error("Something went wrong during prompt optimization; no content produced.")
            conversation.pop()
            yield conversation, ""
    except Exception as error:
        logging.exception("assistant prompt optimization failed")
        gr.Error("Something went wrong during prompt optimization; no content produced.")
        conversation.pop()
        yield conversation, ""


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

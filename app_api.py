from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from threading import Thread
from typing import Optional

import gradio as gr
from PIL import Image

from api import Diffuser, LLM, PromptingRules

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistry:
    """Keeps track of the currently loaded backend models."""

    writer: Optional[LLM] = None
    artist: Optional[Diffuser] = None

    def register(
            self,
            llm: Optional[LLM] = None,
            diffuser: Optional[Diffuser] = None
    ) -> None:
        """Register the backend instances that power the UI.

        Parameters
        ----------
        llm
            Loaded large language model used for prompt authoring.
        diffuser
            Diffusion pipeline responsible for image generation.
        """
        if llm is not None:
            self.writer = llm
        if diffuser is not None:
            self.artist = diffuser

    def require_writer(self) -> LLM:
        """Return the previously registered language model.

        Returns
        -------
        LLM
            Language model ready for inference.

        Raises
        ------
        RuntimeError
            Raised when no language model has been registered yet.
        """
        if self.writer is None:
            message = "No LLM registered. Call `load_model` during startup to provide one."
            logger.error(message)
            raise RuntimeError(message)
        return self.writer

    def require_artist(self) -> Diffuser:
        """Return the previously registered diffusion pipeline.

        Returns
        -------
        Diffuser
            Diffusion pipeline ready for inference.

        Raises
        ------
        RuntimeError
            Raised when no diffusion pipeline has been registered yet.
        """
        if self.artist is None:
            message = "No diffuser registered. Call `load_model` during startup to provide one."
            logger.error(message)
            raise RuntimeError(message)
        return self.artist


MODELS = ModelRegistry()


def load_model(
        llm: Optional[LLM] = None,
        diffuser: Optional[Diffuser] = None
) -> None:
    """Register the provided models so downstream helpers can access them at runtime.

    Parameters
    ----------
    llm
        Language model used for prompt authoring.
    diffuser
        Diffusion pipeline used for image synthesis.
    """
    MODELS.register(llm=llm, diffuser=diffuser)


def get_model_list(
        directory: str,
        model_type: str
) -> list[str]:
    """Return the list of supported model files contained in ``directory``.

    Parameters
    ----------
    directory
        Path that should contain models of the requested type.
    model_type
        Kind of model to discover. Supported values are ``llm``, ``diffuser``,
        and ``optimizer``.

    Returns
    -------
    list[str]
        Sorted collection of file names matching the requested type.

    Raises
    ------
    ValueError
        Raised when ``model_type`` is unknown.
    FileNotFoundError
        Raised when ``directory`` does not exist.
    """
    extension_mapper = {
        "llm": ".gguf",
        "diffuser": ".safetensors",
        "optimizer": ".toml",
    }

    if model_type not in extension_mapper:
        message = f"Unsupported model type '{model_type}'."
        logger.error(message)
        raise ValueError(message)

    path = Path(directory)
    if not path.exists():
        message = f"Directory '{directory}' does not exist."
        logger.error(message)
        raise FileNotFoundError(message)

    suffix = extension_mapper[model_type]
    names = sorted(entry.name for entry in path.iterdir() if entry.is_file() and entry.suffix == suffix)
    if not names:
        logger.warning("No '%s' files found in '%s'.", suffix, directory)
    return names


def log(
        message: str,
        message_type: str = "info",
        progress: Optional[float] = 0.0,
        progressbar: Optional[gr.Progress] = None,
        show_in_ui: bool = False,
) -> None:
    """Log a message and optionally surface it in the Gradio UI.

    Parameters
    ----------
    message
        Human-readable message to record.
    message_type
        Logging level to apply. Accepted values are ``info``, ``warning`` and
        ``error``.
    progress
        Floating-point completion indicator forwarded to the Gradio progress
        component when available.
    progressbar
        Optional Gradio progress component to update.
    show_in_ui
        When ``True``, always display the notification inside the UI.

    Raises
    ------
    gr.Error
        Raised when ``message_type`` is unsupported or explicitly set to
        ``error``.
    """
    level_handlers = {
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
    }

    if message_type not in level_handlers:
        error_message = f"unknown message type '{message_type}' (supported are 'info', 'warning' and 'error')"
        logger.error(error_message)
        raise gr.Error(error_message)

    level_handlers[message_type](message)
    _notify_ui(message=message, severity=message_type, force=show_in_ui)

    if progressbar is not None:
        progressbar(progress=progress, desc=message)

    if message_type == "error":
        raise gr.Error(message)


def generate_response(
        history: Optional[list[dict[str, str]]],
        message: str,
        llm_name: Optional[str],
        llm_dir: str,
        temperature: Optional[float],
        seed: Optional[int | float],
        optimizer_name: Optional[str],
        rules_dir: str,
        creative_mode: bool = True,
) -> Iterator[tuple[list[dict[str, str]], str]]:
    """Stream the updated chat history while the assistant reply is generated.

    Parameters
    ----------
    history
        Existing chat messages exchanged between the user and the assistant.
    message
        Latest user input collected from the UI textbox.
    llm_name
        File name of the llama.cpp model selected in the UI.
    llm_dir
        Directory containing available llama.cpp models.
    temperature
        Sampling temperature selected by the user.
    seed
        Pseudo-random seed used to reproduce responses (-1 disables determinism).
    optimizer_name
        Name of the prompting rules file selected in the UI.
    rules_dir
        Directory containing prompting rule definitions.
    creative_mode
        Whether the assistant should invent missing details when generating prompts.

    Returns
    -------
    Iterator[tuple[list[dict[str, str]], str]]
        Updated chat history snapshots and an empty string to clear the textbox.
    """
    conversation: list[dict[str, str]] = list(history or [])

    if not message or not message.strip():
        gr.Warning("Received empty message for assistant chat; no action taken.")
        yield conversation, ""
        return
    if not llm_name:
        gr.Error("Select an LLM before requesting prompt help.")
        yield conversation, ""
        return
    if not optimizer_name:
        gr.Error("Select a prompt optimizer to continue.")
        yield conversation, ""
        return

    # load model and rules
    try:
        writer = MODELS.require_writer()
        _load_writer_model(writer=writer, model_dir=llm_dir, model_name=llm_name)
    except Exception:
        logger.exception("LLM model loading failed")
        gr.Error("Failed to load the selected LLM model; cannot proceed.")
        yield conversation, ""
        return
    try:
        rule_set = _load_prompting_rules(directory=rules_dir, rule_name=optimizer_name)
        writer.configure_prompting(
            rules=rule_set,
            creative_mode=creative_mode,
        )
    except Exception:
        logger.exception("Prompting rules loading failed")
        gr.Error("Failed to load the selected prompting rules; cannot proceed.")
        yield conversation, ""
        return

    # normalize inference parameters
    temperature_value = _normalize_temperature(temperature)
    seed_value = _normalise_seed(seed)

    # update conversation for UI and inference
    conversation.append({"role": "user", "content": message})
    llm_messages = list(conversation)
    assistant_message: dict[str, str] = {"role": "assistant", "content": ""}
    conversation.append(assistant_message)
    yield conversation, ""

    # perform inference
    assistant_reply = ""
    try:
        response_stream = writer.respond(
            messages=llm_messages,
            temperature=float(temperature_value),
            seed=seed_value,
            stream=True,
            agent_mode=True,
        )
        if isinstance(response_stream, str):
            assistant_reply = response_stream.strip()
            assistant_message["content"] = assistant_reply
            if assistant_reply:
                yield conversation, ""
            else:
                gr.Warning("No content produced by the assistant.")
            return

        for fragment in response_stream:
            if not fragment:
                continue
            assistant_reply += fragment
            assistant_message["content"] = assistant_reply
            yield conversation, ""
    except Exception:
        logger.exception("assistant prompt optimization failed")
        gr.Error("Something went wrong during prompt optimization; no content produced.")
        return

    # finalise response
    assistant_reply = assistant_reply.strip()
    assistant_message["content"] = assistant_reply
    if assistant_reply:
        yield conversation, ""
    else:
        gr.Warning("No content produced by the assistant.")

    return conversation, ""


def generate_image(
        prompt: str,
        diffuser: str,
        diffuser_dir: str,
        negative_prompt: Optional[str] = None,
        steps: int = 25,
        guidance: float = 7.0,
        preview_frequency: int = 0,
        preview_method: str = "fast",
        aspect: str = "square",
        seed: Optional[int] = -1,
) -> Iterator[Image.Image]:
    """Generate an image (and optional previews) using the configured diffusion pipeline.

    Parameters
    ----------
    prompt
        Text description of the image to generate.
    diffuser
        File name of the diffusion weights selected by the user.
    diffuser_dir
        Directory containing available diffusion models.
    negative_prompt
        Optional negative prompt supplied by the user.
    steps
        Number of denoising iterations to perform.
    guidance
        Classifier-free guidance scale.
    preview_frequency
        Interval (in steps) for publishing intermediate previews. ``0`` disables
        previews.
    preview_method
        Preview decoding strategy. Must be ``fast``, ``medium`` or ``full``.
    aspect
        Desired aspect ratio identifier.
    seed
        Optional deterministic seed. Set to ``-1`` to disable determinism.

    Yields
    ------
    Iterator[Image.Image]
        Intermediate preview images followed by the final render.
    """
    queue: SimpleQueue[object] = SimpleQueue()
    sentinel = object()

    # load diffuser model
    try:
        artist = MODELS.require_artist()
        _load_diffuser_model(artist=artist, model_dir=diffuser_dir, model_name=diffuser)
    except Exception:
        logger.exception("Diffuser model loading failed")
        log(message="Failed to load the selected diffuser model; cannot proceed.", message_type="error")
        return

    # normalize inference parameters
    preview_frequency = _normalise_preview_frequency(preview_frequency)
    preview_method = _normalise_preview_method(preview_method)
    seed = _normalise_seed(seed)

    # define pipeline runner
    def handle_preview(
            image: Image.Image,
            step: int
    ) -> None:
        queue.put(image)

    def run_pipeline() -> None:
        try:
            final_image = artist.imagine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                aspect=aspect,
                seed=seed,
                preview_frequency=preview_frequency,
                preview_method=preview_method,
                preview_callback=handle_preview,
            )
            queue.put(final_image)
        except Exception as error:  # pragma: no cover - surfaced to the UI
            queue.put(error)
        finally:
            queue.put(sentinel)

    # perform inference
    try:
        Thread(target=run_pipeline, daemon=True).start()
        while True:
            item = queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                log(
                    message=f"error (image generation): {item}",
                    message_type="error",
                )
                continue
            yield item
    except Exception as error:  # pragma: no cover - surfaced to the UI
        log(
            message=f"error (image generation): {error}",
            message_type="error",
        )
    log(message="done")


def _load_writer_model(
        writer: LLM,
        model_dir: str,
        model_name: str
) -> None:
    model_path = Path(model_dir) / model_name
    try:
        writer.load_model(filepath=str(model_path))
    except Exception as error:  # pragma: no cover - surfaced to the UI
        log(message=f"error (llm loading): {error}", message_type="error")


def _load_prompting_rules(
        directory: str,
        rule_name: str
) -> PromptingRules:
    filepath = Path(directory) / rule_name
    try:
        return PromptingRules.from_toml(filepath=str(filepath))
    except Exception as error:  # pragma: no cover - surfaced to the UI
        log(message=f"error (rules loading): {error}", message_type="error")


def _load_diffuser_model(
        artist: Diffuser,
        model_dir: str,
        model_name: str
) -> None:
    model_path = Path(model_dir) / model_name
    try:
        artist.load_model(filepath=str(model_path))
    except Exception as error:  # pragma: no cover - surfaced to the UI
        log(message=f"error (diffuser loading): {error}", message_type="error")


def _normalise_seed(seed: Optional[int | float]) -> Optional[int]:
    if seed is None:
        return None
    try:
        candidate = int(seed)
    except (TypeError, ValueError):
        return None
    return candidate if candidate >= 0 else None


def _normalise_preview_frequency(value: int) -> int:
    try:
        frequency = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, frequency)


def _normalize_temperature(value: Optional[float]) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(0.0, temperature), 1.0)


def _normalise_preview_method(method: Optional[str]) -> str:
    if not method:
        return "fast"
    method = method.lower()
    if method not in {"fast", "medium", "full"}:
        log(message=f"unsupported preview method '{method}', defaulting to 'fast'", message_type="warning")
        return "fast"
    return method


def _notify_ui(
        message: str,
        severity: str,
        force: bool
) -> None:
    """Render a Gradio notification if needed."""
    display_in_ui = force or severity in {"warning", "error"}
    if not display_in_ui:
        return

    component_name = {
        "info": "Info",
        "warning": "Warning",
    }.get(severity, "Info")

    if severity == "error":
        return

    component = getattr(gr, component_name, None)
    if component is not None:
        component(message)  # type: ignore[callable-async]

import logging
import os
from typing import Optional, List

import gradio as gr
from PIL import Image

from api import Diffuser, LLM

WRITER: LLM
ARTIST: Diffuser


def load_model(
        llm: Optional[LLM] = None,
        diffuser: Optional[Diffuser] = None
) -> None:
    if llm:
        _ = globals()
        _["WRITER"] = llm
    if diffuser:
        _ = globals()
        _["ARTIST"] = diffuser


def get_model_list(directory: str, model_type: str) -> List[str]:
    names = []
    extension = ".gguf" if model_type == "llm" else ".safetensors"
    for f in os.listdir(directory):
        if f.endswith(extension):
            names.append(f)
    return names


def log(
        message: str,
        message_type: str = "info",
        progress: Optional[float] = 0.0,
        progressbar: Optional[gr.Progress] = None,
        show_in_ui: bool = False
) -> None:
    """
    Logs the provided message.

    This method will also display the message in the Gradio UI if ``show_in_ui`` is True or if ``message_type`` is
    'warning' or 'error'.

    Parameters
    ----------
    message: str,
        The message to log.
    message_type: str, {"info", "warning", "error"}, defaults "info",
        Severity of the message.
    progress: float, optional, default 0.0,
        Progress of the task corresponding to the message.
    progressbar: gr.Progress, optional, default None,
        If provided, displays the message in the corresponding progress bar.
    show_in_ui: bool, optional, default False,
        If set to ``True``, displays the message in the Gradio UI even if ``message_type`` is set to ``"info"``.
    """
    if message_type == "info":
        logging.info(message)
        if show_in_ui:
            gr.Info(message)
    elif message_type == "warning":
        logging.warning(message)
        gr.Warning(message)
    elif message_type == "error":
        logging.error(message)
        gr.Error(message)
    else:
        error_message = f"unknown message type '{message_type}' (supported are 'info', 'warning' and 'error')"
        logging.error(error_message)
        raise gr.Error(error_message)
    if progressbar is not None:
        progressbar(progress=progress, desc=message)


def generate_image(
        diffuser: str,
        diffuser_dir: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: int = 25,
        guidance: float = 7.0,
        aspect: str = "square",
        seed: Optional[str] = None,
        llm: Optional[str] = None,
        llm_dir: Optional[str] = None,
        expand_prompt: bool = False,
        optimize_prompt: bool = False,
        optimizer_target: Optional[str] = None,
        project: Optional[str] = None,
        progressbar: gr.Progress = gr.Progress()
) -> tuple[Image, str]:
    """Generates the image corresponding to the provided prompt."""
    # initialize outputs
    optimized_prompt: str = prompt
    image: Image.Image = Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))

    log(message="loading llm", progress=0.00, progressbar=progressbar)
    try:
        WRITER.load_model(filepath=os.path.join(llm_dir, llm))
    except Exception as error:
        log(message=f"error (llm loading): {error}", message_type="error")

    if expand_prompt:
        log(message="expanding prompt", progress=0.05, progressbar=progressbar)
        try:
            optimized_prompt = WRITER.expand_prompt(
                prompt=prompt,
                goal=project,
            )
            log(message=f"expanded prompt: {optimized_prompt}")
        except Exception as error:
            log(message=f"Error (prompt expansion): {error}", message_type="error")

    if optimize_prompt:
        log(message="optimizing prompt", progress=0.35, progressbar=progressbar)
        try:
            optimized_prompt = WRITER.optimize_prompt(
                prompt=optimized_prompt,
                rules=optimizer_target
            )
            log(message=f"optimized prompt: {optimized_prompt}")
        except Exception as error:
            log(message=f"Error (prompt optimization): {error}", message_type="error")

    log(message="loading diffuser", progress=0.65, progressbar=progressbar)
    try:
        ARTIST.load_model(filepath=os.path.join(diffuser_dir, diffuser))
    except Exception as error:
        log(message=f"error (diffuser loading): {error}", message_type="error")

    log(message="generating image", progress=0.7, progressbar=progressbar)
    try:
        image = ARTIST.imagine(
            prompt=optimized_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            aspect=aspect,
            seed=eval(seed) if seed else None,
        )
    except Exception as error:
        log(message=f"Error (image gen): {error}", message_type="error")

    return image, optimized_prompt

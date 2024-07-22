import logging
from typing import Optional

import gradio as gr
from PIL import Image

from api.diffuser import Diffuser
from api.optimizer import PromptOptimizer


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


def generate(
        model: str,
        prompt: str,
        negative_prompt: str,
        steps: int,
        guidance: float,
        aspect: str,
        project: str,
        optimize_prompt: bool,
        optimizer_target: str,
        diffuser: Diffuser,
        optimizer_api: str = "cohere",
        optimizer_model: str = "command-r",
        progressbar: gr.Progress = gr.Progress()
) -> tuple[Image, Diffuser]:
    """Generates the image corresponding to the provided prompt."""
    log(message="loading checkpoint", progress=0, progressbar=progressbar)
    try:
        diffuser.load_model(model=model)
    except Exception as error:
        log(message=f"error (model loading): {error}", message_type="error")

    if optimize_prompt:
        log(message="optimizing prompt", progress=0.4, progressbar=progressbar)
        try:
            optimized_prompt = (
                PromptOptimizer(api=optimizer_api)
                .optimize(prompt=prompt, target=optimizer_target, project=project, model=optimizer_model)
            )
        except Exception as error:
            log(message=f"Error (prompt optimization): {error}", message_type="error")
    else:
        optimized_prompt = prompt
    log(message=f"optimized prompt: {optimized_prompt}")

    log(message="generating image", progress=0.6, progressbar=progressbar)
    try:
        image = diffuser.imagine(
            prompt=optimized_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            aspect=aspect
        )
    except Exception as error:
        log(message=f"Error (image gen): {error}", message_type="error")
    log(message=f"image generated")

    return image, diffuser

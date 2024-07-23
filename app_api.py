import logging
from typing import Optional

import gradio as gr
from PIL import Image

from api import Diffuser
from api.llms import LLM, PromptWriter, PromptOptimizer

LLM_MODEL: LLM = LLM()
DIFFUSION_MODEL: Diffuser = Diffuser()


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


def generate_prompt(
        model: str,
        scene: str,
        project: str,
        optimizer_target: str,
        progressbar: gr.Progress = gr.Progress()
) -> str:
    """Generates a prompt based on the provided image description."""
    try:
        LLM_MODEL.load_model(model=model)
    except Exception as error:
        log(message=f"error (model loading): {error}", message_type="error")

    prompt = scene

    # enhance prompt by adding details
    log(message="enhancing prompt", progress=0.0, progressbar=progressbar)
    try:
        prompt = (
            PromptWriter(llm=LLM_MODEL)
            .optimize(prompt=prompt, goal=project)
        )
    except Exception as error:
        log(message=f"Error (prompt optimization): {error}", message_type="error")

    # optimize prompt for target model
    log(message="enhancing prompt", progress=0., progressbar=progressbar)
    try:
        prompt = (
            PromptOptimizer(llm=LLM_MODEL)
            .optimize(prompt=prompt, target=optimizer_target)
        )
    except Exception as error:
        log(message=f"Error (prompt optimization): {error}", message_type="error")

    return prompt


def generate_image(
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: int = 25,
        guidance: float = 7.0,
        aspect: str = "square",
        seed: Optional[str] = None,
        optimize_prompt: bool = True,
        llm: Optional[str] = None,
        optimizer_target: Optional[str] = None,
        project: Optional[str] = None,
        progressbar: gr.Progress = gr.Progress()
) -> tuple[Image, str]:
    """Generates the image corresponding to the provided prompt."""
    # consistency checks
    if optimize_prompt:
        if (not optimizer_target) or (not llm):
            log(
                message="error: 'optimizer_target' and 'llm' are required when optimizing prompt",
                message_type="error",
            )

    log(message="loading diffuser", progress=0, progressbar=progressbar)
    try:
        DIFFUSION_MODEL.load_model(model=model)
    except Exception as error:
        log(message=f"error (model loading): {error}", message_type="error")

    if not optimize_prompt:
        optimized_prompt = prompt
    else:
        log(message="loading LLM", progress=0, progressbar=progressbar)
        try:
            LLM_MODEL.load_model(model=model)
        except Exception as error:
            log(message=f"error (model loading): {error}", message_type="error")

        # enhance prompt by adding details
        log(message="enhancing prompt", progress=0.0, progressbar=progressbar)
        try:
            optimized_prompt = (
                PromptWriter(llm=LLM_MODEL)
                .optimize(prompt=prompt, goal=project)
            )
        except Exception as error:
            log(message=f"Error (prompt optimization): {error}", message_type="error")

        # optimize prompt for target model
        log(message="enhancing prompt", progress=0.0, progressbar=progressbar)
        try:
            optimized_prompt = (
                PromptOptimizer(llm=LLM_MODEL)
                .optimize(prompt=prompt, target=optimizer_target)
            )
        except Exception as error:
            log(message=f"Error (prompt optimization): {error}", message_type="error")
        log(message=f"optimized prompt: {prompt}")

    log(message="generating image", progress=0.0, progressbar=progressbar)
    try:
        image = DIFFUSION_MODEL.imagine(
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

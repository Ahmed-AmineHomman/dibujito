import logging
from typing import Dict, Tuple, Optional, List

import gradio as gr
from PIL import Image

from api import Diffuser, LLM
from api.clients import create_client

LLM_MODEL: LLM
DIFFUSER_MODEL: Diffuser = Diffuser()
CLIENT_MAPPER: Dict[str, Tuple[str, str]] = {
    "command r": ("cohere", "command-r"),
    "command r+": ("cohere", "command-r-plus"),
    "gpt4o mini": ("openai", "gpt-4o-mini"),
    "gpt4o": ("openai", "gpt-4o")
}


def get_available_llms() -> List[str]:
    return [k for k in CLIENT_MAPPER.keys()]


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
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: int = 25,
        guidance: float = 7.0,
        aspect: str = "square",
        seed: Optional[str] = None,
        llm: Optional[str] = None,
        optimization_level: Optional[str] = "none",
        optimizer_target: Optional[str] = None,
        project: Optional[str] = None,
        progressbar: gr.Progress = gr.Progress()
) -> tuple[Image, str]:
    """Generates the image corresponding to the provided prompt."""
    # consistency checks
    if (optimization_level != "none") and (not llm):
        log(
            message="error: LLM must be provided when requiring prompt optimization",
            message_type="error",
        )
    if llm not in CLIENT_MAPPER.keys():
        log(message=f"LLM '{llm}' is not supported", message_type="error")

    # configure LLM
    llm_api, llm_name = CLIENT_MAPPER.get(llm)
    if not ('LLM_MODEL' in locals()) or ('LLM_MODEL' not in globals()):
        # create the LLM instance if it does not exist
        LLM_MODEL = LLM(client=create_client(llm_api))
    else:
        if LLM_MODEL.client.name != llm_api:
            # update API client if it does not correspond to specified model
            LLM_MODEL.client = create_client(api=llm_api)

    # define progressbar levels
    if optimization_level == "strong":
        expand_step = 0.0
        optim_step = 0.25
        loading_step = 0.5
        diffusion_step = 0.6
    elif optimization_level == "light":
        expand_step = 0.0
        optim_step = 0.0
        loading_step = 0.3
        diffusion_step = 0.4
    else:
        expand_step = 0.0
        optim_step = 0.0
        loading_step = 0.0
        diffusion_step = 0.1

    optimized_prompt = prompt

    # expand prompt by adding details
    if optimization_level == "strong":
        log(message="expanding prompt", progress=expand_step, progressbar=progressbar)
        try:
            optimized_prompt = LLM_MODEL.expand_prompt(prompt=prompt, goal=project, model=llm_name)
            log(message=f"expanded prompt: {optimized_prompt}")
        except Exception as error:
            log(message=f"Error (prompt expansion): {error}", message_type="error")

    # optimize prompt for target model
    if optimization_level in ["strong", "light"]:
        log(message="optimizing prompt", progress=optim_step, progressbar=progressbar)
        try:
            optimized_prompt = LLM_MODEL.optimize_prompt(prompt=optimized_prompt, rules=optimizer_target,
                                                         model=llm_name)
            log(message=f"optimized prompt: {optimized_prompt}")
        except Exception as error:
            log(message=f"Error (prompt optimization): {error}", message_type="error")

    log(message="loading diffuser", progress=loading_step, progressbar=progressbar)
    try:
        DIFFUSER_MODEL.load_model(model=diffuser)
    except Exception as error:
        log(message=f"error (model loading): {error}", message_type="error")

    log(message="generating image", progress=diffusion_step, progressbar=progressbar)
    try:
        image = DIFFUSER_MODEL.imagine(
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

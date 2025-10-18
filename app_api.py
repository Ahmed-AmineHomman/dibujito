import logging
import os
from queue import Queue
from threading import Thread
from typing import Optional, List

import gradio as gr
from PIL import Image
from jedi.debug import warning

from api import Diffuser, LLM, PromptingRules

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


def get_model_list(
        directory: str,
        model_type: str
) -> List[str]:
    # default variables
    extension_mapper = dict(
        llm=".gguf",
        diffuser=".safetensors",
        optimizer=".toml"
    )

    # consistency checks
    if model_type not in extension_mapper.keys():
        message = f"unsupported model type {model_type}"
        log(message=message, message_type="error")

    # get list
    names = []
    for f in os.listdir(directory):
        if f.endswith(extension_mapper.get(model_type)):
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
        warning(gr.Warning(message))
    elif message_type == "error":
        logging.error(message)
        raise gr.Error(message)
    else:
        error_message = f"unknown message type '{message_type}' (supported are 'info', 'warning' and 'error')"
        logging.error(error_message)
        raise gr.Error(error_message)
    if progressbar is not None:
        progressbar(progress=progress, desc=message)


def generate_prompt(
        prompt: str,
        llm: str,
        llm_dir: str,
        rules: str,
        rules_dir: str,
        project: str,
        temperature: float = 0.5,
        seed: int = -1
) -> str:
    try:
        WRITER.load_model(filepath=os.path.join(llm_dir, llm))
    except Exception as error:
        log(message=f"error (llm loading): {error}", message_type="error")

    try:
        rules = PromptingRules.from_toml(filepath=os.path.join(rules_dir, rules))
    except Exception as error:
        log(message=f"error (rules loading): {error}", message_type="error")

    # compute response
    response = WRITER.optimize_prompt(
        prompt=prompt,
        goal=project,
        creative_mode=True,
        rules=rules,
        temperature=temperature,
        seed=seed if seed >= 0 else None,
    )
    output = ""
    for chunk in response:
        output += chunk
        yield output


def generate_image(
        prompt: str,
        diffuser: str,
        diffuser_dir: str,
        negative_prompt: Optional[str] = None,
        steps: int = 25,
        guidance: float = 7.0,
        preview_frequency: int = 0,
        aspect: str = "square",
        seed: Optional[int] = -1,
        progressbar: gr.Progress = gr.Progress()
) -> Image:
    """Generates the image corresponding to the provided prompt."""
    # set preview frequency
    try:
        preview_frequency = int(preview_frequency)
    except (TypeError, ValueError):
        preview_frequency = 0
    preview_frequency = max(0, preview_frequency)

    log(message="loading diffuser", progress=0.25, progressbar=progressbar)
    try:
        ARTIST.load_model(filepath=os.path.join(diffuser_dir, diffuser))
    except Exception as error:
        log(message=f"error (diffuser loading): {error}", message_type="error")

    log(message="generating image", progress=0.75, progressbar=progressbar)
    previews_enabled = preview_frequency > 0
    preview_queue: Queue = Queue()
    worker_exception: Optional[Exception] = None

    def handle_preview(image: Image.Image, step: int) -> None:
        preview_queue.put(("preview", image, step))

    def run_pipeline() -> None:
        nonlocal worker_exception
        try:
            final_image = ARTIST.imagine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                aspect=aspect,
                seed=seed,
                preview_frequency=preview_frequency if previews_enabled else None,
                preview_callback=handle_preview if previews_enabled else None,
            )
            preview_queue.put(("final", final_image, steps))
        except Exception as error:
            worker_exception = error
            preview_queue.put(("error", error, -1))
        finally:
            preview_queue.put(("done", None, -1))

    if previews_enabled:
        Thread(target=run_pipeline, daemon=True).start()
        while True:
            kind, payload, _step = preview_queue.get()
            if kind == "preview":
                yield payload
            elif kind == "final":
                yield payload
            elif kind == "error":
                log(
                    message=f"error (image generation): {worker_exception}",
                    message_type="error",
                    progress=1.0,
                    progressbar=progressbar
                )
            elif kind == "done":
                break
        log(message="done", progress=1.0, progressbar=progressbar)
    else:
        try:
            final_image = ARTIST.imagine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                aspect=aspect,
                seed=seed if seed >= 0 else None,
            )
        except Exception as error:
            log(message=f"error (image generation): {error}", message_type="error", progress=1.0,
                progressbar=progressbar)
        else:
            log(message="done", progress=1.0, progressbar=progressbar)
            yield final_image

import logging
import os
from typing import Optional, List

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from api import DEFAULT_MODEL_DIR, DEFAULT_CHECKPOINT_NAME


class Diffuser:
    """
    Class implementing the diffusion text-to-image model.

    It is based on the ``diffusers`` library, and is compatible with any stable-diffusion 1.5 based models.
    """
    pipeline: StableDiffusionPipeline
    cuda: bool

    _database_dir: str = DEFAULT_MODEL_DIR
    _checkpoint_name = DEFAULT_CHECKPOINT_NAME
    _embeddings: List[str] = []
    _loras: List[str] = []

    def __init__(
            self,
            model_dir: Optional[str] = None,
            checkpoint: Optional[str] = None
    ):
        if model_dir:
            self._database_dir = model_dir
        self.cuda = torch.cuda.is_available()
        self.load_checkpoint(model=checkpoint)

    def load_checkpoint(self, model: Optional[str] = None) -> None:
        """Reset the diffusion pipeline with the provided checkpoint."""
        if not model:
            model = DEFAULT_CHECKPOINT_NAME
        self._checkpoint_name = model
        params = self._set_pipeline_parameters()
        self.pipeline = StableDiffusionPipeline.from_single_file(
            os.path.join(self._database_dir, "checkpoints", model),
            **params
        )
        if self.cuda:
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.enable_model_cpu_offload()

    def load_lora(self, model: str) -> None:
        """Adds the LoRa corresponding to the provided model path to the pipeline."""
        self._loras.append(model)
        self.pipeline.load_lora_weights(
            os.path.join(self._database_dir, "loras", model),
            adapter_name=model.split(".")[0]
        )

    def load_embeddings(self, filename: str) -> None:
        self._embeddings.append(filename)
        self.pipeline.load_textual_inversion(os.path.join(self._database_dir, filename))

    def imagine(
            self,
            prompt: str,
            negative_prompt: Optional[str] = None,
            aspect: str = "square",
            steps: int = 20,
            guidance: float = 7,
    ) -> Image:
        """Generates an image corresponding to the provided prompt."""
        # consistency checks
        if aspect not in ["square", "landscape", "portrait"]:
            message = f"unsupported format '{aspect}'"
            logging.error(message)
            raise ValueError(message)

        # define diffusion parameters
        params = dict(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance)
        if negative_prompt:
            params["negative_prompt"] = f"<BadDream>, {negative_prompt}"
        if aspect == "square":
            params["width"] = 512
            params["height"] = 512
        if aspect == "portrait":
            params["width"] = 512
            params["height"] = 768
        if aspect == "landscape":
            params["width"] = 768
            params["height"] = 512

        # generate image
        try:
            image = self.pipeline(**params).images[0]
        except Exception as error:
            message = f"Error (image gen): {error}"
            logging.error(message)
            raise error

        return image

    def _set_pipeline_parameters(self) -> dict:
        params = dict(
            use_safetensors=True
        )
        if self.cuda:
            params["device"] = "auto"
        return params

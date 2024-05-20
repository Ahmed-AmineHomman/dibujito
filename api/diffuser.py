import logging
import os
from typing import Optional

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, LCMScheduler

from api import DEFAULT_CHECKPOINT_DIR, DEFAULT_EMBEDDINGS_DIR, DEPOSIT_ID


class Diffuser:
    """
    Class implementing the diffusion text-to-image model.

    It is based on the ``diffusers`` library, and is compatible with any stable-diffusion 1.5 based models.
    """
    _checkpoint_name = "DreamShaper8_LCM.safetensors"
    _embeddings_name = "BadDream.pt"

    def __init__(self, model_dir: Optional[str] = None):
        self.cuda = torch.cuda.is_available()

        # set parameters
        params = dict(
            scheduler=LCMScheduler.from_pretrained(DEPOSIT_ID, subfolder="scheduler"),
            use_safetensors=True
        )
        if self.cuda:
            params["device"] = "auto"
            params["torch_dtype"] = torch.float16

        # load pipeline
        self.pipeline = StableDiffusionPipeline.from_single_file(
            os.path.join(DEFAULT_CHECKPOINT_DIR, self._checkpoint_name),
            **params
        )
        if self.cuda:
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.enable_model_cpu_offload()
        self.pipeline.load_textual_inversion(os.path.join(DEFAULT_EMBEDDINGS_DIR, self._embeddings_name))

    def imagine(
            self,
            prompt: str,
            negative_prompt: Optional[str] = None,
            aspect: str = "square",
            steps: int = 20,
            guidance: float = 7,
    ) -> Image:
        """Generates an image corresponding to the provided prompt."""
        if aspect not in ["square", "landscape", "portrait"]:
            message = f"unsupported format '{aspect}'"
            logging.error(message)
            raise ValueError(message)
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

        return self.pipeline(**params).images[0]

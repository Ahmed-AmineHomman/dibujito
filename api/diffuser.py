import gc
import logging
import os
from typing import Optional, List, Dict

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


class Diffuser:
    """
    Class implementing the diffusion text-to-image model.

    It is based on the ``diffusers`` library, and is compatible with any stable-diffusion 1.5 based models.
    """
    pipeline: StableDiffusionPipeline
    embeddings: List[str] = []
    cuda: bool

    _paths: Dict[str, str]

    def __init__(
            self,
            checkpoint_dir: str,
            lora_dir: str,
            embeddings_dir: str,
            checkpoint: Optional[str] = None
    ):
        self._paths = {
            "checkpoints": checkpoint_dir,
            "loras": lora_dir,
            "embeddings": embeddings_dir
        }
        self.cuda = torch.cuda.is_available()
        if checkpoint:
            self.load_checkpoint(filename=checkpoint)

    def reset(self) -> None:
        """Resets the pipeline."""
        del self.pipeline
        gc.collect()
        self.pipeline: StableDiffusionPipeline = None

    def load_checkpoint(self, filename: str) -> None:
        """
        Resets the pipeline with the provided checkpoint.

        **Warning**: loading a new checkpoints will drop all loras & embeddings from the pipeline.
        """
        filepath = os.path.join(self._paths.get("checkpoints"), filename)
        params = self._set_pipeline_parameters()
        self.pipeline = StableDiffusionPipeline.from_single_file(filepath, **params)
        if self.cuda:
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.enable_model_cpu_offload()

    def load_lora(self, filename: str) -> None:
        """Adds the LoRa corresponding to the provided filename to the pipeline."""
        filepath = os.path.join(self._pats.get("loras"), filename)
        name = filename.split(".")[0]
        self.pipeline.load_lora_weights(
            pretrained_model_name_or_path_or_dict=filepath,
            adapter_name=name
        )

    def load_embeddings(self, filename: str) -> None:
        """Adds the textual inversion (=embeddings) corresponding to the provided filename to the pipeline."""
        filepath = os.path.join(self._database_dir, filename)
        name = filename.split(".")[0]
        self.pipeline.load_textual_inversion(
            pretrained_model_name_or_path=filepath,
            token=name
        )

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

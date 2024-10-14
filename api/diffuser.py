import logging
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file


class Diffuser:
    """
    Class implementing the diffusion text-to-image model.

    It is based on the ``diffusers`` library, and is compatible with any model based on stable-diffusion 1.5 or SDXL.
    """
    pipeline: StableDiffusionPipeline
    model_path: str
    cuda: bool
    ready: bool
    architecture: str

    _aspect_mapper = {
        "sd1": {"square": (512, 512), "portrait": (512, 640), "landscape": (640, 512)},
        "sdxl": {"square": (1024, 1024), "portrait": (960, 1280), "landscape": (1280, 960)}
    }

    def __init__(
            self,
            filepath: Optional[str] = None
    ):
        self.ready = False
        self.model_path = ""
        self.cuda = torch.cuda.is_available()
        if filepath:
            self.load_model(filepath=filepath)

    @staticmethod
    def get_supported_aspects() -> List[str]:
        return ["square", "portrait", "landscape"]

    def load_model(self, filepath: str) -> None:
        """ Resets the pipeline with the provided model. """
        # consistency checks
        if self.ready:
            if filepath == self.model_path:
                logging.info("skipping pipeline load since filepath points to a previously loaded file")
                return
        fp = Path(filepath)
        if not fp.exists():
            message = "provided filepath does not exist"
            logging.error(message)
            raise Exception(message)
        if not fp.is_file():
            message = "provided filepath does not points to a file"
            logging.error(message)
            raise Exception(message)
        if fp.suffix != ".safetensors":
            message = "only safetensors files are supported"
            logging.error(message)
            raise Exception(message)

        # load pipeline
        is_sd1 = len(load_file(filepath)) < 2000
        self.architecture = "sd1" if is_sd1 else "sdxl"  # assume its sd1-based=
        self.pipeline = (
            (StableDiffusionPipeline if is_sd1 else StableDiffusionXLPipeline)
            .from_single_file(
                pretrained_model_link_or_path=filepath,
                torch_dtype=torch.float16
            )
        )

        # apply optimizations
        if self.cuda:
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.enable_model_cpu_offload()

        # update instance status
        self.ready = True
        self.model_path = filepath

    def imagine(
            self,
            prompt: str,
            negative_prompt: Optional[str] = None,
            aspect: str = "square",
            steps: int = 20,
            guidance: float = 7,
            seed: int = None,
    ) -> Image:
        """Generates an image corresponding to the provided prompt."""
        # consistency checks
        if not self.ready:
            message = "No model loaded. Please call `load_model` to load a model first."
            logging.error(message)
            raise RuntimeError(message)
        if aspect not in self.get_supported_aspects():
            message = f"unsupported format '{aspect}'"
            logging.error(message)
            raise ValueError(message)

        # define diffusion parameters
        params = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=self._aspect_mapper[self.architecture][aspect][0],
            height=self._aspect_mapper[self.architecture][aspect][1],
            generator=torch.Generator().manual_seed(seed) if seed else torch.Generator(),
        )

        # generate image
        try:
            image = self.pipeline(**params).images[0]
        except Exception as error:
            message = f"Error (image gen): {error}"
            logging.error(message)
            raise error

        return image

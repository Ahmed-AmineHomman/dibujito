import logging
import os
from typing import Optional, List, Dict

from enum import Enum
import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, SchedulerMixin, AutoPipelineForText2Image

class DiffuserSpecs:
    deposit: str
    architecture: str

    def __init__(self, deposit: str, architecture: str):
        self.deposit = deposit
        self.architecture = architecture


class Diffuser:
    """
    Class implementing the diffusion text-to-image model.

    It is based on the ``diffusers`` library, and is compatible with any stable-diffusion 1.5 based models.
    """
    pipeline: DiffusionPipeline
    cuda: bool

    _model: Optional[DiffuserSpecs] = None
    _supported_models: Dict[str, DiffuserSpecs] = {
        "sd1": DiffuserSpecs(deposit="runwayml/stable-diffusion-v1-5", architecture="sd1"),
        "dreamshaper": DiffuserSpecs(deposit="lykon/dreamshaper-8", architecture="sd1"),
        "orangemix": DiffuserSpecs(deposit="WarriorMama777/OrangeMixs", architecture="sd1"),
        "sdxl": DiffuserSpecs(deposit="stabilityai/stable-diffusion-xl-base-1.0", architecture="sdxl"),
        "animagine": DiffuserSpecs(deposit="cagliostrolab/animagine-xl-3.1", architecture="sdxl"),
        "playground": DiffuserSpecs(deposit="playgroundai/playground-v2.5-1024px-aesthetic", architecture="sdxl"),
        "sdxl-turbo": DiffuserSpecs(deposit="stabilityai/sdxl-turbo", architecture="sdxl-turbo"),
    }
    _aspect_mapper = {
        "sd1": {"square": (512, 512), "portrait": (768, 512), "landscape": (512, 768)},
        **{k: {
            "square": (1024, 1024),
            "portrait": (1280, 960),
            "landscape": (960, 1280)
        } for k in ["sdxl", "sdxl-turbo", "playground"]}
    }

    def __init__(
            self,
            model: Optional[str] = None
    ):
        self.cuda = torch.cuda.is_available()
        if model:
            self.load_model(model=model)

    @staticmethod
    def get_supported_models() -> List[str]:
        return [k for k in Diffuser._supported_models.keys()]

    @staticmethod
    def get_supported_aspects() -> List[str]:
        return ["square", "portrait", "landscape"]

    def load_model(self, model: str) -> None:
        """ Resets the pipeline with the provided model. """
        if model not in self.get_supported_models():
            message = f"Unsupported model '{model}'"
            logging.error(message)
            raise ValueError(message)
        if self._supported_models.get(model) == self._model:
            pass

        self._model = self._supported_models.get(model)
        params = self._set_pipeline_parameters()
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._model.deposit,
            **params
        )
        if self.cuda:
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.enable_model_cpu_offload()

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
        if aspect not in self.get_supported_aspects():
            message = f"unsupported format '{aspect}'"
            logging.error(message)
            raise ValueError(message)

        # define diffusion parameters
        params = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=self._aspect_mapper[self._model.architecture]["square"][0],
            height=self._aspect_mapper[self._model.architecture]["square"][1],
            generator=torch.Generator().manual_seed(seed) if seed else torch.Generator(),
            negative_prompt=negative_prompt,
        )

        # generate image
        try:
            image = self.pipeline(**params).images[0]
        except Exception as error:
            message = f"Error (image gen): {error}"
            logging.error(message)
            raise error

        return image

    def _set_pipeline_parameters(
            self
    ) -> dict:
        params = dict()
        if self.cuda:
            params["device"] = "auto"
        return params

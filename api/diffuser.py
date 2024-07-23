import logging
from typing import Optional, List, Dict

import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoPipelineForText2Image


class DiffuserSpecs:
    name: str
    deposit: str
    architecture: str
    fp16: bool
    safetensors: bool

    def __init__(
            self,
            name: str,
            deposit: str,
            architecture: str,
            fp16: bool = False,
            safetensors: bool = False,
    ):
        self.name = name
        self.deposit = deposit
        self.architecture = architecture
        self.fp16 = fp16
        self.safetensors = safetensors


class Diffuser:
    """
    Class implementing the diffusion text-to-image model.

    It is based on the ``diffusers`` library, and is compatible with any stable-diffusion 1.5 based models.
    """
    pipeline: DiffusionPipeline
    cuda: bool

    _model: Optional[DiffuserSpecs] = None
    _supported_models: Dict[str, DiffuserSpecs] = {
        "dreamshaper": DiffuserSpecs(
            name="dreamshaper",
            deposit="lykon/dreamshaper-8",
            architecture="sd1",
            fp16=True,
            safetensors=True,
        ),
        "playground": DiffuserSpecs(
            name="playground",
            deposit="playgroundai/playground-v2.5-1024px-aesthetic",
            architecture="sdxl",
            fp16=True,
            safetensors=True,
        ),
    }
    _aspect_mapper = {
        "sd1": {"square": (512, 512), "portrait": (768, 512), "landscape": (512, 768)},
        **{
            k: {"square": (1024, 1024), "portrait": (1280, 960), "landscape": (960, 1280)}
            for k in ["sdxl", "sdxl-turbo", "playground"]
        }
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

        if (self._model is None) or (model != self._model.name):
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
        if self._model is None:
            message = "No model loaded. Please call `load_model` to load a model first."
            logging.error(message)
            raise RuntimeError(message)

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
            if self._model.fp16:
                params["variant"] = "fp16"
                params["torch_dtype"] = torch.float16
        if self._model.safetensors:
            params["use_safetensors"] = True
        return params

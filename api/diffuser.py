import logging
import os
from json import load
from typing import Optional, List

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image


class Diffuser:
    """
    Class implementing the diffusion text-to-image model.

    It is based on the ``diffusers`` library, and is compatible with any stable-diffusion 1.5 based models.
    """
    pipeline: StableDiffusionPipeline
    cuda: bool
    ready: bool

    _config_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api", "data", "diffusers"))
    _config: dict = {}
    _aspect_mapper = {
        "sd1": {"square": (512, 512), "portrait": (768, 512), "landscape": (512, 768)},
        "sdxl": {"square": (1024, 1024), "portrait": (1280, 960), "landscape": (960, 1280)}
    }

    def __init__(
            self,
            model: Optional[str] = None
    ):
        self.ready = False
        self.cuda = torch.cuda.is_available()
        if model:
            self.load_model(model=model)

    @staticmethod
    def get_supported_models() -> List[str]:
        output = os.listdir(Diffuser._config_dir)
        output = [f for f in output if os.path.isfile(os.path.join(Diffuser._config_dir, f))]
        output = [f for f in output if f.endswith(".json")]
        return [f.split(".")[0] for f in output]

    @staticmethod
    def get_supported_aspects() -> List[str]:
        return ["square", "portrait", "landscape"]

    def load_model(self, model: str) -> None:
        """ Resets the pipeline with the provided model. """
        if model not in self.get_supported_models():
            message = f"Unsupported model '{model}'"
            logging.error(message)
            raise ValueError(message)

        # load pipeline
        if self._config.get("name") == model:
            pass
        else:
            # load model configuration
            with open(os.path.join(self._config_dir, f"{model}.json"), "rb") as fh:
                self._config = load(fh)

            # load pipeline
            params = self._set_pipeline_parameters()
            self.pipeline = AutoPipelineForText2Image.from_pretrained(**params)

            # apply optimizations
            if self.cuda:
                self.pipeline = self.pipeline.to("cuda")
                self.pipeline.enable_model_cpu_offload()

            self.ready = True

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
        params = self._set_generation_parameters(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect=aspect,
            steps=steps,
            guidance=guidance,
            seed=seed
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
            self,
    ) -> dict:
        params = dict(
            pretrained_model_or_path=self._config.get("deposit"),
            token=os.getenv("HF_API_KEY")
        )
        if self.cuda:
            if self._config.get("cuda").get("float16"):
                params["torch_dtype"] = torch.float16
            if "variant" in self._config.get("cuda").keys():
                params["variant"] = self._config.get("cuda").get("variant")
        if self._config.get("safetensors"):
            params["use_safetensors"] = True
        return params

    def _set_generation_parameters(
            self,
            prompt: str,
            negative_prompt: str,
            aspect: str,
            steps: int,
            guidance: float,
            seed: Optional[int] = None
    ) -> dict:
        return dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=self._aspect_mapper[self._config.get("architecture")][aspect][0],
            height=self._aspect_mapper[self._config.get("architecture")][aspect][1],
            generator=torch.Generator().manual_seed(seed) if seed else torch.Generator(),
        )

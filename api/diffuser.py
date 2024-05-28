import logging
import os
from typing import Optional, List, Dict

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, SchedulerMixin


class Diffuser:
    """
    Class implementing the diffusion text-to-image model.

    It is based on the ``diffusers`` library, and is compatible with any stable-diffusion 1.5 based models.
    """
    pipeline: StableDiffusionPipeline
    cuda: bool

    _checkpoint: Optional[str] = None
    _supported_schedulers: Dict[str, SchedulerMixin] = {
        "euler_ancestral": EulerAncestralDiscreteScheduler,
        "euler": EulerDiscreteScheduler
    }

    def __init__(
            self,
            checkpoint_path: str
    ):
        self._embeddings = []
        self.cuda = torch.cuda.is_available()
        self.load_checkpoint(filepath=checkpoint_path)

    def get_checkpoint(self) -> str:
        """Returns the currently loaded checkpoint."""
        return self._checkpoint

    def get_loaded_loras(self) -> List[str]:
        """Returns the list of already loaded LoRAs. """
        adapters = self.pipeline.get_list_adapters()
        adapters = set(adapters.get("text_encoder", []) + adapters.get("unet", []))
        return list(adapters)

    def get_active_loras(self) -> List[str]:
        """Returns the list of currently active LoRAs. """
        return self.pipeline.get_active_adapters()

    def load_checkpoint(self, filepath: str) -> None:
        """
        Resets the pipeline with the provided checkpoint.

        **Warning**: loading a new checkpoints will drop all loras & embeddings from the pipeline.
        """
        model = os.path.basename(filepath).split(".")[0]
        if model != self._checkpoint:
            params = self._set_pipeline_parameters()
            self.pipeline = StableDiffusionPipeline.from_single_file(filepath, **params)
            if self.cuda:
                self.pipeline = self.pipeline.to("cuda")
                self.pipeline.enable_model_cpu_offload()
            self._checkpoint = model

    def load_lora(self, filepath: str) -> None:
        """Adds the LoRa corresponding to the provided filename to the pipeline."""
        model = os.path.basename(filepath).split(".")[0]

        # load lora weights if not already loaded
        if model not in self.get_loaded_loras():
            self.pipeline.load_lora_weights(
                pretrained_model_name_or_path_or_dict=filepath,
                adapter_name=model
            )

        # set lora as active
        active_adapters = self.pipeline.get_active_adapters()
        if model not in active_adapters:
            self.pipeline.set_active_adapters(active_adapters + [model])

    def unload_lora(self, model: str) -> None:
        """Removes the LoRa corresponding to the provided filename from the pipeline."""
        if model not in self.get_loaded_loras():
            logging.warning(f"lora '{model}' not loaded -> skipping unload")

        # set lora as inactive
        active_adapters = self.pipeline.get_active_adapters()
        if model in active_adapters:
            self.pipeline.set_active_adapters(active_adapters.remove(model))

        # delete lora weights
        self.pipeline.delete_adapters(adapter_names=model)

    def set_scheduler(self, scheduler: str) -> None:
        """
        Sets the pipeline scheduler.

        Parameters
        ----------
        scheduler: str, {"euler", "euler-ancestral"}
            The name of the scheduler to use.

        Returns
        -------
        None
        """
        if scheduler not in self._supported_schedulers.keys():
            message = f"unsupported scheduler '{scheduler}"
            logging.error(message)
            raise ValueError(message)
        return self._supported_schedulers[scheduler].from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )

    def get_supported_schedulers(self) -> List[str]:
        """Returns a list of supported schedulers."""
        return list(self._supported_schedulers.keys())

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

    def _set_pipeline_parameters(
            self,
            stochastic: bool = False
    ) -> dict:
        # define scheduler
        repo_id = "runwayml/stable-diffusion-v1-5"
        if stochastic:
            scheduler = EulerDiscreteScheduler.from_config(repo_id, subfolder="scheduler")
        else:
            scheduler = EulerAncestralDiscreteScheduler.from_config(repo_id, subfolder="scheduler")
        params = dict(
            use_safetensors=True,
            scheduler=scheduler
        )
        if self.cuda:
            params["device"] = "auto"
        return params

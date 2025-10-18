import logging
from pathlib import Path
from typing import Optional, List, Callable

import torch
from PIL import Image
from diffusers import AutoencoderTiny
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file


class Diffuser:
    """
    Class implementing the diffusion text-to-image model.

    It is based on the ``diffusers`` library, and is compatible with any model based on stable-diffusion 1.5 or SDXL.
    """
    pipeline: StableDiffusionPipeline
    tiny_vae: AutoencoderTiny
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
        self.tiny_vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl",
            # torch_dtype=dtype
        )

        # apply optimizations
        if self.cuda:
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.enable_model_cpu_offload()
            self.tiny_vae = self.tiny_vae.to("cuda")

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
            preview_frequency: Optional[int] = None,
            preview_method: str = "fast",
            preview_callback: Optional[Callable[[Image.Image, int], None]] = None,
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
        if preview_frequency is None:
            preview_frequency = 0
        if preview_callback is None:
            preview_frequency = 0
        preview_method = (preview_method or "fast").lower()
        preview_decoders = {
            "fast": self._latents_to_image_fast,
            "medium": self.latents_to_image_medium,
            "full": self._latents_to_image_full,
        }
        if preview_method not in preview_decoders:
            logging.warning("unsupported preview method '%s', defaulting to 'fast'", preview_method)
            preview_method = "fast"
        decode_preview = preview_decoders[preview_method]

        # define callback
        def _callback(pipeline, step: int, t: torch.Tensor, kwargs: dict):
            if (preview_frequency > 0) and ((step + 1) % preview_frequency == 0):
                print(f"generating preview at step {step}")
                latents = kwargs["latents"]
                try:
                    image = decode_preview(latents)
                    preview_callback(image, step)
                except Exception as error:
                    logging.error(f"error while decoding preview (step {step}): {error}")
                    return
            return kwargs

        # define diffusion parameters
        params = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=self._aspect_mapper[self.architecture][aspect][0],
            height=self._aspect_mapper[self.architecture][aspect][1],
            generator=torch.Generator().manual_seed(seed) if seed else torch.Generator(),
            callback_on_step_end=_callback,
            callback_on_step_end_tensor_input=["latents"]
        )

        # generate image
        try:
            image = self.pipeline(**params).images[0]
        except Exception as error:
            message = f"Error (image gen): {error}"
            logging.error(message)
            raise error

        return image

    @torch.inference_mode()
    def _latents_to_image_fast(self, latents: torch.Tensor) -> Image.Image:
        """Cheap 128x128-ish RGB preview from SDXL latents (4ch -> 3ch projection).
        This mirrors the method shown in the Diffusers docs for quick visualization.
        """
        # latents: [B, 4, H/8, W/8]  (compressed space)
        _, _, H_latent, W_latent = latents.shape
        weights = (
            (60, -60, 25, -70),
            (60, -5, 15, -50),
            (60, 10, -5, -35),
        )
        W = torch.tensor(weights, dtype=latents.dtype, device=latents.device).T  # [4,3]
        b = torch.tensor((150, 140, 130), dtype=latents.dtype, device=latents.device)  # [3]
        # einsum: [...,4,h,w] x [4,3] -> [...,3,h,w]
        rgb = torch.einsum("b l h w, l r -> b r h w", latents, W) + b.view(1, -1, 1, 1)
        img = rgb[0].clamp(0, 255).byte().detach().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(img).resize((W_latent * 8, H_latent * 8), Image.BICUBIC)

    @torch.inference_mode()
    def latents_to_image_medium(self, latents: torch.Tensor) -> Image.Image:
        """
        Decode SDXL latents with the Tiny AutoEncoder (TAESDXL).
        Note: do NOT divide by the standard VAE scaling factor.
        """
        sample = self.tiny_vae.decode(latents, return_dict=False)[0]  # [-1, 1]
        images = self.pipeline.image_processor.postprocess(sample, output_type="pil")
        return images[0]

    @torch.inference_mode()
    def _latents_to_image_full(self, latents: torch.Tensor) -> Image.Image:
        """Converts latent tensors to a single PIL image."""
        latents = latents / self.pipeline.vae.config.scaling_factor
        sample = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(sample, output_type="pil")[0]
        return images

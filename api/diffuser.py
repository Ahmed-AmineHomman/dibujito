from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from PIL import Image
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class Diffuser:
    """Wraps an SD 1.5 or SDXL diffusion pipeline with sane defaults and preview helpers."""

    pipeline: StableDiffusionPipeline
    tiny_vae: AutoencoderTiny
    model_path: str
    cuda: bool
    ready: bool
    architecture: str

    _ASPECT_RESOLUTIONS: Dict[str, Dict[str, Tuple[int, int]]] = {
        "sd1": {"square": (512, 512), "portrait": (512, 640), "landscape": (640, 512)},
        "sdxl": {"square": (1024, 1024), "portrait": (960, 1280), "landscape": (1280, 960)},
    }

    def __init__(
            self,
            filepath: Optional[str] = None
    ) -> None:
        """Initialise the diffusion pipeline wrapper.

        Parameters
        ----------
        filepath
            Optional path to a ``.safetensors`` checkpoint loaded during
            instantiation.
        """
        self.ready = False
        self.model_path = ""
        self.architecture = "sd1"
        self.cuda = torch.cuda.is_available()
        if filepath:
            self.load_model(filepath=filepath)

    @staticmethod
    def get_supported_aspects() -> list[str]:
        """Return the aspect ratio identifiers supported by the pipeline.

        Returns
        -------
        list[str]
            Supported aspect ratio identifiers.
        """
        return ["square", "portrait", "landscape"]

    def load_model(
            self,
            filepath: str
    ) -> None:
        """Load the diffusion weights from ``filepath`` and prepare the runtime pipeline.

        Parameters
        ----------
        filepath
            Path to a ``.safetensors`` model file.

        Raises
        ------
        FileNotFoundError
            Raised when the supplied path does not exist.
        IsADirectoryError
            Raised when the supplied path points to a directory.
        ValueError
            Raised when the supplied file extension is unsupported.
        """
        if self.ready and filepath == self.model_path:
            logger.info("Skipping pipeline load; '%s' already active.", filepath)
            return

        path = self._validate_model_path(filepath)
        self.architecture = self._detect_architecture(filepath)
        self.pipeline = self._create_pipeline(filepath, self.architecture)
        self.tiny_vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl")

        if self.cuda:
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.enable_model_cpu_offload()
            self.tiny_vae = self.tiny_vae.to("cuda")
            logger.info("Pipeline offloaded to CUDA")

        self.ready = True
        self.model_path = str(path)

    def imagine(
            self,
            prompt: str,
            negative_prompt: Optional[str] = None,
            aspect: str = "square",
            steps: int = 20,
            guidance: float = 7,
            seed: Optional[int] = None,
            preview_frequency: Optional[int] = None,
            preview_method: str = "fast",
            preview_callback: Optional[Callable[[Image.Image, int], None]] = None,
    ) -> Image.Image:
        """Generate an image corresponding to ``prompt``.

        Parameters
        ----------
        prompt
            Text description of the desired outcome.
        negative_prompt
            Optional instructions that should be avoided.
        aspect
            Aspect ratio identifier, as returned by ``get_supported_aspects``.
        steps
            Number of sampling steps to perform.
        guidance
            Classifier-free guidance scale.
        seed
            Optional deterministic seed.
        preview_frequency
            Interval (in steps) at which previews should be emitted. ``None``
            or ``0`` disables previews.
        preview_method
            Decoder used for previews. Accepts ``fast``, ``medium`` or ``full``.
        preview_callback
            Callable receiving preview images along with the current step.

        Returns
        -------
        Image.Image
            Final image produced by the diffusion pipeline.

        Raises
        ------
        RuntimeError
            Raised when no model is loaded before invocation.
        ValueError
            Raised when ``aspect`` is unsupported.
        """
        if not self.ready:
            message = "No model loaded. Please call `load_model` to load a model first."
            logger.error(message)
            raise RuntimeError(message)
        if aspect not in self.get_supported_aspects():
            message = f"unsupported format '{aspect}'"
            logger.error(message)
            raise ValueError(message)

        previews_enabled = preview_frequency and preview_frequency > 0 and preview_callback is not None
        preview_frequency = preview_frequency if previews_enabled else 0
        decode_preview = self._select_preview_decoder(preview_method)

        def _callback(
                _pipeline,
                step: int,
                _t: torch.Tensor,
                kwargs: dict
        ) -> dict:
            if previews_enabled and ((step + 1) % preview_frequency == 0):
                latents = kwargs["latents"]
                try:
                    image = decode_preview(latents)
                    preview_callback(image, step)  # type: ignore[arg-type]
                except Exception as error:  # pragma: no cover - only triggered with faulty preview decoding
                    logger.error("Error while decoding preview at step %s: %s", step, error)
            return kwargs

        width, height = self._ASPECT_RESOLUTIONS[self.architecture][aspect]
        generator = self._build_generator(seed)

        try:
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
                callback_on_step_end=_callback if previews_enabled else None,
                callback_on_step_end_tensor_input=["latents"] if previews_enabled else None,
            )
        except Exception as error:
            logger.error("Error (image generation): %s", error)
            raise

        return output.images[0]

    def _validate_model_path(
            self,
            filepath: str
    ) -> Path:
        path = Path(filepath)
        if not path.exists():
            message = f"Provided filepath '{filepath}' does not exist."
            logger.error(message)
            raise FileNotFoundError(message)
        if not path.is_file():
            message = f"Provided filepath '{filepath}' is not a file."
            logger.error(message)
            raise IsADirectoryError(message)
        if path.suffix != ".safetensors":
            message = "Only .safetensors files are supported."
            logger.error(message)
            raise ValueError(message)
        return path

    def _detect_architecture(
            self,
            filepath: str
    ) -> str:
        return "sd1" if len(load_file(filepath)) < 2000 else "sdxl"

    def _create_pipeline(
            self,
            filepath: str,
            architecture: str
    ) -> StableDiffusionPipeline:
        pipeline_cls = StableDiffusionPipeline if architecture == "sd1" else StableDiffusionXLPipeline
        return pipeline_cls.from_single_file(pretrained_model_link_or_path=filepath, torch_dtype=torch.float16)

    def _select_preview_decoder(
            self,
            preview_method: str
    ) -> Callable[[torch.Tensor], Image.Image]:
        method = (preview_method or "fast").lower()
        decoders = {
            "fast": self._latents_to_image_fast,
            "medium": self.latents_to_image_medium,
            "full": self._latents_to_image_full,
        }
        if method not in decoders:
            logger.warning("Unsupported preview method '%s'; defaulting to 'fast'.", method)
            method = "fast"
        return decoders[method]

    def _build_generator(
            self,
            seed: Optional[int]
    ) -> torch.Generator:
        generator = torch.Generator(device="cuda" if self.cuda else "cpu")
        if seed is not None:
            generator.manual_seed(seed)
        return generator

    @torch.inference_mode()
    def _latents_to_image_fast(
            self,
            latents: torch.Tensor
    ) -> Image.Image:
        """
        Produce a quick-and-dirty RGB preview (~128px) from SD latents.
        Mirrors the method shown in the diffusers documentation for fast previews.
        """
        _, _, height_latent, width_latent = latents.shape
        weights = (
            (60, -60, 25, -70),
            (60, -5, 15, -50),
            (60, 10, -5, -35),
        )
        matrix = torch.tensor(weights, dtype=latents.dtype, device=latents.device).T
        bias = torch.tensor((150, 140, 130), dtype=latents.dtype, device=latents.device)
        rgb = torch.einsum("b l h w, l r -> b r h w", latents, matrix) + bias.view(1, -1, 1, 1)
        image = rgb[0].clamp(0, 255).byte().detach().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(image).resize((width_latent * 8, height_latent * 8), Image.BICUBIC)

    @torch.inference_mode()
    def latents_to_image_medium(
            self,
            latents: torch.Tensor
    ) -> Image.Image:
        """Decode SDXL latents using the Tiny AutoEncoder (TAESDXL).

        Parameters
        ----------
        latents
            Latent tensor produced by the diffusion scheduler.

        Returns
        -------
        Image.Image
            Decoded preview image.
        """
        sample = self.tiny_vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(sample, output_type="pil")
        return images[0]

    @torch.inference_mode()
    def _latents_to_image_full(
            self,
            latents: torch.Tensor
    ) -> Image.Image:
        """Fully decode latents with the heavyweight VAE for a high-quality preview."""
        latents = latents / self.pipeline.vae.config.scaling_factor
        sample = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(sample, output_type="pil")[0]
        return images

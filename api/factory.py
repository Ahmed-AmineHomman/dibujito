import json
import logging
import os
from typing import Optional, List, Dict

from huggingface_hub import hf_hub_download

from .config import HF_API_KEY_ENV_NAME


class ModelFactory:
    """
    Class managing model database.

    Allows for retrieval of model files (checkpoints, loras, etc...) from various external sources (Hugging Face Hub, Civitai, etc...).
    Also manages the local model database.
    """
    paths: Dict[str, str]
    _models: dict
    _api_key: str = None

    def __init__(
            self,
            paths: Dict[str, str],
    ):
        self.paths = paths
        for filepath in self.paths.values():  # create directories if they don't exist
            if not os.path.exists(filepath):
                os.makedirs(filepath)
        filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "models.json")
        with open(filepath, "r") as fh:
            self._models = json.load(fh)

    def get_supported_models(self, model_type: Optional[str] = None) -> List[str]:
        """Returns the list of models supported by the factory."""
        if model_type not in ["checkpoints", "loras", "embeddings"]:
            raise ValueError("Unsupported model type '{model_type}'")
        if not model_type:
            return list(self._models.keys())
        else:
            return [model for model in self._models.keys() if self._models.get(model).get("type") == model_type]

    def get_path(self, model: str) -> str:
        """Returns the path of the specified ``model``. """
        if model not in self._models.keys():
            raise ValueError(f"Model '{model}' not supported.")
        model_type = self._models.get(model).get("type")
        return os.path.join(self.paths.get(model_type), f"{model}.safetensors")

    def download(self, model: str) -> None:
        """
        Retrieves model files from the Hugging Face Hub.

        Parameters
        ----------
        model: str
            The name of the model to retrieve.
        Returns
        -------
        None

        See Also
        --------
        huggingface_hub.hf_hub_download:  Downloads a model from the Hugging Face Hub.
        get_supported_models:  Returns the list of models supported by the factory.
        """
        # consistency check
        if model not in self._models.keys():
            raise ValueError(f"Model '{model}' not supported.")

        # if files already exists -> skip download
        model_type = self._models.get(model).get("type")
        target_path = os.path.join(self.paths.get(model_type), f"{model}.safetensors")
        if os.path.exists(target_path):
            logging.warning(f"model already exists at {target_path} -> skipping download")
            return
        else:
            logging.info(f"downloading model {model} to {target_path}")

        # download file from the Hugging Face Hub
        specs = self._models.get(model).get("specs")
        hf_hub_download(
            repo_type="model",
            local_dir=self.paths.get(model_type),
            token=os.getenv(HF_API_KEY_ENV_NAME),
            **specs
        )
        if len(specs.get("subfolder")) > 0:
            filepath = os.path.join(self.paths.get(model_type), *specs.get("subfolder").split("/"),
                                    specs.get("filename"))
        else:
            filepath = os.path.join(self.paths.get(model_type), specs.get("filename"))
        os.rename(
            src=filepath,
            dst=os.path.join(self.paths.get(model_type), f"{model}.safetensors")
        )

    def download_all(self) -> None:
        """
        Downloads all models from the database (can be large!).

        See Also
        --------
        download:  Downloads a model from the Hugging Face Hub.
        get_supported_models:  Returns the list of models supported by the factory.
        """
        for key in self._models.keys():
            self.download(model=key)

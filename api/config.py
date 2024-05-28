import json
import os
from typing import Optional, Dict

COHERE_API_KEY_ENV_NAME = "COHERE_API_KEY"
HF_API_KEY_ENV_NAME = "HF_API_KEY"
OLLAMA_HOST_ENV_NAME = "OLLAMA_HOST"


class AppConfig:
    """
    Class managing the model paths (checkpoints, loras, etc...) and environment variables.
    """
    paths: Dict[str, str]
    environment: Dict[str, str]

    def __init__(
            self,
            paths: Dict[str, str] = None,
            environment: Dict[str, str] = None,
    ):
        # default parameters' values
        if not paths:
            paths = {}
        if not environment:
            environment = {}

        # default values
        self.paths = {
            "checkpoints": "./models/checkpoints",
            "loras": "./models/loras",
            "embeddings": "./models/embeddings",
        }
        self.environment = {
            COHERE_API_KEY_ENV_NAME: os.getenv(COHERE_API_KEY_ENV_NAME, ""),
            OLLAMA_HOST_ENV_NAME: os.getenv(OLLAMA_HOST_ENV_NAME, "http://localhost:11434")
        }

        # replace by provided values
        for key, value in paths.items():
            if key in self.paths.keys():
                self.paths[key] = value
        for key, value in environment.items():
            if key in self.environment.keys():
                self.environment[key] = value

    def dump(self, filepath: str) -> None:
        """Dumps the config into a JSON object in the provided ``filepath``."""
        with open(filepath, "w") as f:
            json.dump(
                obj={
                    "paths": self.paths,
                    "environment": self.environment,
                },
                fp=f,
                indent=4
            )

    @staticmethod
    def load(
            filepath: str,
            paths: Optional[Dict[str, str]] = None,
            environment: Optional[Dict[str, str]] = None,
    ) -> "AppConfig":
        """
        Loads the config from a JSON object in the provided ``filepath``.

        **Remark**: if specific values are provided, they will be used instead of the ones in the JSON file.
        """
        # load config from file
        with open(filepath, "r") as f:
            config = json.load(f)
        if ("paths" not in config.keys()) or ("environment" not in config.keys()):
            raise ValueError("Invalid config file.")

        # update values with provided one
        if paths:
            for key, value in paths.items():
                config["paths"][key] = value
        if environment:
            for key, value in environment.items():
                config["environment"][key] = value

        return AppConfig(**config)

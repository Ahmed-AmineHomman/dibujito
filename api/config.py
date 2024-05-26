import json
from typing import Optional


class AppConfig:
    """
    Class managing the model paths (checkpoints, loras, etc...).
    """
    checkpoints: str = "./models/checkpoints"
    loras: str = "./models/loras"
    embeddings: str = "./models/embeddings"

    def __init__(
            self,
            checkpoints: Optional[str] = None,
            loras: Optional[str] = None,
            embeddings: Optional[str] = None,
    ):
        if checkpoints:
            self.checkpoints = checkpoints
        if loras:
            self.loras = loras
        if embeddings:
            self.embeddings = embeddings

    def get(self, subtype: str) -> str:
        if subtype == "checkpoints":
            return self.checkpoints
        elif subtype == "loras":
            return self.loras
        elif subtype == "embeddings":
            return self.embeddings
        else:
            raise ValueError(f"Unknown subtype {subtype}")

    def dump(self, filepath: str) -> None:
        """Dumps the config into a JSON object in the provided ``filepath``."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @staticmethod
    def load(
            filepath: str,
            checkpoints: Optional[str] = None,
            loras: Optional[str] = None,
            embeddings: Optional[str] = None,
    ) -> "AppConfig":
        """
        Loads the config from a JSON object in the provided ``filepath``.

        **Remark**: if specific paths are provided, they will be used instead of the ones in the JSON file.
        """
        with open(filepath, "r") as f:
            config = json.load(f)
        if checkpoints:
            config["checkpoints"] = checkpoints
        if loras:
            config["loras"] = loras
        if embeddings:
            config["embeddings"] = embeddings
        return AppConfig(**config)

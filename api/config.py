import json
from typing import Optional


class AppConfig:
    """
    Class managing the model paths (checkpoints, loras, etc...).
    """
    checkpoints: str = "./models/checkpoints"
    loras: str = "./models/loras"

    def __init__(
            self,
            checkpoints: Optional[str] = None,
            loras: Optional[str] = None,
    ):
        if checkpoints:
            self.checkpoints = checkpoints
        if loras:
            self.loras = loras

    def get(self, subtype: str) -> str:
        if subtype == "checkpoints":
            return self.checkpoints
        elif subtype == "loras":
            return self.loras
        else:
            raise ValueError(f"Unknown subtype {subtype}")

    def dump(self, filepath: str) -> None:
        """Dumps the config into a JSON object in the provided ``filepath``."""
        with open(filepath, "w") as f:
            json.dump(
                obj={"checkpoints": self.checkpoints, "loras": self.loras},
                fp=f,
                indent=4
            )

    @staticmethod
    def load(
            filepath: str,
            checkpoints: Optional[str] = None,
            loras: Optional[str] = None,
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
        return AppConfig(**config)

from typing import List, Optional

from toml import load, dump


class PromptingRules:
    """
    Class defining the various rules for prompt optimization.
    """
    rules: str
    prefix: str
    suffix: str
    examples: List[str]

    def __init__(
            self,
            rules: str,
            prefix: Optional[str] = "",
            suffix: Optional[str] = "",
            examples: Optional[List[str]] = None,
    ):
        if not examples:
            examples = []

        self.rules = rules
        self.prefix = prefix
        self.suffix = suffix
        self.examples = [e for e in examples]

    def to_dict(self) -> dict:
        return dict(
            rules=self.rules,
            prefix=self.prefix,
            suffix=self.suffix,
            examples=self.examples
        )

    @staticmethod
    def from_dict(config: dict):
        return PromptingRules(
            rules=config.get("rules"),
            prefix=config.get("prefix", ""),
            suffix=config.get("suffix", ""),
            examples=config.get("examples", [])
        )

    def to_toml(self, filepath) -> None:
        dump(o=self.to_dict(), f=filepath)

    @staticmethod
    def from_toml(filepath: str):
        return PromptingRules.from_dict(load(f=filepath))

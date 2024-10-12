from typing import List, Optional

from toml import load, dump


class PromptingRules:
    """
    Class defining the various rules for prompt optimization.
    """
    format: str
    structure: str
    prefix: str
    suffix: str
    guidelines: str
    examples: List[str]

    def __init__(
            self,
            format: str,
            structure: str,
            prefix: Optional[str] = "",
            suffix: Optional[str] = "",
            guidelines: Optional[str] = "",
            examples: Optional[List[str]] = None,
    ):
        if not examples:
            examples = []

        self.format = format
        self.structure = structure
        self.prefix = prefix
        self.suffix = suffix
        self.guidelines = guidelines
        self.examples = [e for e in examples]

    def to_dict(self) -> dict:
        return dict(
            format=self.format,
            structure=self.structure,
            guidelines=self.guidelines,
            prefix=self.prefix,
            suffix=self.suffix,
            examples=self.examples
        )

    @staticmethod
    def from_dict(config: dict):
        return PromptingRules(
            format=config.get("format"),
            structure=config.get("structure"),
            guidelines=config.get("guidelines"),
            prefix=config.get("prefix"),
            suffix=config.get("suffix"),
            examples=config.get("examples")
        )

    def to_toml(self, filepath) -> None:
        dump(o=self.to_dict(), f=filepath)

    @staticmethod
    def from_toml(filepath: str):
        return PromptingRules.from_dict(load(f=filepath))

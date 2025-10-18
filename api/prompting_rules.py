from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from toml import dump, load


@dataclass(frozen=True)
class PromptingRules:
    """Serializable container describing how prompts should be optimised."""

    rules: str
    prefix: str = ""
    suffix: str = ""
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules": self.rules,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "examples": list(self.examples),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PromptingRules":
        return cls(
            rules=config.get("rules", ""),
            prefix=config.get("prefix", ""),
            suffix=config.get("suffix", ""),
            examples=config.get("examples", []),
        )

    def to_toml(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            dump(self.to_dict(), handle)

    @classmethod
    def from_toml(cls, filepath: str) -> "PromptingRules":
        with open(filepath, "r", encoding="utf-8") as handle:
            data = load(handle)
        return cls.from_dict(data)

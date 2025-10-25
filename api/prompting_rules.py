from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
        """Serialise the rule definition into a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            Mapping ready for TOML export.
        """
        return {
            "rules": self.rules,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "examples": list(self.examples),
        }

    @classmethod
    def from_dict(
            cls,
            config: Dict[str, Any]
    ) -> "PromptingRules":
        """Construct an instance from a dictionary.

        Parameters
        ----------
        config
            Mapping containing rule fields.

        Returns
        -------
        PromptingRules
            Fully populated rule configuration.
        """
        return cls(
            rules=config.get("rules", ""),
            prefix=config.get("prefix", ""),
            suffix=config.get("suffix", ""),
            examples=config.get("examples", []),
        )

    def to_toml(
            self,
            filepath: str
    ) -> None:
        """Persist the ruleset to ``filepath`` in TOML format.

        Parameters
        ----------
        filepath
            Destination TOML file.
        """
        with open(filepath, "w", encoding="utf-8") as handle:
            dump(self.to_dict(), handle)

    @classmethod
    def from_toml(
            cls,
            filepath: str
    ) -> "PromptingRules":
        """Load a ruleset from a TOML document.

        Parameters
        ----------
        filepath
            Path to a TOML document on disk.

        Returns
        -------
        PromptingRules
            Parsed rule configuration.
        """
        with open(filepath, "r", encoding="utf-8") as handle:
            data = load(handle)
        return cls.from_dict(data)


def get_supported_rules() -> list[str]:
    """Expose the available prompting rule presets shipped with the app.

    Returns
    -------
    list[str]
        Sorted file names of bundled prompting rules.
    """
    rules_dir = Path(__file__).resolve().parent.parent / "data" / "prompting_rules"
    if not rules_dir.exists():
        return []
    return sorted(entry.name for entry in rules_dir.glob("*.toml") if entry.is_file())

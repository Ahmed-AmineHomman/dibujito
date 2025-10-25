from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from llama_cpp import Llama

from .prompting_rules import PromptingRules

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are a friendly prompt-engineering assistant helping users craft prompts for a text-to-image diffusion model.

Primary goal: {goal}

Model prompting rules:
{rules}

Reference examples:
{examples}

Formatting requirements:
- Prefix to apply to every finalized prompt: {prefix}
- Suffix to apply to every finalized prompt: {suffix}

Guidelines:
- Ask clarifying questions when information is missing or ambiguous.
- Suggest incremental improvements and explain the reasoning briefly.
- When you share an optimized prompt, place it inside a fenced code block labelled `prompt` and include the configured prefix/suffix (omit the value when it is `[none]`).
- Stay collaborative and concise; keep the user focused on building the best possible prompt.
- {creative_instruction}
"""


class LLM:
    """Convenience wrapper around llama.cpp to optimise prompts based on prompting rules."""

    ready: bool
    model_path: str
    llm: Llama

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.ready = False
        self.model_path = ""
        if filepath:
            self.load_model(filepath=filepath)

    def load_model(self, filepath: str) -> None:
        """Load a llama.cpp-compatible model from ``filepath``."""
        if self.ready and filepath == self.model_path:
            logger.info("Skipping LLM load; '%s' already active.", filepath)
            return

        path = self._validate_model_path(filepath)
        self.llm = Llama(model_path=str(path), n_ctx=0)
        self.ready = True
        self.model_path = str(path)

    def optimize_prompt(
        self,
        conversation: List[Dict[str, str]],
        rules: PromptingRules,
        goal: Optional[str] = "Create beautiful and aesthetically pleasing images",
        creative_mode: bool = True,
        **kwargs,
    ) -> Iterator[str]:
        """Stream a conversational response that helps refine the user's prompt."""
        prepared_conversation = self._prepare_conversation(conversation)
        system_prompt = self._build_system_prompt(
            rules=rules,
            instructions=self._build_instruction_block(creative_mode=creative_mode),
            goal=goal,
        )

        chat_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        chat_messages.extend(prepared_conversation)

        yield from self._respond(messages=chat_messages, **kwargs)

    def _respond(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        if not self.ready:
            message = "No model loaded. Please call `load_model` to load a model first."
            logger.error(message)
            raise RuntimeError(message)

        response = self.llm.create_chat_completion(messages=messages, stream=True, **kwargs)
        for chunk in response:
            data = chunk["choices"][0]["delta"]
            content = data.get("content")
            if content:
                yield content

    def _prepare_conversation(self, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Return a sanitized copy of ``conversation`` compatible with llama.cpp."""
        prepared: List[Dict[str, str]] = []
        for entry in conversation:
            role = entry.get("role")
            content = entry.get("content")
            if role not in {"user", "assistant"}:
                continue
            if content is None:
                continue
            prepared.append({"role": role, "content": content})
        return prepared

    def _build_instruction_block(self, creative_mode: bool) -> str:
        if not creative_mode:
            return (
                "Ensure that your prompt reflects all information contained in the user's description.\n"
                "Do not invent new elements; only rely on the provided description. "
                "If a section cannot be filled, omit it gracefully."
            )
        return (
            "Ensure that your prompt covers every part of the expected structure.\n"
            "When the user description lacks details, invent coherent additions that align with the stated goal."
        )

    def _build_system_prompt(
        self,
        rules: PromptingRules,
        instructions: str,
        goal: Optional[str],
    ) -> str:
        examples = "\n".join(f"- {example}" for example in rules.examples) if rules.examples else "No examples provided."
        prefix = rules.prefix if rules.prefix else "[none]"
        suffix = rules.suffix if rules.suffix else "[none]"
        creative_instruction = instructions.replace("\n", " ")
        goal_text = goal or "Create beautiful and aesthetically pleasing images"
        return CHAT_SYSTEM_PROMPT.format(
            rules=rules.rules.strip(),
            examples=examples.strip(),
            prefix=prefix.strip(),
            suffix=suffix.strip(),
            creative_instruction=creative_instruction.strip(),
            goal=goal_text.strip(),
        )

    def _validate_model_path(self, filepath: str) -> Path:
        path = Path(filepath)
        if not path.exists():
            message = f"Provided filepath '{filepath}' does not exist."
            logger.error(message)
            raise FileNotFoundError(message)
        if not path.is_file():
            message = f"Provided filepath '{filepath}' is not a file."
            logger.error(message)
            raise IsADirectoryError(message)
        if path.suffix != ".gguf":
            message = "Only GGUF files are supported."
            logger.error(message)
            raise ValueError(message)
        return path

    @staticmethod
    def get_supported_rules() -> list[str]:
        """Expose the available prompting rule presets shipped with the app."""
        rules_dir = Path(__file__).resolve().parent.parent / "data" / "prompting_rules"
        if not rules_dir.exists():
            return []
        return sorted(entry.name for entry in rules_dir.glob("*.toml") if entry.is_file())

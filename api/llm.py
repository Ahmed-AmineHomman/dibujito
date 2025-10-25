from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

from llama_cpp import Llama

from .prompting_rules import PromptingRules

if TYPE_CHECKING:
    from gradio.data_classes import ChatMessage
else:
    try:  # pragma: no cover - runtime import guard
        from gradio.data_classes import ChatMessage  # type: ignore
    except (ImportError, AttributeError):  # pragma: no cover - gradio fallback
        ChatMessage = Any  # type: ignore

logger = logging.getLogger(__name__)

MessageLike = Union["ChatMessage", Dict[str, Any]]

CHAT_SYSTEM_PROMPT = """You are a friendly prompt-engineering assistant helping users craft prompts for a text-to-image diffusion model.

Model prompting rules:
{rules}

Reference examples:
{examples}

Guidelines:
- Ask clarifying questions when information is missing or ambiguous.
- Suggest incremental improvements and explain the reasoning briefly.
- When you share an optimized prompt, place it inside a fenced code block labelled `prompt`.
- Stay collaborative and concise; keep the user focused on building the best possible prompt.
- {creative_instruction}
"""


class LLM:
    """Simple llama.cpp wrapper compatible with Gradio chat messages."""

    system_instructions: str = ""

    def __init__(
            self,
            filepath: Optional[str] = None,
            *,
            instructions: Optional[str] = None
    ) -> None:
        self._llm: Optional[Llama] = None
        self._model_path: Optional[str] = None
        self._context_length: Optional[int] = None

        if instructions is not None:
            self.set_instructions(instructions)

        if filepath is not None:
            self.load_model(filepath)

    @property
    def ready(self) -> bool:
        """Return True when a llama.cpp model is loaded and ready for inference."""
        return self._llm is not None

    def load_model(
            self,
            filepath: str,
            *,
            context_length: Optional[int] = None
    ) -> None:
        """Load a llama.cpp-compatible model from ``filepath``."""
        path = self._validate_model_path(filepath)

        if self._model_path == str(path) and self.ready:
            logger.info("Skipping LLM load; '%s' already active.", path)
            return

        init_kwargs: Dict[str, Any] = {"model_path": str(path)}
        if context_length is not None:
            init_kwargs["n_ctx"] = context_length
        else:
            init_kwargs["n_ctx"] = 0

        try:
            self._llm = Llama(**init_kwargs)
        except Exception as error:  # pragma: no cover - surfaced to caller
            logger.exception("Failed to load llama.cpp model from '%s'.", path)
            raise

        self._model_path = str(path)
        self._context_length = init_kwargs["n_ctx"]
        logger.info("Loaded llama.cpp model from '%s'.", path)

    def set_instructions(
            self,
            instructions: str
    ) -> None:
        """Set the system instructions used for chat completions."""
        type(self).system_instructions = instructions.strip()

    def configure_prompting(
            self,
            rules: PromptingRules,
            *,
            creative_mode: bool = True,
    ) -> None:
        """Prepare system instructions based on the provided prompting rules."""
        instruction_block = self._build_instruction_block(creative_mode=creative_mode)
        system_prompt = self._build_system_prompt(
            rules=rules,
            instructions=instruction_block,
        )
        self.set_instructions(system_prompt)

    def respond(
            self,
            messages: Sequence[MessageLike],
            *,
            temperature: float = 0.2,
            seed: Optional[int] = None,
    ) -> str:
        """Return a single chat completion produced by the loaded model."""
        if not self.ready or self._llm is None:
            message = "No model loaded. Call `load_model` before using `respond`."
            logger.error(message)
            raise RuntimeError(message)

        chat_messages = self._prepare_messages(messages)
        system_prompt = self.system_instructions.strip()
        if system_prompt:
            chat_messages.insert(0, {"role": "system", "content": system_prompt})

        params: Dict[str, Any] = {
            "messages": chat_messages,
            "temperature": float(temperature),
            "stream": False,
        }

        normalised_seed = self._normalise_seed(seed)
        if normalised_seed is not None:
            params["seed"] = normalised_seed

        logger.debug("Invoking llama.cpp with %d message(s).", len(chat_messages))
        result = self._llm.create_chat_completion(**params)
        choices = result.get("choices", [])
        if not choices:
            logger.warning("llama.cpp returned an empty response.")
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")
        return content or ""

    def _prepare_messages(
            self,
            messages: Sequence[MessageLike]
    ) -> List[Dict[str, str]]:
        prepared: List[Dict[str, str]] = []
        for entry in messages:
            role: Optional[str]
            content: Optional[Any]
            if hasattr(entry, "role") and hasattr(entry, "content"):
                role = getattr(entry, "role")
                content = getattr(entry, "content")
            elif isinstance(entry, dict):
                role = entry.get("role")
                content = entry.get("content")
            else:
                role = None
                content = None

            if role not in {"user", "assistant", "system"}:
                continue
            if content is None:
                continue

            prepared.append({"role": str(role), "content": str(content)})

        return prepared

    def _build_instruction_block(
            self,
            creative_mode: bool
    ) -> str:
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
    ) -> str:
        examples = "\n".join(
            f"- {example}" for example in rules.examples) if rules.examples else "No examples provided."
        prefix = rules.prefix if rules.prefix else "[none]"
        suffix = rules.suffix if rules.suffix else "[none]"
        creative_instruction = instructions.replace("\n", " ")
        return CHAT_SYSTEM_PROMPT.format(
            rules=rules.rules.strip(),
            examples=examples.strip(),
            prefix=prefix.strip(),
            suffix=suffix.strip(),
            creative_instruction=creative_instruction.strip(),
        )

    def _normalise_seed(
            self,
            seed: Optional[int | float]
    ) -> Optional[int]:
        if seed is None:
            return None
        try:
            candidate = int(seed)
        except (TypeError, ValueError):
            return None
        return candidate if candidate >= 0 else None

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
        if path.suffix != ".gguf":
            message = "Only GGUF files are supported."
            logger.error(message)
            raise ValueError(message)
        return path

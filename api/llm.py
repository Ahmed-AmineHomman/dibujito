from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union, TYPE_CHECKING

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


@dataclass(frozen=True)
class AgentTool:
    """Describe a callable tool that can extend the assistant."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], str]

    def invoke(
            self,
            arguments: Dict[str, Any]
    ) -> str:
        """Execute the tool with the provided argument payload."""
        return self.handler(arguments)


@dataclass(frozen=True)
class ToolCall:
    """Represent a structured tool call requested by the model."""

    name: str
    arguments: Dict[str, Any]


class LLM:
    """Simple llama.cpp wrapper compatible with Gradio chat messages."""

    system_instructions: str = ""

    def __init__(
            self,
            filepath: Optional[str] = None,
            instructions: Optional[str] = None,
            llama_verbose: bool = False,
    ) -> None:
        """Initialise the llama.cpp wrapper.

        Parameters
        ----------
        filepath
            Optional GGUF checkpoint to load on instantiation.
        instructions
            Optional system prompt applied before responding.
        llama_verbose
            When ``True`` surface llama.cpp internal logs. Otherwise keep the
            backend quiet.
        """
        self._llm: Optional[Llama] = None
        self._model_path: Optional[str] = None
        self._context_length: Optional[int] = None
        self._llama_verbose: bool = llama_verbose
        self._base_instructions: str = ""
        self._tools: Dict[str, AgentTool] = {}
        self._register_default_tools()

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
            context_length: Optional[int] = None
    ) -> None:
        """Load a llama.cpp-compatible model from ``filepath``.

        Parameters
        ----------
        filepath
            Path to a GGUF model compatible with llama.cpp.
        context_length
            Optional context window override. When omitted the llama.cpp
            default is used.

        Raises
        ------
        FileNotFoundError
            Raised when ``filepath`` does not point to an existing file.
        IsADirectoryError
            Raised when ``filepath`` targets a directory.
        ValueError
            Raised when the supplied file extension is unsupported.
        """
        path = self._validate_model_path(filepath)

        if self._model_path == str(path) and self.ready:
            logger.info("Skipping LLM load; '%s' already active.", path)
            return

        init_kwargs: Dict[str, Any] = {
            "model_path": str(path),
            "verbose": self._llama_verbose,
        }
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
        """Set the system instructions used for chat completions.

        Parameters
        ----------
        instructions
            System prompt to prepend to all conversations.
        """
        self._base_instructions = instructions.strip()
        self._refresh_instructions()

    def configure_prompting(
            self,
            rules: PromptingRules,
            creative_mode: bool = True,
    ) -> None:
        """Prepare system instructions based on the provided prompting rules.

        Parameters
        ----------
        rules
            Prompting rules describing structure, prefix, suffix, and examples.
        creative_mode
            When ``True`` the assistant may invent details to complete prompts.
        """
        creative_instructions = (
            self
            ._build_creative_instructions(creative_mode=creative_mode)
            .replace("\n", " ")
        )
        examples = (
            "\n".join(f"- {example}" for example in rules.examples)
            if rules.examples else "No examples provided."
        )
        instructions = (
            CHAT_SYSTEM_PROMPT
            .format(
                rules=rules.rules.strip(),
                examples=examples.strip(),
                creative_instruction=creative_instructions.strip(),
            )
            .strip()
        )
        self.set_instructions(instructions)

    def register_tool(
            self,
            tool: AgentTool
    ) -> None:
        """Register an additional tool for the assistant to leverage."""
        self._tools[tool.name] = tool
        self._refresh_instructions()

    def _register_default_tools(self) -> None:
        """Populate the toolbox with built-in helpers."""
        self.register_tool(self._build_dummy_tool())

    def _refresh_instructions(self) -> None:
        """Update the active system prompt to reflect tool availability."""
        composed = self._compose_system_instructions()
        type(self).system_instructions = composed

    def _compose_system_instructions(self) -> str:
        base = self._base_instructions.strip()
        tool_block = self._build_tool_instructions()
        parts = [segment for segment in (base, tool_block) if segment]
        return "\n\n".join(parts).strip()

    def _build_tool_instructions(self) -> str:
        if not self._tools:
            return ""

        tool_descriptions = []
        for tool in self._tools.values():
            schema = json.dumps(tool.parameters, indent=2, sort_keys=True)
            tool_descriptions.append(
                f"- {tool.name}: {tool.description}\n  Parameters JSON Schema:\n{schema}"
            )

        specification = "\n".join(tool_descriptions)
        protocol = (
            "Tool usage protocol:\n"
            "- When you decide to call a tool, respond with JSON on a single line.\n"
            "- The JSON MUST contain the keys 'name' and 'arguments'.\n"
            "- Example: {\"name\": \"dummy_tool\", \"arguments\": {\"text\": \"example\"}}\n"
            "- The application executes the tool and provides its output as a 'tool' role message.\n"
            "- After receiving the tool result, continue the conversation with a helpful assistant reply."
        )

        return f"Available tools:\n{specification}\n\n{protocol}"

    @staticmethod
    def _build_dummy_tool() -> AgentTool:
        def _dummy_handler(arguments: Dict[str, Any]) -> str:
            text = arguments.get("text")
            if not isinstance(text, str):
                return "Dummy tool expects a 'text' field containing the message to echo."
            return f"Dummy tool received: {text}"

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Arbitrary text to echo back for debugging the agent loop.",
                }
            },
            "required": ["text"],
            "additionalProperties": False,
        }

        return AgentTool(
            name="dummy_tool",
            description="Echo the provided text to verify that tool execution works end-to-end.",
            parameters=schema,
            handler=_dummy_handler,
        )

    def respond(
            self,
            messages: Sequence[MessageLike],
            temperature: float = 0.2,
            seed: Optional[int] = None,
            stream: bool = False,
            agent_mode: bool = False,
    ) -> Union[str, Iterator[str]]:
        """Return a chat completion produced by the loaded model.

        Parameters
        ----------
        messages
            Conversation history to condition the assistant on.
        temperature
            Sampling temperature for the completion.
        seed
            Pseudo-random seed used when supported by llama.cpp.
        stream
            When True, yield completion fragments as they become available.
        agent_mode
            When True, enable the agentic loop so the assistant can request
            and consume tool calls before answering the user.

        Returns
        -------
        Union[str, Iterator[str]]
            Full assistant reply when ``stream`` is False, otherwise an iterator
            yielding completion fragments.

        Raises
        ------
        RuntimeError
            Raised when no llama.cpp model has been loaded.
        """
        if not self.ready or self._llm is None:
            message = "No model loaded. Call `load_model` before using `respond`."
            logger.error(message)
            raise RuntimeError(message)

        if agent_mode and self._tools:
            return self._agentic_respond(
                messages=messages,
                temperature=temperature,
                seed=seed,
                stream=stream,
            )

        return self._basic_respond(
            messages=messages,
            temperature=temperature,
            seed=seed,
            stream=stream,
        )

    def _basic_respond(
            self,
            messages: Sequence[MessageLike],
            temperature: float,
            seed: Optional[int],
            stream: bool,
    ) -> Union[str, Iterator[str]]:
        params = self._build_completion_params(
            messages=messages,
            temperature=temperature,
            seed=seed,
            stream=stream,
        )
        logger.debug("Invoking llama.cpp with %d message(s).", len(params["messages"]))

        if stream:
            def _token_iterator() -> Iterator[str]:
                assert self._llm is not None  # for type checkers
                for chunk in self._llm.create_chat_completion(**params):
                    token = self._extract_stream_content(chunk)
                    if token:
                        yield token

            return _token_iterator()

        assert self._llm is not None  # for type checkers
        result = self._llm.create_chat_completion(**params)
        choices = result.get("choices", [])
        if not choices:
            logger.warning("llama.cpp returned an empty response.")
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")
        return content or ""

    def _agentic_respond(
            self,
            messages: Sequence[MessageLike],
            temperature: float,
            seed: Optional[int],
            stream: bool,
    ) -> Union[str, Iterator[str]]:
        conversation = self._prepare_messages(messages)
        if stream:
            return self._agentic_stream(
                conversation=conversation,
                temperature=temperature,
                seed=seed,
            )

        return self._agentic_text(
            conversation=conversation,
            temperature=temperature,
            seed=seed,
        )

    def _agentic_text(
            self,
            conversation: Sequence[Dict[str, Any]],
            temperature: float,
            seed: Optional[int],
    ) -> str:
        dialogue = list(conversation)
        max_iterations = 5

        for iteration in range(max_iterations):
            logger.debug("Agent iteration %d with %d message(s).", iteration + 1, len(dialogue))
            assistant_output = self._basic_respond(
                messages=dialogue,
                temperature=temperature,
                seed=seed,
                stream=False,
            )
            if isinstance(assistant_output, str):
                assistant_text = assistant_output.strip()
            else:
                assistant_chunks = list(assistant_output)
                assistant_text = "".join(assistant_chunks).strip()

            if not assistant_text:
                logger.debug("Assistant returned empty content during agent loop.")
                break

            dialogue.append({"role": "assistant", "content": assistant_text})
            tool_call = self._parse_tool_call(assistant_text)
            if tool_call is None:
                return assistant_text

            tool_result = self._run_tool(tool_call)
            dialogue.append(
                {
                    "role": "tool",
                    "name": tool_call.name,
                    "content": tool_result,
                }
            )

        logger.warning("Agent loop exceeded the maximum number of iterations.")
        return (
            "The assistant could not complete the task because it reached the iteration limit while calling tools."
        )

    def _agentic_stream(
            self,
            conversation: Sequence[Dict[str, Any]],
            temperature: float,
            seed: Optional[int],
    ) -> Iterator[str]:
        dialogue = list(conversation)
        max_iterations = 5

        def _iterator() -> Iterator[str]:
            for iteration in range(max_iterations):
                logger.debug("Agent iteration %d with %d message(s).", iteration + 1, len(dialogue))
                params = self._build_completion_params(
                    messages=dialogue,
                    temperature=temperature,
                    seed=seed,
                    stream=True,
                )
                assert self._llm is not None  # for type checkers
                stream_iter = self._llm.create_chat_completion(**params)

                buffered_tokens: List[str] = []
                published_tokens: List[str] = []
                streaming_enabled = False
                first_non_whitespace: Optional[str] = None

                for chunk in stream_iter:
                    token = self._extract_stream_content(chunk)
                    if not token:
                        continue

                    published_tokens.append(token)

                    if not streaming_enabled:
                        # Buffer early tokens until we know whether this is a tool call (JSON) or a user-facing reply.
                        buffered_tokens.append(token)
                        if first_non_whitespace is None:
                            stripped = token.lstrip()
                            if stripped:
                                first_non_whitespace = stripped[0]
                                if first_non_whitespace != "{":
                                    streaming_enabled = True
                                    # Flush everything collected so far once we are sure this is a natural-language answer.
                                    for buffered in buffered_tokens:
                                        yield buffered
                                    buffered_tokens.clear()
                                    continue
                        continue

                    yield token

                assistant_text = "".join(published_tokens).strip()
                if not assistant_text:
                    logger.debug("Assistant returned empty content during agent loop.")
                    return

                dialogue.append({"role": "assistant", "content": assistant_text})

                if first_non_whitespace == "{":
                    tool_call = self._parse_tool_call(assistant_text)
                    if tool_call is not None:
                        tool_result = self._run_tool(tool_call)
                        dialogue.append(
                            {
                                "role": "tool",
                                "name": tool_call.name,
                                "content": tool_result,
                            }
                        )
                        continue

                    # JSON payload was malformed; treat as a plain answer.
                    if buffered_tokens:
                        for token in buffered_tokens:
                            yield token
                    elif not streaming_enabled:
                        for token in published_tokens:
                            yield token
                    return

                if not streaming_enabled:
                    for token in buffered_tokens or published_tokens:
                        if token:
                            yield token
                return

            logger.warning("Agent loop exceeded the maximum number of iterations.")
            yield (
                "The assistant could not complete the task because it reached the iteration limit while calling tools."
            )

        return _iterator()

    def _run_tool(
            self,
            tool_call: ToolCall,
    ) -> str:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            logger.warning(
                "Assistant requested unknown tool '%s' with arguments: %s.",
                tool_call.name,
                self._format_tool_payload(tool_call.arguments),
            )
            return f"Tool '{tool_call.name}' is not available."

        logger.info(
            "Assistant invoking tool '%s' with arguments: %s",
            tool_call.name,
            self._format_tool_payload(tool_call.arguments),
        )
        try:
            result = tool.invoke(tool_call.arguments)
        except Exception:  # pragma: no cover - surfaced to chat UI
            logger.exception("Tool '%s' execution failed.", tool_call.name)
            return f"Tool '{tool_call.name}' failed during execution."

        logger.info(
            "Tool '%s' completed with result: %s",
            tool_call.name,
            self._format_tool_payload(result),
        )
        return result

    @staticmethod
    def _format_tool_payload(
            payload: Any,
            *,
            limit: int = 200,
    ) -> str:
        if isinstance(payload, str):
            preview = payload
        else:
            try:
                preview = json.dumps(payload, sort_keys=True)
            except TypeError:
                preview = str(payload)

        if len(preview) > limit:
            return f"{preview[:limit]}…"
        return preview

    @staticmethod
    def _parse_tool_call(
            content: str
    ) -> Optional[ToolCall]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        name = payload.get("name") or payload.get("tool_name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None

        return ToolCall(name=name, arguments=arguments)

    @staticmethod
    def _prepare_messages(
            messages: Sequence[MessageLike]
    ) -> List[Dict[str, str]]:
        prepared: List[Dict[str, str]] = []
        for entry in messages:
            role: Optional[str]
            content: Optional[Any]
            name: Optional[str]
            if hasattr(entry, "role") and hasattr(entry, "content"):
                role = getattr(entry, "role")
                content = getattr(entry, "content")
                name = getattr(entry, "name", None)
            elif isinstance(entry, dict):
                role = entry.get("role")
                content = entry.get("content")
                name = entry.get("name")
            else:
                role = None
                content = None
                name = None

            if role not in {"user", "assistant", "system", "tool"}:
                continue
            if content is None:
                continue

            prepared_entry: Dict[str, str] = {"role": str(role), "content": str(content)}
            if role == "tool" and name:
                prepared_entry["name"] = str(name)

            prepared.append(prepared_entry)

        return prepared

    @staticmethod
    def _build_creative_instructions(
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

    @staticmethod
    def _normalise_seed(
            seed: Optional[int | float]
    ) -> Optional[int]:
        if seed is None:
            return None
        try:
            candidate = int(seed)
        except (TypeError, ValueError):
            return None
        return candidate if candidate >= 0 else None

    @staticmethod
    def _validate_model_path(
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

    def _build_completion_params(
            self,
            messages: Sequence[MessageLike],
            temperature: float,
            seed: Optional[int],
            stream: bool,
    ) -> Dict[str, Any]:
        chat_messages = self._prepare_messages(messages)
        system_prompt = self.system_instructions.strip()
        if system_prompt:
            chat_messages.insert(0, {"role": "system", "content": system_prompt})

        params: Dict[str, Any] = {
            "messages": chat_messages,
            "temperature": float(temperature),
            "stream": stream,
        }

        normalised_seed = self._normalise_seed(seed)
        if normalised_seed is not None:
            params["seed"] = normalised_seed

        return params

    @staticmethod
    def _extract_stream_content(chunk: Any) -> str:
        if not isinstance(chunk, dict):
            return ""
        choices = chunk.get("choices")
        if not choices:
            return ""

        choice = choices[0] or {}
        if "delta" in choice:
            delta = choice.get("delta") or {}
            return str(delta.get("content") or "")
        if "text" in choice:
            return str(choice.get("text") or "")
        if "content" in choice:
            return str(choice.get("content") or "")
        if "message" in choice:
            message = choice.get("message") or {}
            return str(message.get("content") or "")
        return ""

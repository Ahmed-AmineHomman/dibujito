import logging
import os
from tomllib import load
from typing import List, Optional, Dict
from pathlib import Path

from api.clients import BaseClient
from llama_cpp import Llama


class LLM:
    """
    Class managing the interactions with LLM models.
    """
    _rules_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompting_rules"))
    _prompt_template: str = """
Turn the image description provided by the user into an optimized prompt for text-to-image diffusion models.
Return your prompt as plain text only, with no additional text, introductions or interpretations.

<instructions>

Structure your prompt according to the provided structure, and ensure it is formatted according to the provided format.
Finally, follow the provided guidelines when designing your prompt.

---
Format:
<format>

Structure:
<structure>

Guidelines:
<guidelines>

Examples:
<examples>
---
"""

    ready: bool
    model_path: str
    llm: Llama
    rules: Dict[str, str]

    def __init__(
            self,
            filepath: Optional[str] = None,
            rules: str = "default"
    ):
        self.ready = False
        self.model_path = ""

        # load model from file
        if filepath:
            self.load_model(filepath=filepath)

        # retrieve prompting rules
        _rules = rules
        if _rules not in self.get_supported_rules():
            message = "provided rules cannot be found -> skipping to default"
            logging.warning(message)
            _rules = "default"
        with open(os.path.join(self._rules_dir, f"{_rules}.toml"), "rb") as fp:
            self.rules = load(fp)


    @staticmethod
    def get_supported_rules() -> List[str]:
        """ Returns a list of supported prompting rules. """
        rules = os.listdir(LLM._rules_dir)
        rules = [f for f in rules if os.path.isfile(os.path.join(LLM._rules_dir, f))]
        rules = [f for f in rules if f.endswith(".toml")]
        return [f.split(".")[0] for f in rules]

    def load_model(self, filepath: str) -> None:
        """ Loads a llama.cpp-compatible model from the provided filepath. """
        # consistency checks
        if self.ready:
            if filepath == self.model_path:
                logging.info("skipping pipeline load since filepath points to a previously loaded file")
                return
        fp = Path(filepath)
        if not fp.exists():
            message = "provided filepath does not exist"
            logging.error(message)
            raise Exception(message)
        if not fp.is_file():
            message = "provided filepath does not points to a file"
            logging.error(message)
            raise Exception(message)
        if fp.suffix != ".gguf":
            message = "only gguf files are supported"
            logging.error(message)
            raise Exception(message)

        # load model
        self.llm = Llama(
            model_path=filepath,
            n_ctx=4096,
            chat_format="llama-2"
        )

        # update status
        self.ready = True
        self.model_path = filepath

    def optimize_prompt(
            self,
            prompt: str,
            goal: Optional[str] = "Create beautiful and aesthetically pleasing images",
            creative_mode: bool = False,
            **kwargs
    ) -> str:
        """Optimizes the provided prompt using the specified method."""
        # set instructions
        instructions = "Ensure that your prompt reflects all the information contained in the user's description."
        query = f"Description:\n{prompt}"
        if not creative_mode:
            instructions += """
Do not add elements of your own and only use elements extracted from the user's description.
If you cannot specify a part of the provided structure because of this, skip it.
"""
        else:
            instructions = """
Ensure that your prompt specifies all parts of the provided structure.
Fill the gaps left by the user's description by inventing elements that concord with the user's goal.
"""
            query += f"\nGoal:\n{goal}"

        # compute prompt
        output = self._respond(
            prompt=prompt,
            system_prompt=self._build_instructions(instructions),
            stop=["\n"],
            **kwargs
        )

        # apply post-processing
        output = f"{self.rules.get('prefix', '')},{output},{self.rules.get('suffix', '')}"
        output = output.replace(".", ",")
        output = [_ for _ in output.split(',') if len(_) > 0]
        output = ",".join(output)

        return output

    def _respond(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            **kwargs
    ) -> str:
        # consistency checks
        if not self.ready:
            message = "No model loaded. Please call `load_model` to load a model first."
            logging.error(message)
            raise RuntimeError(message)

        # cast conversation history into supported format
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages += [dict(role="system", content=system_prompt)]
        messages += [dict(role="user", content=prompt)]

        # request api
        try:
            response = (
                self.llm
                .create_chat_completion(
                    messages=messages,
                    **kwargs
                )
                .get("choices")[0]
                .get("message")
                .get("content")
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise Exception(message)

        return response

    def _build_instructions(self, instructions: str) -> str:
        """ Builds the system prompt from the prompting rules & provided instructions. """
        return (
            self._prompt_template
            .replace("<instructions>", instructions)
            .replace("<format>", self.rules.get("format", ""))
            .replace("<structure>", self.rules.get("structure", ""))
            .replace("<guidelines>", self.rules.get("guidelines", ""))
            .replace("<examples>", "\n".join(self.rules.get("examples", [])))
        )


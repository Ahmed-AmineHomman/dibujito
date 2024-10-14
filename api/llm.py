import logging
from pathlib import Path
from typing import List, Optional, Dict

from llama_cpp import Llama

from .prompting_rules import PromptingRules


class LLM:
    """
    Class managing the interactions with LLM models.
    """
    _prompt_template: str = """
You are a prompt engineer.
Specifically, you will be provided an image description (prompt) and a goal, generated by a user of a text-to-image diffusion model.
Your job is to craft an optimized prompt from the provided description in order for the diffusion model to generate the most aesthetically pleasing image consistent with the provided goal.
Return your prompt as plain text only, with no additional text, introductions or interpretations.

Here are the prompting rules of the diffusion model you must follow in order to design optimal prompts:

<instructions>

Here are some examples to help you understand what is required of you:
<examples>
"""

    ready: bool
    model_path: str
    llm: Llama

    def __init__(
            self,
            filepath: Optional[str] = None,
    ):
        self.ready = False
        self.model_path = ""

        # load model from file
        if filepath:
            self.load_model(filepath=filepath)

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
        self.llm = Llama(model_path=filepath, n_ctx=0)

        # update status
        self.ready = True
        self.model_path = filepath

    def optimize_prompt(
            self,
            prompt: str,
            rules: PromptingRules,
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

        # compute response
        if len(rules.prefix) > 0:
            yield rules.prefix
        response = self._respond(
            prompt=prompt,
            system_prompt=self._build_instructions(rules=rules, instructions=instructions),
            stop=["\n"],
            **kwargs
        )
        for chunk in response:
            yield chunk
        if len(rules.suffix) > 0:
            yield rules.suffix

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
        response = self.llm.create_chat_completion(messages=messages, stream=True, **kwargs)
        for chunk in response:
            data = chunk.get("choices")[0].get("delta")
            if "content" in data.keys():
                yield data.get("content")

    def _build_instructions(self, rules: PromptingRules, instructions: str) -> str:
        """ Builds the system prompt from the prompting rules & provided instructions. """
        return (
            self._prompt_template
            .replace("<instructions>", instructions)
            .replace("<rules>", rules.rules)
        )

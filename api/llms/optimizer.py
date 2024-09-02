import logging
import os
import tomllib
from typing import Optional, List

from .base import LLM

SYSTEM_PROMPT = """
Turn the image description provided by the user into an optimized prompt for text-to-image diffusion models.
Follow the following rules when crafting your prompts:

---
<rules>
---

In addition of the above rules, make sure to satisfy the constraints below ordered by importance:

1. Clarity: craft each part of your prompt to be direct and descriptive, avoiding unnecessary verbosity.
2. Accuracy: ensure all elements of the user's description are present in the prompt.
3. Technicity: use appropriate technical Photographic, Painting or Artistic terms when relevant.
4. Conciseness: ensure the prompt does not exceeds 50 or so words.

Return your prompt as plain text only, with no additional text, introductions or interpretations.
"""


class PromptOptimizer:
    """
    Prompt optimizer class.
    """
    rules_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "prompting_rules"))
    llm: LLM

    def __init__(
            self,
            llm: LLM,
    ):
        self.llm = llm

    @staticmethod
    def get_supported_models() -> List[str]:
        """
        Returns a list of supported prompting rules.
        """
        rules = os.listdir(PromptOptimizer.rules_dir)
        rules = [f for f in rules if os.path.isfile(os.path.join(PromptOptimizer.rules_dir, f))]
        rules = [f for f in rules if f.endswith(".toml")]
        return [f.split(".")[0] for f in rules]

    def optimize(
            self,
            prompt: str,
            target: str,
            model: Optional[str] = None,
    ) -> str:
        """
        Optimizes the provided prompt for the specified model.

        Parameters
        ----------
        prompt: str,
            The image description to optimize.
        target: str
            Target optimization whose rules to follow when optimizing the prompt.
        model: str, optional
            The LLM to use for optimization.

        Returns
        -------
        str,
            The optimized prompt.

        See Also
        --------
        api.clients.APIClient: base class for API clients.
        """
        if target not in self.get_supported_models():
            message = f"Model {target} not supported"
            logging.error(message)
            raise ValueError(message)

        # retrieve prompting rules for the specified model
        with open(os.path.join(self.rules_dir, f"{target}.toml"), "rb") as fp:
            config = tomllib.load(fp)

        # define system prompt
        rules = f"{config.get('rules')}"
        system_prompt = SYSTEM_PROMPT.replace("<rules>", rules)

        # compute response
        try:
            optimized_prompt = self.llm.respond(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_history=[],
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise ValueError(message)

        # append prefix & suffix
        optimized_prompt = f"{config.get('additionals').get('prefix')} {optimized_prompt} {config.get('additionals').get('suffix')}"

        # post-processing
        optimized_prompt = " ".join([w for w in optimized_prompt.split(" ") if len(w) > 0])

        return optimized_prompt

import logging
import os
import tomllib
from typing import Optional, List

from .clients import BaseClient, ConversationExchange, APIClientFactory

SYSTEM_PROMPT = """
You are an AI assistant tasked with optimizing user-provided image descriptions to create highly effective prompts for a specific text-to-image diffusion model.
Your role in the overall process is to:

1.  Understand the user's project and the intended use of the images.
2.  Follow the prompting rules specific to the chosen diffusion model.
3.  Enhance the image descriptions by adding aesthetic details that are consistent with the user's project.
4.  Return ONLY the optimized prompt without any explanation or introduction.

User's project:

<project>

Prompting rules:

<rules>

Additional Instructions:

1.  If the description leaves room for imagination, fill it with details of your choosing.
2.  Ensure that your optimized prompt is consistent with the user's project, description and satisfy the prompting rules.
3.  If the description is unclear, do your best and return something.
    If the project is unclear, do your best and return something.
    If the prompting rules are unclear, do your best and return something.
    ALWAYS return something.
4.  ONLY return the optimized prompt, without any explanation, introduction or any other text.
"""


class PromptOptimizer:
    """
    Prompt optimizer class.
    """
    rules_dir: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "prompting_rules")
    client: BaseClient

    def __init__(
            self,
            api: str,
    ):
        self.client = APIClientFactory.create(api=api)

    @staticmethod
    def get_supported_rules() -> List[str]:
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
            project: Optional[str] = None,
            model: Optional[str] = None,
    ) -> str:
        """
        Optimizes the provided prompt for the specified model.

        Parameters
        ----------
        prompt: str,
            The image description to optimize.
        target: str
            The name of the diffusion model to optimize for.
        project: str, optional
            The global project description, i.e. the context in which the images are generated or will be used.
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
        if not project:
            project = "Generate beautiful and aesthetic images"
        if target not in self.get_supported_rules():
            message = f"Model {target} not supported"
            logging.error(message)
            raise ValueError(message)

        # retrieve prompting rules for the specified model
        with open(os.path.join(self.rules_dir, f"{target}.toml"), "rb") as fp:
            rules = tomllib.load(fp)

        # define system prompt
        system_prompt = (
            SYSTEM_PROMPT
            .replace("<project>", project)
            .replace("<rules>", rules.get("rules"))
        )

        # compute conversation history
        conversation_history = []
        for i, query in enumerate(rules.get("examples").get("inputs")):
            response = rules.get("examples").get("outputs")[i]
            conversation_history += [ConversationExchange(query=query, response=response)]

        # compute response
        try:
            optimized_prompt = self.client.respond(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise ValueError(message)

        # append prefix & suffix
        optimized_prompt = f"{rules.get('additionals').get('prefix')} {optimized_prompt} {rules.get('additionals').get('suffix')}"

        # post-processing
        optimized_prompt = " ".join([w for w in optimized_prompt.split(" ") if len(w) > 0])

        return optimized_prompt

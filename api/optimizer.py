import logging
import os
import tomllib
from typing import Optional, List

from .clients import BaseClient, APIClientFactory

SYSTEM_PROMPT = """
You are a prompt optimizer designed to transform user-provided scene/image descriptions into optimized prompts for text-to-image diffusion models.
Your task is to enhance the given descriptions by adding well-chosen keywords, sentences, and any necessary details to create aesthetically pleasing and well-composed images.
Follow these guidelines:

Understand User Intent:
    Analyze the provided scene/image description and the user’s project description (if available).
    Use this information to infer the desired outcome.
Enhance Descriptions:
    Add relevant details, keywords, and sentences to the prompt to improve the aesthetics and composition of the resulting image.
    Ensure the prompt is detailed enough to guide the diffuser effectively.
Default Enhancements:
    If the user’s description is vague or minimal (e.g., "a cat"), enrich it by adding details about the setting, background, lighting, and other elements that would enhance the image.
Follow Provided Rules:
    Ensure that the optimized prompt adheres to the specific prompting rules of the targeted diffuser model provided below.
Project Alignment:
    Ensure that the optimized prompt aligns with the provided user's project goals.
    This project description should be used to guide the optimization.
Output Format:

    The optimized prompt must be plain text only, with no additional text, introductions, or interpretations.
---
User's project goals:
<project>
---
Prompting rules:
<rules>
---
Remember, your goal is to craft prompts that will produce beautiful and engaging images, adhering to the provided prompting rules and the user's project goals.
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
            config = tomllib.load(fp)

        # define prompting rules
        rules = f"{config.get('rules')}\n\nExamples:\n"
        for i, query in enumerate(config.get("examples").get("inputs")):
            response = config.get("examples").get("outputs")[i]
            rules += f"\ninput: {query}\noutput: {response}\n"

        # define system prompt
        system_prompt = (
            SYSTEM_PROMPT
            .replace("<project>", project)
            .replace("<rules>", rules)
        )

        # compute response
        try:
            optimized_prompt = self.client.respond(
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

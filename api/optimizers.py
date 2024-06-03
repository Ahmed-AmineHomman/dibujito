import logging
import os
import tomllib
from typing import Optional

from .llm import LLM

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

    def __init__(
            self,
            api: str,
            api_model: Optional[str] = None
    ):
        self.llm = LLM(api=api, api_model=api_model)

        # get supported rules
        self.rules = dict()
        rule_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "prompting_rules")
        for f in os.listdir(rule_dir):
            if f.endswith("toml"):
                with open(os.path.join(rule_dir, f), "rb") as fp:
                    self.rules[f.split(".")[0]] = tomllib.load(fp)

    def optimize(
            self,
            description: str,
            model: str,
            project: Optional[str] = None
    ) -> str:
        """
        Optimizes the provided prompt for the specified model.

        Parameters
        ----------
        description: str,
            The image description to optimize.
        model: str,
            The name of the diffusion model to optimize for.
            Call ``optimizer_supported_models`` to get a list of the supported models by this method.
        project: str, optional
            The global project description, i.e. the context in which the images are generated or will be used.

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
        if model not in self.rules.keys():
            message = f"Model {model} not supported"
            logging.error(message)
            raise ValueError(message)
        rules = self.rules.get(model)

        # define system prompt
        system_prompt = (
            SYSTEM_PROMPT
            .replace("<project>", project)
            .replace("<rules>", rules.get("rules"))
        )

        # reset model with updated system prompts & examples
        self.llm.reset(system_prompt=system_prompt)
        for i, query in enumerate(rules.get("examples").get("inputs")):
            response = rules.get("examples").get("outputs")[i]
            self.llm.add_exchange(query=query, response=response)

        # optimize
        try:
            response = self.llm.chat(query=description)
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise ValueError(message)

        # append prefix & suffix
        response = f"{rules.get('additionals').get('prefix')} {response} {rules.get('additionals').get('suffix')}"

        # post-processing
        response = " ".join([w for w in response.split(" ") if len(w) > 0])

        return response

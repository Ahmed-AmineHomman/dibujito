import logging
from typing import Dict, List, Optional, Tuple

from api.clients import BaseClient, ConversationExchange
import os
from tomllib import load

class LLM:
    """
    Class managing the interactions with LLM models.
    """
    _rules_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "prompting_rules"))

    client: BaseClient


    def __init__(
            self,
            client: BaseClient,
    ):
        self.client = client

    @staticmethod
    def get_supported_rules() -> List[str]:
        """ Returns a list of supported prompting rules. """
        rules = os.listdir(LLM._rules_dir)
        rules = [f for f in rules if os.path.isfile(os.path.join(LLM._rules_dir, f))]
        rules = [f for f in rules if f.endswith(".toml")]
        return [f.split(".")[0] for f in rules]

    @staticmethod
    def get_supported_models(
            api: Optional[str] = None
    ) -> List[str]:
        if api:
            return [k for k, v in LLM._client_mapper.items() if v[0] == api]
        else:
            return [k for k in LLM._client_mapper.keys()]

    def optimize_prompt(
            self,
            prompt: str,
            rules: str,
            **kwargs
    ) -> str:
        """Optimizes the provided prompt using the specified method."""
        _template: str = """
Turn the image description provided by the user into an optimized prompt for text-to-image diffusion models.
Follow the rules below when crafting your prompts:

---
<rules>
---

In addition of the above rules, make sure to satisfy the constraints below ordered by importance:

1. Clarity: Be direct and descriptive. Avoid unnecessary words.
2. Accuracy: Reflect the user's description precisely, without omissions.
3. Conciseness: Aim for around 50 words or fewer.
4. Technicality: Use relevant technical terms from photography or art where appropriate.

Return your prompt as plain text only, with no additional text, introductions or interpretations.
"""

        # consistency checks
        if rules not in self.get_supported_rules():
            message = f"Rules {rules} not supported"
            logging.error(message)
            raise ValueError(message)

        # retrieve prompting rules for the specified model
        with open(os.path.join(self._rules_dir, f"{rules}.toml"), "rb") as fp:
            config = load(fp)
        rules = f"{config.get('rules')}"

        # compute response
        try:
            response = self.client.respond(
                prompt=prompt,
                system_prompt=_template.replace("<rules>", rules),
                **kwargs
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise ValueError(message)

        return response

    def expand_prompt(
            self,
            prompt: str,
            goal: str,
            **kwargs
    ) -> str:
        """Expands the provided prompt with additional details."""
        _template: str = """
Enhance the user provided image or scene descriptions by adding appropriate details.
If the user also provides an end goal, take it into account when designing your description.
Return a single natural language paragraph detailing all the elements below:

- Subject: the primary focus of the image (e.g., person, animal, object).
- Action: what the subject is doing, adding dynamism or narrative.
- Environment/Setting: the background or scene surrounding the subject.
- Style: The artistic style (eg. oil painting, charcoal drawing, etc.) or method of rendering (photography, digital art, etc.).
- Color: Dominant colors or color schemes.
- Mood/Atmosphere: The emotional or atmospheric quality.
- Lighting: Specific lighting conditions or effects.
- Perspective/Viewpoint: The angle or perspective from which the scene is viewed.

In addition, ensure to satisfy the constraints below ordered by importance:

1. Clarity: craft each part of your prompt to be direct and descriptive, avoiding unnecessary verbosity.
2. Accuracy: ensure your prompt is true to the user's description and misses nothing in it.
3. Creativity: fill in the gaps, i.e. invent details for each element left unspecified by the user's description.
4. Technicality: use appropriate technical Photographic, Painting or Artistic terms when relevant.

Return your prompt as plain text only, with no additional text, introductions or interpretations.
"""

        if not goal:
            goal = "Create detailed and aesthetically pleasing images."

        # define query
        query = f"---\nImage description: {prompt}\n---"
        if goal:
            query += f"End goal: {goal}\n---"

        # compute response
        try:
            response = self.client.respond(
                prompt=query,
                system_prompt=_template,
                **kwargs
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise ValueError(message)

        return response
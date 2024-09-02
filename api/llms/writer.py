import logging
from typing import Optional

from .base import LLM

SYSTEM_PROMPT = """
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
2. Technicity: use appropriate technical Photographic, Painting or Artistic terms when relevant.
3. Accuracy: ensure all elements of the user's description are present in the prompt.
4. Creativity: fill in the gaps, i.e. invent details for each element left unspecified by the user's description.

Return your prompt as plain text only, with no additional text, introductions or interpretations.
"""


class PromptWriter:
    """
    Class responsible for enhancing user's prompts.
    """
    llm: LLM

    def __init__(
            self,
            llm: LLM,
    ):
        self.llm = llm

    def optimize(
            self,
            prompt: str,
            goal: Optional[str],
            model: Optional[str] = None,
    ) -> str:
        """
        Enhances the provide prompt according to the provided goal.

        Parameters
        ----------
        prompt: str,
            The image description to enhance.
        goal: str,
            The user's goal for the end images
        model: str
            The LLM to use for optimization.

        Returns
        -------
        str,
            The enhanced prompt.

        See Also
        --------
        api.clients.APIClient: base class for API clients.
        """
        if not goal:
            goal = "Create detailed and aesthetically pleasing images."

        # define query
        query = f"---\nImage description: {prompt}\n---"
        if goal:
            query += f"End goal: {goal}\n---"

        # compute response
        try:
            optimized_prompt = self.llm.respond(
                model=model,
                prompt=query,
                system_prompt=SYSTEM_PROMPT,
                conversation_history=[],
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise ValueError(message)

        return optimized_prompt

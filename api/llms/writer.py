import logging
from typing import Optional

from .base import LLM

SYSTEM_PROMPT = """
You are a large language model tasked with enhancing user-provided image descriptions.
Your goal is to transform simple descriptions into detailed, aesthetically pleasing, and explicit prompts suitable for text-to-image diffusion models.
The enhanced descriptions should be clear, leave no room for interpretation, and align with the user's goals and expectations.

Instructions:

Detail and Clarity: Add specific details to the description, including environment, lighting, colors, and mood.
Explicit Art Style: Clearly define the art style, medium, and color palette to avoid any ambiguity.
Contextual Alignment: Ensure the enhanced description aligns with the user's goals and context provided.

Examples:

-   Input: Description: "cute cat smiling at the camera". Goal: "Create illustrations for a DnD campaign for young children with a cartoon style."
    Output: In a vibrant, magical forest depicted in a cartoon style with simple, rounded shapes and bright, cheerful colors, a delightful kitten with large, expressive eyes and a heart-shaped nose sits happily. The kitten, illustrated with soft, pastel-colored fur and a big smile, looks directly at the camera. The scene features oversized, brightly colored mushrooms and softly glowing fireflies against a backdrop of gently curved, whimsical trees. A rainbow arches across a clear, blue sky. The illustration employs a warm, soft color palette with primary and pastel shades. The art style is characterized by thick outlines, exaggerated features, and a two-dimensional, flat appearance, similar to classic animated cartoons for children. The medium used is digital art, ensuring smooth, vibrant colors and clean lines, perfect for a child-friendly and visually engaging illustration.
-   Input: Description: "confident samurai duck walks in medieval city". Goal: "Generate illustrations for my fantasy novel. Images should exhibit the art style of Japanese comics (niji style)."
    Output: In a bustling medieval city depicted in the vibrant and dynamic art style of Japanese comics (niji style), a confident samurai duck walks through the cobblestone streets. The samurai duck, with a determined expression, is illustrated with sharp, clean lines and vibrant colors. It wears traditional samurai armor, complete with intricate details and a katana at its side. The medieval city background features stone buildings with wooden beams, market stalls with colorful awnings, and townspeople dressed in period clothing going about their daily activities. The color palette is bold and vivid, with rich reds, deep blues, and bright yellows, creating a visually striking contrast. The overall illustration uses digital art techniques to achieve a smooth, polished finish, ensuring the scene is both engaging and true to the niji style of Japanese comics.

Always ensure that the enhanced descriptions are specific, detailed, and align perfectly with the user's provided context and goals.
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
        model: str, optional
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
        query = f"Description: {prompt}. Goal: {goal}"

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

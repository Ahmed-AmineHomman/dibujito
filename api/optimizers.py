import logging
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
EXAMPLES = [
    "Woman wearing a long and elegant black dress dances alone in a deserted ballroom. The woman is dressed like a princess, and the ballroom is reminiscent of a fairy tale palace. The scene takes place at nigh.",
    "A catmen musketeer, with complete blue musketeer uniform, walks confidently in the streets of renaissance paris by night.  The streets are deserted, the lighting is scarce, and the overall vibe is slightly oppressive: by looking at the picture, one has the feeling that something dangerous is about to happen to the cat. But the catmen looks strong, comfident and looks like he can overcome any difficulty. The picture is a fantasy character illustration such as one can find for DnD characters.",
]
TEMPLATES = {
    "sd1": {
        "rules": """
1. Structure: follow this format: [subject][medium][style][artists][resolution][additional details].
2. Keywords: use comma-separated keywords to ensure clarity and model compatibility.
3. Components:
    - Subject: Describe the main subject, including its pose and action. Example: "forest at sunrise, deer, birds".
    - Medium: Specify the material or method of artwork. Example: "digital art, oil painting, photography".
    - Style: Indicate the artistic style. Example: "surrealism, impressionism, fantasy".
    - Artists: Reference famous artists to influence the style. Example: "in the style of Van Gogh, anime, photorealism".
    - Resolution: Define the sharpness and detail level. Example: "sharp, highly detailed".
    - Additional Details: Include color palette, lighting, and overall vibe. Example: "pastel colors, golden hour, dramatic lighting, bucolic".
4. Emphasis and Weighting: use parentheses to add emphasis or adjust weighting. Example: "(vibrant) flowers, (lush) greenery".
5. Clarity: Ensure the prompt is concise and free from unnecessary complexity.
""",
        "examples": [
            "woman in a long, elegant black dress, dancing alone in a deserted ballroom, digital art, fantasy style, in the style of a fairy tale palace, night scene, highly detailed, dramatic lighting, princess-like, mystical ambiance, shimmering moonlight",
            "catmen musketeer, complete blue musketeer uniform, walks confidently in the deserted streets of Renaissance Paris by night, digital art, fantasy character illustration, in the style of DnD characters, sharp, highly detailed, scarce lighting, slightly oppressive vibe, something dangerous about to happen, strong and confident, capable of overcoming any difficulty"
        ]
    }
}


def optimize(
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
    optimizer_supported_models: returns a list of the supported models by the ``optimize`` method.
    """
    if not project:
        project = "Generate beautiful and aesthetic images"
    if model not in TEMPLATES.keys():
        message = f"Model {model} not supported"
        logging.error(message)
        raise ValueError(message)

    # define system prompt
    system_prompt = (
        SYSTEM_PROMPT
        .replace("<project>", project)
        .replace("<rules>", TEMPLATES[model]["rules"])
    )

    # load model
    llm = LLM(system_prompt=system_prompt)

    # add examples
    for i, query in enumerate(EXAMPLES):
        response = TEMPLATES.get(model).get("examples")[i]
        llm.add_exchange(query=query, response=response)

    # optimize
    try:
        response = llm.chat(query=description)
    except Exception as error:
        message = f"error during API call: {error}"
        logging.error(message)
        raise ValueError(message)

    return response

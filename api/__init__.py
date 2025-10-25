from .diffuser import Diffuser
from .prompting_rules import PromptingRules, get_supported_rules
from .llm import LLM


def get_supported_optimizers() -> list[str]:
    """Return the prompting rule presets bundled with the application.

    Returns
    -------
    list[str]
        Prompting rule file names that ship with the app.
    """
    return get_supported_rules()


def get_supported_image_ratios() -> list[str]:
    """Return the list of aspect ratios supported by the diffusion backend.

    Returns
    -------
    list[str]
        Aspect ratio identifiers recognisable by the diffuser.
    """
    return Diffuser.get_supported_aspects()

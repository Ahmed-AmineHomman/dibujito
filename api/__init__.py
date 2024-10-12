from .diffuser import Diffuser
from .prompting_rules import PromptingRules
from .llm import LLM


def get_supported_optimizers() -> list[str]:
    return LLM.get_supported_rules()


def get_supported_image_ratios() -> list[str]:
    return Diffuser.get_supported_aspects()

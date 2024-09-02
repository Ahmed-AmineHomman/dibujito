import logging
from typing import Dict, List, Optional

from api.clients import BaseClient, APIClientFactory, ConversationExchange


class LLM:
    """
    Class managing the interactions with LLM models.
    """
    client_mapper: Dict[str, BaseClient] = {
        "command r": ("cohere", "command-r"),
        "command r+": ("cohere", "command-r-plus"),
        "gpt4o mini": ("openai", "gpt-4o-mini"),
        "gpt4o": ("openai", "gpt-4o"),
    }
    client: BaseClient
    model: str
    api: str

    @staticmethod
    def get_supported_models() -> List[str]:
        return list(LLM.client_mapper.keys())

    def respond(
            self,
            model: str,
            prompt: str,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
    ) -> str:
        if model not in self.get_supported_models():
            message = f"Unsupported LLM model '{model}'"
            logging.error(message)
            raise ValueError(message)
        api, model = self.client_mapper.get(model)
        return (
            APIClientFactory
            .create(api)
            .respond(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                conversation_history=conversation_history,
            )
        )

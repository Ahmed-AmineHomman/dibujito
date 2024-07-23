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
    }
    client: BaseClient
    model: str
    api: str

    @staticmethod
    def get_supported_models() -> List[str]:
        return list(LLM.client_mapper.keys())

    def load_model(self, model: str) -> None:
        """Loads the specified LLM model."""
        if model not in LLM.get_supported_models():
            message = f"Unsupported LLM model '{model}'"
            logging.error(message)
            raise ValueError(message)

        self.api, self.model = LLM.client_mapper[model]
        self.client = APIClientFactory.create(self.api)

    def respond(
            self,
            model: str,
            prompt: str,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
    ) -> str:
        return self.client.respond(
            prompt=prompt,
            model=self.model,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
        )

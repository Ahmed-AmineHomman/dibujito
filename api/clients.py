import logging
import os
from typing import List, Optional

from PIL import Image
from cohere import Client as CohereClient, ChatMessage as CohereChatMessage


class ConversationExchange:
    """Implements an exchange in the conversation between the user and the assistant."""
    query: str
    response: str

    def __init__(self, query: str, response: str):
        self.query = query
        self.response = response


class APIClient:
    """
    Base class for API clients.
    """
    _environment_key: str

    def __init__(self, api_key: Optional[str] = None):
        if (not api_key) and (not os.environ.get(self._environment_key)):
            raise ValueError("No API key provided.")
        self.api_key = api_key if api_key else os.environ.get(self._environment_key)

    def respond(
            self,
            prompt: str,
            image_prompt: Optional[Image] = None,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
            **kwargs
    ) -> str:
        """"""
        raise NotImplementedError

    def draw(
            self,
            prompt: str,
            negative_prompt: str,
            image_prompt: Image,
            **kwargs
    ) -> Image:
        raise NotImplementedError


class CohereAPIClient(APIClient):
    """
    Cohere API client.
    """
    _environment_key = "COHERE_API_KEY"

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key)
        self.client = CohereClient(api_key)

    def respond(
            self,
            prompt: str,
            image_prompt: Optional[Image] = None,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
            model: str = "command-r",
            **kwargs
    ) -> str:
        if not conversation_history:
            conversation_history = []

        # cast conversation history into supported format
        chat_history = []
        for exchange in conversation_history:
            chat_history += [
                CohereChatMessage(role="USER", message=exchange.query),
                CohereChatMessage(role="ASSISTANT", message=exchange.response)
            ]

        # request api
        try:
            response = (
                self.client.chat(
                    message=prompt,
                    model=model,
                    preamble=system_prompt,
                    chat_history=chat_history,
                )
                .text
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise Exception(message)

        return response
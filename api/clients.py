import logging
import os
from typing import List, Optional

from PIL import Image
from cohere import Client as CohereClient, ChatMessage as CohereChatMessage
from ollama import Client as OllamaClient

from .config import OLLAMA_HOST_ENV_NAME, COHERE_API_KEY_ENV_NAME


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

    def respond(
            self,
            model: str,
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
            model: str,
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

    def __init__(self):
        self.client = CohereClient(api_key=os.getenv(COHERE_API_KEY_ENV_NAME))

    def respond(
            self,
            prompt: str,
            model: str = "command-r",
            image_prompt: Optional[Image] = None,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
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


class OllamaAPIClient(APIClient):
    """
    Ollama API client.
    """

    def __init__(self):
        self.client = OllamaClient(host=os.getenv(OLLAMA_HOST_ENV_NAME))

    def respond(
            self,
            prompt: str,
            model: str = "phi3",
            image_prompt: Optional[Image] = None,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
            **kwargs
    ) -> str:
        if not conversation_history:
            conversation_history = []

        # cast conversation history into supported format
        messages = [dict(role="system", content=system_prompt)]
        for exchange in conversation_history:
            messages += [
                dict(role="user", content=exchange.query),
                dict(role="assistant", content=exchange.response)
            ]
        messages.append(dict(role="user", content=prompt))

        # request api
        try:
            response = (
                self.client.chat(model=model, messages=messages, )
                ["message"]
                ["content"]
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise Exception(message)

        return response


class APIClientFactory:
    """
    Factory class for API clients.
    """
    _clients = {
        "cohere": CohereAPIClient,
        "ollama": OllamaAPIClient,
    }

    @staticmethod
    def get_supported_clients() -> List[str]:
        """
        Returns a list of supported clients.
        """
        return list(APIClientFactory._clients.keys())

    @staticmethod
    def create(
            api: str,
            **kwargs
    ) -> APIClient:
        """
        Creates an API client.
        """
        if api not in APIClientFactory.get_supported_clients():
            message = f"Client {api} not supported."
            logging.error(message)
            raise ValueError(message)
        return APIClientFactory._clients[api](**kwargs)

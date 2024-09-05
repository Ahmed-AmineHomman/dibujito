import logging
from typing import List

from api.clients.base import BaseClient
from api.clients.cohere import CohereClient
from api.clients.ollama import OllamaClient
from api.clients.openai import OpenAIClient
from api.clients.rwkv import RWKVClient


class APIClientFactory:
    """
    Factory class for API clients.
    """
    _clients = {
        "cohere": CohereClient,
        "openai": OpenAIClient
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
    ) -> BaseClient:
        """
        Creates an API client.
        """
        if api not in APIClientFactory.get_supported_clients():
            message = f"Client {api} not supported."
            logging.error(message)
            raise ValueError(message)
        return APIClientFactory._clients[api](**kwargs)

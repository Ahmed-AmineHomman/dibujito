import logging
import os
from typing import Optional, List

from PIL import Image
from ollama import Client

from api.clients import BaseClient, ConversationExchange


class OllamaClient(BaseClient):
    """
    Ollama API client.
    """
    name: str = "ollama"

    def __init__(self):
        self.client = Client(host=os.getenv("OLLAMA_HOST"))

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
                self.client.chat(model=model, messages=messages)
                ["message"]
                ["content"]
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise Exception(message)

        return response

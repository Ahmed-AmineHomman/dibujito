import logging
import os
from typing import List, Optional

from PIL import Image
from cohere import Client, ChatMessage

from .base import ConversationExchange, BaseClient


class CohereAPIClient(BaseClient):
    """
    Cohere API client.
    """

    def __init__(self):
        self.client = Client(api_key=os.getenv("COHERE_API_KEY"))

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
                ChatMessage(role="User", message=exchange.query),
                ChatMessage(role="Chatbot", message=exchange.response)
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

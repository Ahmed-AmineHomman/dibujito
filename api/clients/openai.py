import logging
import os
from typing import List, Optional, Dict

from PIL import Image
from openai import Client

from .base import ConversationExchange, BaseClient


class OpenAIClient(BaseClient):
    """
    OpenAI API client.
    """

    def __init__(self):
        self.client = Client(api_key=os.getenv("OPENAI_API_KEY"))

    def respond(
            self,
            prompt: str,
            model: str = "gpt-4o-mini",
            image_prompt: Optional[Image] = None,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
            **kwargs
    ) -> str:
        if not conversation_history:
            conversation_history = []

        # cast conversation history into supported format
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages += [
                dict(role="system", content=system_prompt)
            ]
        for exchange in conversation_history:
            messages += [
                dict(role="user", message=exchange.query),
                dict(role="assistant", message=exchange.response)
            ]
        messages += [
            dict(role="user", content=prompt)
        ]

        # request api
        try:
            response = (
                self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                .choices[0]
                .message
                .content
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise Exception(message)

        return response

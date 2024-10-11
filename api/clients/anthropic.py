import logging
import os
from typing import List, Optional, Dict

from PIL import Image
from anthropic import Anthropic as Client

from .base import ConversationExchange, BaseClient


class AnthropicClient(BaseClient):
    """
    Anthropic API client.
    """
    name: str = "anthropic"

    def __init__(self):
        self.client = Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def respond(
            self,
            prompt: str,
            model: str = "claude-3-haiku-20240307",
            image_prompt: Optional[Image] = None,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
            max_tokens: int = 1024,
            **kwargs
    ) -> str:
        if not conversation_history:
            conversation_history = []

        # cast conversation history into supported format
        messages: List[Dict[str, str]] = []
        for exchange in conversation_history:
            messages += [
                dict(role="user", content=exchange.query),
                dict(role="assistant", content=exchange.response)
            ]
        messages += [
            dict(role="user", content=prompt)
        ]

        # request api
        try:
            response = (
                self.client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                    **kwargs
                )
                .content[0]
                .text
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise Exception(message)

        return response

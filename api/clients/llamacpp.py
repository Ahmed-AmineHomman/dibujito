from llama_cpp import Llama as Client

import logging
import os
from typing import List, Optional, Dict

from PIL import Image

from .base import ConversationExchange, BaseClient


class LlamaCppClient(BaseClient):
    """
    Llama.cpp API client.
    """
    name: str = "llama.cpp"

    def __init__(self, model_path: str):
        self.client = Client(
            model_path=model_path,
            n_ctx=4096,
            chat_format="llama_2"
        )

    def respond(
            self,
            prompt: str,
            model: Optional[str] = None,
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
                dict(role="user", content=exchange.query),
                dict(role="assistant", content=exchange.response)
            ]
        messages += [
            dict(role="user", content=prompt)
        ]

        # request api
        try:
            response = (
                self.client
                .create_chat_completion(
                    messages=messages,
                    **kwargs
                )
                .get("choices")[0]
                .get("message")
                .get("content")
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise Exception(message)

        return response

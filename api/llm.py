import logging
import os
from typing import Optional, List

from cohere import Client, ChatMessage


class LLM:
    """Class implementing the LLM."""
    _default_system = "You are a helpful assistant."
    _environment_key = "COHERE_API_KEY"

    def __init__(
            self,
            api_key: Optional[str] = None,
            system_prompt: Optional[str] = None
    ):
        if (not api_key) and (not os.environ.get(self._environment_key)):
            raise ValueError("No API key provided.")

        # create API client
        self.client = Client(api_key=api_key if api_key else os.environ.get(self._environment_key))
        self.conversation: List[ChatMessage] = []
        self.system_prompt = system_prompt if system_prompt else self._default_system

    def reset(
            self,
            system_prompt: Optional[str] = None
    ) -> None:
        """
        Resets the conversation history.

        If ``system_prompt`` is provided, it will override the existing system prompt.
        """
        self.conversation = []
        if system_prompt:
            self.system_prompt = system_prompt

    def chat(
            self,
            query: str
    ) -> str:
        """
        Responds to the provided query.

        **Note**: the underlying model will use the conversation history when responding to the query.
        """
        # api call
        try:
            response = (
                self.client.chat(
                    message=query,
                    model="command-r-plus",
                    preamble=self.system_prompt,
                    chat_history=self.conversation,
                )
                .text
            )
        except Exception as error:
            message = f"error during API call: {error}"
            logging.error(message)
            raise Exception(message)

        # update conversation history
        self.add_exchange(query=query, response=response)

        return response

    def add_exchange(self, query: str, response: str):
        """Adds the provided exchange to the conversation history"""
        self.conversation += [
            ChatMessage(role="USER", message=query),
            ChatMessage(role="CHATBOT", message=response)
        ]

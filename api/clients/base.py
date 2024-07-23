from typing import Optional, List


class ConversationExchange:
    """Implements an exchange in the conversation between the user and the assistant."""
    query: str
    response: str

    def __init__(self, query: str, response: str):
        self.query = query
        self.response = response


class BaseClient:
    """
    Base class for API clients.
    """

    def respond(
            self,
            model: str,
            prompt: str,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
            **kwargs
    ) -> str:
        """"""
        raise NotImplementedError

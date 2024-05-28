from typing import Optional, List

from .clients import CohereAPIClient, ConversationExchange


class LLM:
    """Class implementing the LLM."""
    _default_system = "You are a helpful assistant."
    _conversation: List[ConversationExchange]
    _supported_apis = ["cohere"]

    def __init__(
            self,
            api: str,
            api_key: Optional[str] = None,
            api_model: Optional[str] = None,
            system_prompt: Optional[str] = None,
    ):
        # create API client
        if api == "cohere":
            self.client = CohereAPIClient(api_key=api_key)
        self.conversation = []
        self.api_model = api_model
        self.system_prompt = system_prompt if system_prompt else self._default_system

    def reset(
            self,
            system_prompt: Optional[str] = None,
            api_model: Optional[str] = None
    ) -> None:
        """
        Resets the conversation history.

        If ``system_prompt`` is provided, it will override the existing system prompt.
        If ``api_model`` is provided, it will override the existing API model.
        """
        self.conversation = []
        if system_prompt:
            self.system_prompt = system_prompt
        if api_model:
            self.api_model = api_model

    def chat(
            self,
            query: str
    ) -> str:
        """
        Responds to the provided query.

        **Note**: the underlying model will use the conversation history when responding to the query.
        """
        response = self.client.respond(
            prompt=query,
            conversation_history=self.conversation,
            system_prompt=self.system_prompt,
            model=self.api_model
        )
        self.add_exchange(query=query, response=response)
        return response

    def add_exchange(
            self,
            query: str,
            response: str
    ):
        """Adds an exchange to the conversation history."""
        self.conversation.append(ConversationExchange(query=query, response=response))

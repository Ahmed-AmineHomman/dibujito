from .base import BaseClient, ConversationExchange
from .factory import APIClientFactory


def create_client(
        api: str
) -> BaseClient:
    """
    Creates the provided API client.

    Parameters
    ----------
    api: str
        The API label whose client should be returned.

    Returns
    -------
    BaseClient:
        Instance of the API client corresponding to the specified API.
    """
    return APIClientFactory.create(api=api)

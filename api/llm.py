from cohere import Client, ChatMessage

from app import KEYS

SYSTEM_PROMPT = """
As a professional artist and photographer, reformulate the provided verbose image description into one, potentially very long, sentence.
Use powerful keywords rather than phrases.
Organize your prompt according to the following architecture:

[subject][medium][style][artists][resolution][additional details].

- subject: description of the main subject of the image, its pose and its action,
- medium: material used to make artwork (oil painting, digital art, photography, etc..),
- style: artistic style (surrealism, impressionism, fantasy, etc..)
- artists: examples of famous artists of the image style,
- resolution: how sharp and detailed the image is (sharp, very detailed, etc..),
- additional details: color palette (pastel, golden hour, etc..), lighting (dramatic, cinematic, etc..), vibe (dystopian, bucolic, etc...).

Here is one example of such prompt:

"a beautiful and powerful mysterious sorceress, smile, sitting on a rock, lightning magic, hat, detailed leather clothing with gemstones, dress, castle background, digital art, hyperrealistic, fantasy, dark art, artstation, highly detailed, sharp focus, sci-fi, dystopian, iridescent gold, studio lighting."

Ensure to return a prompt whose keywords are consistent with the provided description.
Add keywords that you think enhance the associated image if they are consistent with the provided description.
Use your prior knowledge to find artists.
Use very technical terms from professional photography.
Only return your summarized description.
"""
EXAMPLES = [
    {
        "input": "A woman waling in an outdoor marketplace on a sunday morning",
        "output": "A woman, walking through a bustling marketplace, Sunday morning light, oil painting, impressionist style, medium close-up, reminiscent of Renoir and Van Gogh, warm color palette, soft focus, with bright sunlight streaming through, capturing the lively atmosphere.",
    }
]


def optimize(prompt: str) -> str:
    """Optimizes the provided prompt for ingestion by a text-to-image model."""
    # initialize API client
    client = Client(api_key=KEYS.get("cohere"))

    # build fictitious exchange with examples
    history = []
    for example in EXAMPLES:
        history += [
            ChatMessage(role="USER", message=example.get("input")),
            ChatMessage(role="CHATBOT", message=example.get("input")),
        ]

    # retrieve response
    response = client.chat(
        message=prompt,
        model="command-r-plus",
        preamble=SYSTEM_PROMPT,
        chat_history=history,
    )

    return response.text

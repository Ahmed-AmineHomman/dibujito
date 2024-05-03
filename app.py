import requests

import gradio as gr
from cohere import Client, ChatMessage
from PIL import Image
from os import environ
from io import BytesIO

APP_NAME = "Ijeomman Diffusion"
APP_DESCRIPTION = """
Une interface web permettant de générer toutes les images imaginables, à utiliser en famille !
Commence par décrire ce que tu souhaites, et l'application fera le reste.

Quelques conseils:

- **Ecris en anglais**: les modèles ne comprennent pas d'autres langues :-(,
- Fournis des détails : plus tu donneras des détails sur l'image que tu as en tête, meilleur sera le résultat.
"""
KEYS = {"cohere": environ.get("COHERE_API_KEY"), "hf": environ.get("HF_API_KEY")}
SYSTEM_PROMPT = """
As a professional artist and photographer, reformulate the provided verbose image description into one, potentially very long, sentence formed by comma-separated powerful keywords.
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


def imagine(prompt: str) -> Image:
    """Generates an image corresponding to the provided prompt."""
    headers = {"Authorization": f"Bearer {KEYS.get('hf')}"}
    url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    payload = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=payload)
    return Image.open(BytesIO(response.content))


def generate(prompt: str) -> Image:
    """Generates the image corresponding to the provided prompt."""
    # optimize prompt
    try:
        optimized_prompt = optimize(prompt)
    except Exception as error:
        gr.Error(f"Error (prompt optim): {error}")

    # generate image
    try:
        image = imagine(optimized_prompt)
    except Exception as error:
        gr.Error(f"Error (image gen): {error}")

    return image, optimized_prompt


if __name__ == "__main__":
    # define UI
    with gr.Blocks() as app:
        gr.Markdown(f"# {APP_NAME}\n\n{APP_DESCRIPTION}")
        with gr.Row():
            image = gr.Image(
                label="Image", format="png", height=512, width=512, scale=4
            )
            with gr.Accordion(label="Additional Details", open=False):
                opt_prompt = gr.TextArea(label="Optimized prompt")
        with gr.Row():
            prompt = gr.TextArea(
                label="Description",
                placeholder="décris ce que tu souhaites générer ici...",
                scale=4,
            )
            button = gr.Button(value="Générer", variant="primary", scale=1)
        button.click(fn=generate, inputs=[prompt], outputs=[image, opt_prompt])

    # run app
    app.launch(inbrowser=False, share=True)

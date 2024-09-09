# Dibujito

Welcome to Dibujito, a simple web app for llm-powered diffusion!

This app combines the power of both *LLMs* (generative AI text-to-text models) and Diffusers (generative AI
text-to-image models) in order to bring your ideas to life!
Forget about prompt engineering or complex prompt syntaxes: the LLM plugged to your prompt will convert whatever you
write into an optimized and accurate prompt, allowing the diffuser to craft a beautiful image corresponding to what you
wrote.

## Prerequisites

Dibujito uses local hugging face hub diffusers for the image generation part, but can use cloud-hosted LLMs from the
most popular APIs (OpenAI, Cohere, etc.).
If using such services, you will need to create an API token identifying you to the API.
You can create it in the account setting of the targeted API.

**Note**: using such APIs is rarely free.
You will probably need to provide a payment mean prior to using the API models.

## Quickstart

Prior to using Dibujito, you must first install, then configure it.

### Installation

Simply clone this repository on your system and install the dependencies specified
in [requirements.txt](requirements.txt):

```shell
pip install -r requirements.txt
```

**Note**: it is advised to first configure a virtual environment prior to installing these.

### Configuration

Configuring Dibujito is done by copying the [config_example.toml](config_example.toml) file in the root directory of the
solution and rename it to ``config.toml``.
You must then set the variables defined in the file to their appropriate values.
Follow the instructions in the file in order to do so.

Once the configuration file is properly set-up, you can run the app simply by calling [app.py](app.py) with your python
interpreter:

```shell
python app.py
```

## Local Use

Dibujito allows for a fully local use of generative AI models by supporting the [ollama API](https://ollama.com).
This tool allows use many popular open-source LLMs locally on your PC.
Ollama is based on [`llama.cpp`](https://github.com/ggerganov/llama.cpp), which is a library providing C++
implementations of many LLMs.
Since these versions are implemented in C++ without any dependency, they are optimized for speed and can even decently
run on your RAM and CPU, without the need of a GPU (for larger models, it is strongly recommended though).
Therefore, you can combine both Ollama for the LLM prompt-optimization and Hugging Face for the text-to-image generation
and enjoy a fully local experience.

In order to use the ``ollama`` API, start by running a local server on your PC (or any other PC you own).
Then set the corresponding variables in the ``config.toml`` file to the ``"ollama"`` API.
Also provide the hostname of your server (``"localhost"`` if the server is running on the same computer).
Once this is done, you're all set and ready to go!
Simply run the app and enjoy!

## Recommendend Specs

Since the text-to-image part is always performed in your PC, a decent GPU is required in order to be able to generate
images with a reasonable latency.
Dibujito supports natively the following diffusion models of various sizes:

| model                                                                              | architecture        | VRAM needed | lowest appropriate nvidia GPU |
|------------------------------------------------------------------------------------|---------------------|-------------|-------------------------------|
| [Dreamshaper](https://huggingface.co/Lykon/dreamshaper-8)                          | Stable Diffusion 1  | 6 Gb        | GTX 1660, RTX 3050,           |
| [Playground](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) | Stable Diffusion XL | 12 Gb       | RTX 3060                      |
| [Juggernaut](https://huggingface.co/RunDiffusion/Juggernaut-XI-v11)                | Stable Diffusion XL | 12 Gb       | RTX 3060                      |


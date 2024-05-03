# Dibujito

Welcome to Dibujito, a simple web app for llm-powered diffusion!

This app allows to boost your prompt via the [Command-R model](https://docs.cohere.com/docs/command-r) from [Cohere](https://cohere.com/) before sending it to [StableDiffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5). Prepare to see the best images you'll ever generate!

## Quickstart

Simply clone this repository on your system and install the dependencies specified in [requirements.txt](requirements.txt):

```shell
pip install -r requirements.txt
```

**Note**: it is advised to first configure a virtual environment prior to installing these.

Then retrieve or create the following API keys:

- **Cohere**: register at the [Cohere Dashboard](https://dashboard.cohere.com) and create a free API key in the [API keys section](https://dashboard.cohere.com/api-keys).
- **HuggingFace**: register at [HuggingFace](https://huggingface.co) and create a free API token in the [Settings](https://huggingface.co/settings/tokens).

Once you have copied both keys somewhere safe, you can run the app simply by calling [app.py](app.py) with your python interpreter and adding both of the above keys to the appropriate environment variables. You can do so with the following commands:

- For Linux users:

    ```shell
    COHERE_API_KEY="my_cohere_key" HF_API_KEY="my_hf_token" python app.py
    ```

- For Windows users:

  ```shell
  $env:COHERE_API_KEY="my_cohere_key"; $env:HF_API_KEY="my_hf_token"; python app.py
  ```

In the above commands, replace the values of the environment variables by the keys you generated previously.
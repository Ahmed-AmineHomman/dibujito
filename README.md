# Dibujito ✨

*Prompt crafting, meet visual imagination.*

Dibujito is a playful yet powerful Gradio app that brings together two worlds: text and image generation. 💬➡️🖼️
It couples a **Large Language Model** (LLM) to help you refine your prompts, with a **diffusion model** that turns them
into images — all within a single interface. Start from a rough idea, chat your way to the perfect prompt, and generate
the corresponding visuals without leaving the app!

---

## 🧠 How It Works (A Quick Technical Tour)

Behind the scenes, Dibujito runs on two main engines:

* A chat assistant powered by [`llama.cpp`](https://github.com/ggml-org/llama.cpp) through the [
  `llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) bindings.
*
A [Stable Diffusion XL pipeline](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline)
from the [`diffusers`](https://huggingface.co/docs/diffusers/index) library.

The choice of `llama.cpp` is deliberate: it provides excellent inference speed even on CPUs, unlike many other
frameworks such as `transformers`. The long-term goal is to allow **decoupled inference**, meaning Dibujito will be able
to run the LLM on the CPU while the diffusion model runs on the GPU — helping you make the most of your hardware. ⚙️
*(This feature is under development — for now, allocation is handled by your system automatically.)*

> 💡 **Note:** Dibujito has only been tested with **Stable Diffusion XL** (SDXL). Other architectures may not work out of
> the box.

---

## 🚀 Getting Started

Clone the repository on your machine and create a virtual environment (recommended):

```bash
python -m venv .venv
```

Then activate it:

```bash
source .venv/bin/activate  # On Windows use: .\.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

🧩 The `--extra-index-url` flag ensures you get the right CUDA-enabled PyTorch build. If your CUDA version differs, check
the [official PyTorch installation page](https://pytorch.org/get-started/locally/) for the correct URL.
You can remove this flag if you plan to run on CPU only (not recommended).

Next, download your models:

* Diffusion models (`.safetensors`) → [Civitai](https://civitai.com/) or [Hugging Face](https://huggingface.co/)
* LLMs (`.gguf`) → [Hugging Face](https://huggingface.co/) and
  the [lmstudio-community collection](https://huggingface.co/lmstudio-community)

Copy the example config file:

```bash
cp config_example.toml config.toml
```

Then edit `config.toml` to point to the folders containing your models.
Once done, launch the app:

```bash
python app.py
```

Open your browser at [http://localhost:7860](http://localhost:7860) — and voilà! Dibujito is ready to turn your words
into visuals. 🌈

---

## ⚠️ Important Notes

* **Models are not included.** You must provide your own `.safetensors` and `.gguf` files.
* **Only SDXL** has been tested so far.
* **Tests performed only on my Nvidia GPU.** AMD and Intel GPU support is untested.
* **All models run locally.** No data leaves your machine. You’ll need a GPU with **≥12 GB VRAM** for smooth diffusion
  inference.
* **Performance tip:** For now, both LLM and diffusion inference share your default compute device. CPU offloading is
  planned for a future release.

---

Enjoy playing with prompts, and happy drawing with Dibujito! 🎨

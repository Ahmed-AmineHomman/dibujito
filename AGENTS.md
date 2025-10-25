# Repository Guidelines

## Project Structure & Module Organization

- `app.py`: launches the interactive Gradio Blocks UI and wires controls to the API helpers.
- `app_api.py`: reusable backend routines (`optimize_prompt`, `generate_image`, `load_model`) shared by the UI and
  external callers.
- `api/`: adapters for LLMs, diffusers, and prompting rules; extend these when integrating new providers.
- `data/`: default resources (`locales/en.toml`, `prompting_rules/*.toml`) that seed the UI; keep overrides in the same
  structure.
- `config.toml`: runtime configuration; copy and tailor from `config_example.toml` before running.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows) to isolate
  dependencies.
- `pip install -r requirements.txt` installs app dependencies.
- `python app.py` starts the local UI using the configured models.

## Coding Style & Naming Conventions

Adhere to idiomatic, type-annotated Python. Always favor code clarity over cleverness. Do not optimise prematurely.

- Use 4-space indentation, PEP 8 naming (`snake_case` for functions, `CapWords` for classes, upper snake for constants).
- Document everything under the numpydoc convention.
- Define function parameters with type hints and return types. Define one parameter per line for functions with multiple
  parameters.
- Group domain logic in `api/`, keep Gradio callbacks in `app.py` thin and define core logic in `app_api.py`.

## Commit & Pull Request Guidelines

- Follow the existing history: short, present-tense commit subjects (e.g., `add streaming prompt updates`); keep bodies
  wrapped at 72 chars.
- Reference related issues in PR descriptions, list config changes, and include screenshots or GIFs when UI updates
  affect the workflow.
- Confirm that `config.toml` and large model binaries stay out of version control; note any manual setup steps for
  reviewers.

## Configuration & Model Assets

- Store `.gguf`, `.safetensors`, and `.toml` optimizer files in the directories referenced by `config.toml` so
  `get_model_list` can discover them.
- Keep API keys and remote endpoints in environment variables read by the config; never hard-code secrets.
- After changing model inventories, restart the UI to refresh dropdown choices sourced from the filesystem.

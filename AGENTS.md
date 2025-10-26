# Repository Guidelines

## Project Structure & Module Organization

- `app.py`: Gradio UI definition and app entry point.
- `app_api.py`: UI callbacks.
- `api/`: main library of the solution, contains core logic for llms, diffusers, prompting rules, etc...
- `data/`: resources directory, contains UI locales & default prompting rules.
- `config.toml`: runtime configuration; copy and tailor from `config_example.toml` before running.

## Build, Test, and Development Commands

- No need to create a virtualenv or isolate dependencies: assume a clean Python 3.12 environment.
- `python -m pip install -r requirements.txt` installs app dependencies.
- `python app.py` starts the local UI using the configured models.
- Currently, no tests have been implemented -> no need to run test commands.

## Coding Style & Naming Conventions

- Adhere to idiomatic, type-annotated Python.
- Always favor code clarity over cleverness.
- Use 4-space indentation, PEP 8 naming (`snake_case` for functions, `CapWords` for classes, upper snake for constants).
- Document everything under the numpydoc convention.
- Define function parameters with type hints and return types.
- Define one parameter per line for functions with multiple parameters.
- Group domain logic in `api/`, define app main callbacks in `app_api.py` and keep Gradio callbacks in `app.py` thin.

## Commit & Pull Request Guidelines

- Follow the existing history: short, present-tense commit subjects (e.g., `add streaming prompt updates`); keep bodies
  wrapped at 72 chars.
- Reference related issues in PR descriptions, list config changes, and include screenshots or GIFs when UI updates
  affect the workflow.
- Confirm that `config.toml` and large model binaries stay out of version control; note any manual setup steps for
  reviewers.

## Configuration & Model Assets

- Keep API keys and remote endpoints in environment variables read by the config; never hard-code secrets.
- After changing model inventories, restart the UI to refresh dropdown choices sourced from the filesystem.

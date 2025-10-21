# MLLM-Shap

This repo now uses **[uv](https://github.com/astral-sh/uv)** for envs/installs and **pre-commit** for formatting/linting.

## Quick start

```bash
ADMEtall.sh | sh

# 2) Set up the project (creates .venv, installs deps + hooks)
make install

# 3) Work on code / notebooks
uv run python -m pip -V
uv run jupyter lab

# 3.5) Or enable venv
source .venv/bin/activate

# 4) Run checks locally
make pre-commit        # staged files
make pre-commit-all    # all files
```

> Target Python: **3.12** (CI uses the same).

## Project layout (workspace)

```
MLLM-Shap/
├─ mllm_shap/            # library (editable)
│  ├─ src/mllm_shap/…
│  └─ pyproject.toml
├─ experiments/           # notebooks / scripts
├─ pyproject.toml         # root deps + uv workspace (not a package)
├─ uv.lock                # lockfile (commit this)
└─ Makefile               # convenience targets
```

The root is **not** a package; the `mllm_shap` package is installed **editable** via the uv workspace.

## Daily commands

### Using the Makefile (recommended)

```bash
make install        # create/update .venv, install all deps, install hooks
make update         # upgrade locked versions within constraints
make pre-commit     # run hooks on staged files
make pre-commit-all # run hooks on all files
make clean          # remove .venv + caches (no conda anymore)
```

### Using uv directly

```bash
uv python install 3.12                   # ensure correct Python
uv sync --all-groups --no-active         # install deps from pyproject/uv.lock
uv run <tool> ...                        # run tools inside project env
```

Examples:

```bash
uv run python -V
uv run python -c "import mllm_shap; print(mllm_shap.__file__)"
uv run jupyter lab
uv run black --version
```

## Pre-commit

Hooks are configured to run tools via `uv run …`, so you don’t need to activate the venv.

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## CI

GitHub Actions installs `uv`, uses Python  **3.12** , syncs deps, and runs the hooks:

* `astral-sh/setup-uv@v6`
* `uv python install 3.12`
* `uv sync --locked --all-groups`
* `uv run pre-commit run --all-files`

Commit `uv.lock` for reproducible CI.

## Troubleshooting

* **“Executable `<tool>` not found” during commit**

  Run `uv sync --all-groups`. Our hooks call `uv run …`; no venv activation required.
* **“VIRTUAL_ENV … does not match .venv” warning**

  We already use `--no-active` in Makefile; safe to ignore.
* **Import path isn’t from the repo**

  Use the env: `uv run python -c "import sys; print(sys.prefix)"` → should end with `/.venv`.

That’s it — happy hacking!

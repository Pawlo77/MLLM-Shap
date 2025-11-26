# MLLM-Shap

This repository uses **[uv](https://github.com/astral-sh/uv)** for environment management and installs, and **pre-commit** for formatting and linting. The guidance below reflects the current monorepo structure and developer workflows.

## Quick start

```bash
# (optional) initial helper (if you have ADMEtall available locally)
ADMEtall.sh | sh

# 1) Set up the project (creates/updates .venv, installs deps + pre-commit hooks)
make install

# 2) Work on code / notebooks inside the project environment
uv run python -V
uv run jupyter lab

# 3) (optional) activate venv manually
source .venv/bin/activate

# 4) Run checks locally
make pre-commit        # run hooks on staged files
make pre-commit-all    # run hooks on all files
```

Target Python: **3.12** (CI uses the same).

## Project layout (workspace)

```
MLLM-Shap/
├─ mllm_shap
│  ├─ docs/                   # Sphinx documentation sources
│  ├─ tests/                  # package tests
│  ├─ src/                    # main package sources
│  │  └─ mllm_shap/           # MLLM SHAP package
│  ├─ LICENSE                 # package license
│  ├─ MANIFEST.in             # package manifest
│  ├─ pyproject.toml          #
package-specific deps + settings
│  ├─ pytest.ini              # test configuration
│  └─ README.md               # package overview
├─ examples/                 # user-facing notebooks and quickstarts
├─ experiments/              # research notebooks, data and experiment scripts
├─ paper/                    # papers, notes, figures
├─ pyproject.toml            # root deps + uv workspace (not a package)
├─ uv.lock                   # locked dependency set
└─ Makefile                  # convenience targets
```

Notes:
- The root is **not** a Python package; the `mllm_shap` package lives under `src/` and is installed editable using `uv`/the workspace tools.
- Sphinx sources and additional docs live under `mllm_shap/docs/` (see Docs section below).

## Daily commands

### Recommended: using the `Makefile`

```bash
make install        # create/update .venv, install deps, install pre-commit hooks
make update         # update lockfile / upgrade pinned versions (if needed)
make pre-commit     # run hooks on staged files
make pre-commit-all # run hooks on all files
make clean          # remove .venv + caches
```

### Using `uv` directly

```bash
uv python install 3.12                    # ensure correct Python is available
uv sync --all-groups --no-active          # install deps from pyproject/uv.lock
uv run <tool> ...                         # run tools inside project env (no activation required)
```

Examples:

```bash
uv run python -V
uv run python -c "import mllm_shap; print(mllm_shap.__file__)"
uv run jupyter lab
uv run black --version
```

## Pre-commit hooks

Hooks are configured to run tools via `uv run …`, so you don’t need to activate the venv manually. Install and run the hooks with:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

If a hook reports a missing executable, run `uv sync --all-groups` to ensure tools are installed in the workspace environment.

## CI

GitHub Actions uses `uv`, installs Python 3.12, syncs dependencies from `uv.lock`, and runs pre-commit hooks. Typical CI steps include:

- `astral-sh/setup-uv@v6`
- `uv python install 3.12`
- `uv sync --locked --all-groups`
- `uv run pre-commit run --all-files`

Keep `uv.lock` committed for reproducible CI runs.

## Troubleshooting

- “Executable `<tool>` not found” during commit

  Run `uv sync --all-groups`. Hooks call `uv run …` so the tool should be available once deps are synced.

- “VIRTUAL_ENV … does not match .venv” warning

  This can happen in some shells; it is usually safe to ignore if you use the `uv`-managed environment. The Makefile uses `--no-active` in many commands for a predictable environment.

- Import path isn’t from the repo

  Verify the active environment: `uv run python -c "import sys; print(sys.prefix)"` → should point to `.../.venv` when run inside the project env.

## Docs — generating and viewing Sphinx API docs

- Generate Sphinx `.rst` sources for the Python packages (run from repository root):

```bash
# create/update rst files for all modules under src/
sphinx-apidoc -o mllm_shap/docs/ src
```

- Build the HTML docs (run from `mllm_shap/docs`):

```bash
cd mllm_shap/docs
make clean html
```

- Open generated docs in your browser (macOS):

```bash
open mllm_shap/docs/_build/html/index.html
```

- Quick scan for README-like files across the repo:

```bash
find . -type f -iname 'readme*'
```

## Contributing notes and conventions

- Prefer module-level docstrings and keep Sphinx sources in `mllm_shap/docs/` for API docs.
- Add a short `README.md` or `README.rst` next to new major modules or packages (for example inside `src/mllm_shap/connectors/`) to give quick usage and design notes.
- Keep user-facing examples in `examples/` as notebooks with clear top-cell instructions.

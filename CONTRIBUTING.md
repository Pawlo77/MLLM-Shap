# Contributing to MLLM-SHAP

<div align="center">
  <p><strong>Thank you for contributing. This guide keeps workflows consistent, reproducible, and review-ready.</strong></p>
</div>

## 🧭 Contribution Standard

- Use `Makefile` targets first.
- Keep changes scoped and documented.
- Run hooks/tests before opening PR.
- Keep `uv.lock` committed when dependencies change.

Target Python: **3.12**.

## ⚡ Quick Start (Make-first)

```bash
# 1) setup env + deps + hooks
make install

# 2) run checks while working
make pre-commit
make pre-commit-all
make tests
```

## 🧰 Daily Commands

```bash
make install        # setup/update env and install hooks
make update         # upgrade lockfile and sync dependencies
make pre-commit     # run hooks on staged files
make pre-commit-all # run hooks on all files
make tests          # run package tests
make clean          # remove venv and caches
```

## 🗂️ Repository Layout

```text
MLLM-Shap/
├─ mllm_shap/              # package, docs, tests
├─ examples/               # usage notebooks
├─ experiments/            # research pipelines and outputs
├─ paper/                  # publication assets
├─ pyproject.toml          # workspace dependencies/groups
├─ uv.lock                 # locked dependency set
└─ Makefile                # primary developer interface
```

Notes:
- workspace root is not publishable package
- package code lives in `mllm_shap/src/mllm_shap/`

## ✅ Pre-commit and CI Expectations

- Local: run `make pre-commit-all` before pushing.
- CI: uses `uv` under the hood, synced from `uv.lock`, and runs pre-commit pipeline.
- If dependency set changes, include updated `uv.lock` in PR.

## 🧪 Documentation Workflow

Generate API docs sources:

```bash
sphinx-apidoc -o mllm_shap/docs/ src
```

Build HTML docs:

```bash
cd mllm_shap/docs
make clean html
```

## 🛠️ Troubleshooting

- Missing tool in hooks:
  - run `make install`
- Wrong environment path/import source:
  - run `make install` then retry command
- Broken local setup:
  - run `make clean` then `make install`

## ✍️ Style and Structure Expectations

- Keep module docstrings and type hints updated.
- Keep user-facing examples in `examples/` clear and runnable.
- Add local `README.md` for new major modules when useful.
- Prefer small, reviewable pull requests over large mixed changes.

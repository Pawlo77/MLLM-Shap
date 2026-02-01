.PHONY: install update clean activate pre-commit pre-commit-all

install:
	uv python install 3.12
	uv sync --all-groups --no-active
	uv run pre-commit install

update:
	uv lock --upgrade
	uv sync --all-groups --no-active

clean:
	rm -rf .venv/
	rm -f uv.lock
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".cache" -exec rm -r {} \; 2>/dev/null || true

activate:
	@echo "Activate with: source .venv/bin/activate"
	@echo "…or run commands without activating: uv run <cmd>"

pre-commit:
	uv run pre-commit run

pre-commit-all:
	uv run pre-commit run --all-files

tests:
	uv run pytest mllm_shap/tests/

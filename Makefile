.PHONY: install update clean activate pre-commit pre-commit-all benchmarks bench-light coverage tests tests-mllm-shapx

help:
	@echo "Available targets:"
	@echo "  install          Set up the development environment"
	@echo "  update           Update dependencies to their latest versions"
	@echo "  clean            Remove virtual environment and cache files"
	@echo "  activate         Instructions to activate the virtual environment"
	@echo "  pre-commit       Run pre-commit checks on changed files"
	@echo "  pre-commit-all   Run pre-commit checks on all files"
	@echo "  tests            Run the test suite"
	@echo "  bench-light      Run tests with lightweight performance tracking"
	@echo "  benchmarks       Run detailed performance benchmarks"
	@echo "  coverage         Run tests with coverage report"
	@echo "  docs             Build the documentation"

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
	find . -type d \( -name ".cache" -o -name "*__pycache__" -o -name ".*_cache" \) -exec rm -rf {} + 2>/dev/null || true

activate:
	@echo "Activate with: source .venv/bin/activate"
	@echo "…or run commands without activating: uv run <cmd>"

pre-commit:
	uv run pre-commit run

pre-commit-all:
	uv run pre-commit run --all-files

tests:
	uv run pytest mllm_shap/tests/

tests-mllm-shapx:
	uv run pytest experiments/mllm_shapx/tests/

bench-light:
	uv run pytest mllm_shap/tests/ --benchmark-light

benchmarks:
	uv run python -m mllm_shap.benchmarks.bench_api_perf --bench all

coverage:
	uv run pytest --cov=./mllm_shap/src/mllm_shap --cov-report=term-missing mllm_shap/tests/

docs:
	uv run sphinx-apidoc -o mllm_shap/docs/ mllm_shap/src/ && uv run make -C mllm_shap/docs clean html

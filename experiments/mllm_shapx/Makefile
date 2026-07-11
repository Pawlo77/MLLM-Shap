.PHONY: help test test-cli validate plan run run-resume monitor mlflow mlflow-stop

ROOT_DIR := $(abspath ../..)
CLI_MODULE := experiments.mllm_shapx.src.cli
DEFAULT_CONFIG := experiments/mllm_shapx/configs/mc_minimal.json
CONFIG ?= $(DEFAULT_CONFIG)

# MLflow server settings
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5050
MLFLOW_BACKEND_STORE ?= sqlite:///$(ROOT_DIR)/experiments/mllm_shapx/mlflow.db
MLFLOW_ARTIFACT_ROOT ?= $(ROOT_DIR)/experiments/mllm_shapx/mlruns
MLFLOW_TRACKING_URI ?=

help:
	@echo "mllm_shapx Makefile targets:"
	@echo "  test         Run all mllm_shapx tests"
	@echo "  test-cli     Run CLI-focused tests"
	@echo "  validate     Validate a config (CONFIG=...)"
	@echo "  plan         Preview expanded run plan (CONFIG=...)"
	@echo "  run          Execute config (CONFIG=...)"
	@echo "  run-resume   Execute config with --resume (CONFIG=...)"
	@echo "  monitor      Start Streamlit MLflow monitor"
	@echo "  mlflow       Start MLflow tracking server (local DB by default)"
	@echo "  mlflow-stop  Stop MLflow tracking server"
	@echo ""
	@echo "MLflow options:"
	@echo "  MLFLOW_HOST=127.0.0.1          Server host"
	@echo "  MLFLOW_PORT=5050               Server port"
	@echo "  MLFLOW_TRACKING_URI=           Remote URI (overrides local DB)"
	@echo "  MLFLOW_BACKEND_STORE=sqlite:///...  Backend store for local mode"
	@echo "  MLFLOW_ARTIFACT_ROOT=...       Artifact storage path"
	@echo ""
	@echo "Examples:"
	@echo "  make validate CONFIG=experiments/mllm_shapx/configs/mc_minimal.json"
	@echo "  make run-resume CONFIG=experiments/mllm_shapx/configs/package_exp/exact.json"
	@echo "  make mlflow"
	@echo "  make mlflow MLFLOW_TRACKING_URI=http://remote-server:5050"

test:
	cd $(ROOT_DIR) && uv run pytest -q experiments/mllm_shapx/tests/

test-cli:
	cd $(ROOT_DIR) && uv run pytest -q experiments/mllm_shapx/tests/test_cli.py

validate:
	cd $(ROOT_DIR) && uv run python -m $(CLI_MODULE) validate --config $(CONFIG)

plan:
	cd $(ROOT_DIR) && uv run python -m $(CLI_MODULE) plan --config $(CONFIG)

run:
	cd $(ROOT_DIR) && uv run python -m $(CLI_MODULE) run --config $(CONFIG)

run-resume:
	cd $(ROOT_DIR) && uv run python -m $(CLI_MODULE) run --config $(CONFIG) --resume

monitor:
	cd $(ROOT_DIR)/experiments/mllm_shapx/gui && uv run streamlit run app.py

mlflow:
ifneq ($(MLFLOW_TRACKING_URI),)
	@echo "Using remote MLflow server: $(MLFLOW_TRACKING_URI)"
	@echo "Set MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) in your environment to point clients there."
else
	@echo "Starting local MLflow server at http://$(MLFLOW_HOST):$(MLFLOW_PORT)"
	@echo "Backend store: $(MLFLOW_BACKEND_STORE)"
	@echo "Artifact root: $(MLFLOW_ARTIFACT_ROOT)"
	cd $(ROOT_DIR) && uv run mlflow server \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT) \
		--backend-store-uri $(MLFLOW_BACKEND_STORE) \
		--default-artifact-root $(MLFLOW_ARTIFACT_ROOT)
endif

mlflow-stop:
	@echo "Stopping MLflow server on port $(MLFLOW_PORT)..."
	@-lsof -ti :$(MLFLOW_PORT) | xargs kill 2>/dev/null || echo "No MLflow server running on port $(MLFLOW_PORT)"

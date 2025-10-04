install:
	@conda env list | grep bachelor >/dev/null || conda create -n bachelor python=3.12 -y
	@eval "$$(conda shell.bash hook)" && \
		conda activate bachelor && \
		which poetry >/dev/null || (conda run -n bachelor pip install poetry && \
		export PATH="$$HOME/.local/bin:$$PATH") && \
		cd ./audio_shap && poetry install && \
		cd ./../ && poetry install --no-root && \
		pre-commit install

update:
	@eval "$$(conda shell.bash hook)" && \
		conda activate bachelor && \
		poetry self update && \
		poetry update

clean:
	rm -rf .venv/
	rm -rf audio_shap/poetry.lock
	rm -f poetry.lock
	find . -type d -name "__pycache__" -exec rm -r {} +
	conda env remove -n bachelor -y

make activate:
	@echo "Run 'conda activate bachelor' to activate the environment"

pre-commit:
	@eval "$$(conda shell.bash hook)" && \
		conda activate bachelor && \
		pre-commit run --all-files

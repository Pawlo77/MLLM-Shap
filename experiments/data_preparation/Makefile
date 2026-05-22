# Hub upload shortcuts for Pawlo77/mllm-shap builder parquets.
#
# Usage (from repo root or this directory):
#   make -C experiments/data_preparation help
#   make -C experiments/data_preparation upload-single-sentence-voice-bench
#
# Requires HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) with write access, or `hf auth login`.

.PHONY: help test tests list revision dry-run upload-all upload-all-with-readme upload-readme dry-run-readme \
	upload-single-sentence-voice-bench upload-multi-sentence-voice-bench \
	upload-single-sentence-librispeech-asr upload-multi-lingual-infinity-instruct \
	upload-vb-single upload-vb-multi upload-ls upload-ii
REPO_ROOT := $(abspath ../..)
UV ?= uv
UPLOAD := cd $(REPO_ROOT) && $(UV) run python experiments/data_preparation/upload_to_hub.py

CONFIG_SINGLE_SENTENCE_VB := single_sentence__voice_bench
CONFIG_MULTI_SENTENCE_VB := multi_sentence__voice_bench
CONFIG_SINGLE_SENTENCE_LS := single_sentence__librispeech_asr
CONFIG_MULTI_LINGUAL_II := multi_lingual__infinity_instruct

UPLOAD_FLAGS :=
ifdef MESSAGE
UPLOAD_FLAGS += -m "$(MESSAGE)"
endif
ifdef CREATE_PR
UPLOAD_FLAGS += --create-pr
endif

.PHONY: help test tests list revision dry-run upload-all upload-all-with-readme upload-readme dry-run-readme \
	upload-single-sentence-voice-bench upload-multi-sentence-voice-bench \
	upload-single-sentence-librispeech-asr upload-multi-lingual-infinity-instruct \
	upload-vb-single upload-vb-multi upload-ls upload-ii

help:
	@echo "Data preparation — Hub upload (dataset: Pawlo77/mllm-shap)"
	@echo ""
	@echo "  make test              Run unit tests (alias: tests)"
	@echo "  make list              List configs and local parquet status"
	@echo "  make revision          Print current Hub commit hash"
	@echo "  make dry-run           Preview all uploads (no API calls)"
	@echo "  make upload-all        Upload every config that has a parquet file"
	@echo "  make upload-readme     Upload hf/README.md (dataset card) to Hub root"
	@echo "  make upload-all-with-readme  Parquets + README in one commit"
	@echo "  make dry-run-readme    Preview README upload only"
	@echo ""
	@echo "Per-config uploads:"
	@echo "  make upload-single-sentence-voice-bench   ($(CONFIG_SINGLE_SENTENCE_VB))"
	@echo "  make upload-multi-sentence-voice-bench    ($(CONFIG_MULTI_SENTENCE_VB))"
	@echo "  make upload-single-sentence-librispeech-asr ($(CONFIG_SINGLE_SENTENCE_LS))"
	@echo "  make upload-multi-lingual-infinity-instruct ($(CONFIG_MULTI_LINGUAL_II))"
	@echo ""
	@echo "Short aliases: upload-vb-single, upload-vb-multi, upload-ls, upload-ii"
	@echo ""
	@echo "Optional variables:"
	@echo "  MESSAGE=\"...\"   Custom Hub commit message"
	@echo "  CREATE_PR=1       Open a dataset PR instead of pushing to main"

list:
	$(UPLOAD) --list

revision:
	$(UPLOAD) --show-revision

dry-run:
	$(UPLOAD) --all --dry-run

upload-all:
	$(UPLOAD) --all $(UPLOAD_FLAGS)

upload-readme:
	$(UPLOAD) --readme $(UPLOAD_FLAGS)

dry-run-readme:
	$(UPLOAD) --readme --dry-run

upload-all-with-readme:
	$(UPLOAD) --all --readme $(UPLOAD_FLAGS)

upload-single-sentence-voice-bench: upload-vb-single
upload-vb-single:
	$(UPLOAD) --config $(CONFIG_SINGLE_SENTENCE_VB) $(UPLOAD_FLAGS)

upload-multi-sentence-voice-bench: upload-vb-multi
upload-vb-multi:
	$(UPLOAD) --config $(CONFIG_MULTI_SENTENCE_VB) $(UPLOAD_FLAGS)

upload-single-sentence-librispeech-asr: upload-ls
upload-ls:
	$(UPLOAD) --config $(CONFIG_SINGLE_SENTENCE_LS) $(UPLOAD_FLAGS)

upload-multi-lingual-infinity-instruct: upload-ii
upload-ii:
	$(UPLOAD) --config $(CONFIG_MULTI_LINGUAL_II) $(UPLOAD_FLAGS)

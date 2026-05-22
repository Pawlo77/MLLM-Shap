"""Constants and configuration objects for the experiments package.

Defines dataset and TTS configuration Pydantic models, common dataset
config instances used by the builder notebooks and small Hub-related
constants such as target sample sizes and publish targets.
"""

from pathlib import Path

from google.cloud.texttospeech import (
    SsmlVoiceGender,
)
from pydantic import BaseModel

from .io import ensure_dir

EXPERIMENTS_ROOT_DIR: Path = ensure_dir(Path(__file__).resolve().parent.parent)
"""Root directory for the experiments package, used to resolve data paths."""
DATA_DIR: Path = ensure_dir(EXPERIMENTS_ROOT_DIR / "data")
"""Directory for all data used by the experiments, including cached datasets and builder outputs."""


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    dataset_name: str
    """Name of the dataset on the Hugging Face Hub, e.g. ``BAAI/Infinity-Instruct``."""
    data_dir: Path
    """Local directory for dataset-related files, e.g. cached datasets and builder outputs."""
    cache_dir: Path
    """Local directory for Hugging Face dataset caching. Should be a subdirectory of `data_dir` to keep all dataset-related files together."""
    revision: str
    """Specific commit hash to load from the Hub for reproducibility. Should be a 40-character hexadecimal string. No default to encourage explicit pinning."""
    configs: dict[str, str]
    """Mapping of human-friendly config names to actual Hugging Face dataset config names. E.g. ``{"clean": "clean"}`` for LibriSpeech ASR, or ``{"3.46M": "3M", "660k": "0625"}`` for Infinity-Instruct."""
    languages: set[str]
    """Set of language codes present in the dataset, used for filtering and TTS configuration."""


class TTSConfig(BaseModel):
    """Configuration for TTS."""

    language_code: str
    """BCP-47 language code for the TTS voice, e.g. "en-GB" or "fr-FR". Should match the language codes used in the dataset configs."""
    gender: int
    """SSML voice gender, using the `SsmlVoiceGender` enum from the Google Cloud Text-to-Speech API. E.g. `SsmlVoiceGender.MALE` or `SsmlVoiceGender.FEMALE`. Should be compatible with the voices available for the specified language code."""
    voice_name: str | None = None
    """Optional specific voice name to use for synthesis. If not provided, the TTS client will select a default voice matching the language code and"""


VOICE_BENCH__CONFIG = DatasetConfig(
    dataset_name="hlt-lab/voicebench",
    data_dir=ensure_dir(DATA_DIR / "voicebench"),
    cache_dir=ensure_dir(DATA_DIR / "voicebench" / ".cache"),
    revision="b02edcef1330480be3a11bd6f7434ac32f05ad08",
    configs={
        "AdvBench": "advbench",
        "AlpacaEval": "alpacaeval",
        "AlpacaEval-Full": "alpacaeval_full",
        "AlpacaEval-Speaker": "alpacaeval_speaker",
        "BBH": "bbh",
        "CommonEval": "commoneval",
        "IFEval": "ifeval",
        "MMSU": "mmsu",
        "MT-Bench": "mtbench",
        "OpenBookQA": "openbookqa",
        "SD-QA": "sd-qa",
        "WildVoice": "wildvoice",
    },
    languages={"en"},
)

INFINITY_INSTRUCT__CONFIG = DatasetConfig(
    dataset_name="BAAI/Infinity-Instruct",
    data_dir=ensure_dir(DATA_DIR / "infinity_instruct"),
    cache_dir=ensure_dir(DATA_DIR / "infinity_instruct" / ".cache"),
    revision="6e9534fbd3a6c98302755753f0b5fa3d3554a006",
    configs={"3.46M": "3M", "660k": "0625"},
    languages={"en", "fr", "es"},
)

LIBRISPEECH_ASR__CONFIG = DatasetConfig(
    dataset_name="openslr/librispeech_asr",
    data_dir=ensure_dir(DATA_DIR / "librispeech_asr"),
    cache_dir=ensure_dir(DATA_DIR / "librispeech_asr" / ".cache"),
    revision="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    configs={"clean": "clean"},
    languages={"en"},
)


HUB_REPO_ID: str = "Pawlo77/mllm-shap"
HUB_README_PATH: Path = EXPERIMENTS_ROOT_DIR / "hf" / "README.md"
"""Dataset card source for the Hub repo root (YAML frontmatter + markdown)."""
HUB_README_PATH_IN_REPO: str = "README.md"
HUB_DEFAULT_SPLIT: str = "test"
HUB_TARGET_SAMPLES: int = 1000
"""Target row count for published Hub configs (~1k)."""

SINGLE_SENTENCE__VOICE_BENCH: str = "single_sentence__voice_bench"
SINGLE_SENTENCE__LIBRISPEECH_ASR: str = "single_sentence__librispeech_asr"
MULTI_SENTENCE__VOICE_BENCH: str = "multi_sentence__voice_bench"
MULTI_LINGUAL__INFINITY_INSTRUCT: str = "multi_lingual__infinity_instruct"
"""Hub dataset config names (parquet basename matches config name)."""

HUB_SAMPLES_PER_LANGUAGE: int = HUB_TARGET_SAMPLES // 3
"""For multilingual configs, target row count per language (assuming 3 languages in Infinity-Instruct)."""


class HubPublishTarget(BaseModel):
    """Maps a local builder parquet to a Hub dataset config path."""

    hub_config: str
    """Hub dataset config name, e.g. "single_sentence__voice_bench". Should match the keys in the `configs` field of the corresponding `DatasetConfig` instance and the parquet basename used by the builder notebooks."""
    parquet_path: Path
    """Path to the local parquet file to upload for this target. Should be the output path used by the builder notebooks for the corresponding dataset and config."""
    split: str = HUB_DEFAULT_SPLIT
    """Split name to use for the Hub dataset, defaulting to "test" to match the loading split used by the demos. The builder notebooks should save parquets with this split name in the filename, e.g. "single_sentence__voice_bench_test.parquet". The Hub dataset will then have a single split with this name, containing the uploaded parquet data."""
    description: str = ""
    """Human-friendly description of the target config, used for documentation and commit messages."""


def hub_parquet_path_in_repo(hub_config: str, split: str = HUB_DEFAULT_SPLIT) -> str:
    """Path inside the Hub repo matching ``mllm_shapx`` parquet loading."""
    return f"{hub_config}/{split}/0000.parquet"


HUB_PUBLISH_TARGETS: tuple[HubPublishTarget, ...] = (
    HubPublishTarget(
        hub_config=SINGLE_SENTENCE__VOICE_BENCH,
        parquet_path=VOICE_BENCH__CONFIG.data_dir
        / f"{SINGLE_SENTENCE__VOICE_BENCH}.parquet",
        description="VoiceBench NLP-filtered single-sentence split (~1k rows)",
    ),
    HubPublishTarget(
        hub_config=MULTI_SENTENCE__VOICE_BENCH,
        parquet_path=VOICE_BENCH__CONFIG.data_dir
        / f"{MULTI_SENTENCE__VOICE_BENCH}.parquet",
        description="VoiceBench multi-sentence split (~1k rows)",
    ),
    HubPublishTarget(
        hub_config=SINGLE_SENTENCE__LIBRISPEECH_ASR,
        parquet_path=LIBRISPEECH_ASR__CONFIG.data_dir
        / f"{SINGLE_SENTENCE__LIBRISPEECH_ASR}.parquet",
        description="LibriSpeech ASR recorded-audio single-sentence split (~1k rows)",
    ),
    HubPublishTarget(
        hub_config=MULTI_LINGUAL__INFINITY_INSTRUCT,
        parquet_path=INFINITY_INSTRUCT__CONFIG.data_dir
        / f"{MULTI_LINGUAL__INFINITY_INSTRUCT}.parquet",
        description="Infinity-Instruct multilingual conversation split (~1k rows)",
    ),
)
"""Defined Hub publish targets for the builder outputs, mapping local parquet paths to Hub dataset config paths and descriptions used for documentation and commit messages."""


TTS_CONFIGS: dict[str, dict[str, TTSConfig]] = {
    "fr": {
        "male": TTSConfig(language_code="fr-FR", gender=SsmlVoiceGender.MALE),
        "female": TTSConfig(language_code="fr-FR", gender=SsmlVoiceGender.FEMALE),
    },
    "en": {
        "male": TTSConfig(language_code="en-GB", gender=SsmlVoiceGender.MALE),
        "female": TTSConfig(language_code="en-GB", gender=SsmlVoiceGender.FEMALE),
    },
    "es": {
        "male": TTSConfig(language_code="es-ES", gender=SsmlVoiceGender.MALE),
        "female": TTSConfig(language_code="es-ES", gender=SsmlVoiceGender.FEMALE),
    },
}
"""Predefined TTS configurations for supported languages, used by the builder notebooks to select appropriate voices for synthesis based on the language codes present in the datasets. The structure is a nested dictionary mapping language codes to voice configurations."""

"""Constants for experiments."""

from pathlib import Path

from google.cloud.texttospeech import (
    SsmlVoiceGender,
)
from pydantic import BaseModel

from .io import ensure_dir

EXPERIMENTS_ROOT_DIR: Path = ensure_dir(Path(__file__).resolve().parent)
DATA_DIR: Path = ensure_dir(EXPERIMENTS_ROOT_DIR / "data")


# pylint: disable=too-few-public-methods
class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    dataset_name: str
    data_dir: Path
    cache_dir: Path
    revision: str
    configs: dict[str, str]
    languages: set[str]


class TTSConfig(BaseModel):
    """Configuration for TTS."""

    language_code: str
    gender: int
    voice_name: str | None = None


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

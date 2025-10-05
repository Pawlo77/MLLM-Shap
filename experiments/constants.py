"""Constants for experiments."""

from pathlib import Path

from pydantic import BaseModel

from .utils import ensure_dir

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
)

INFINITY_INSTRUCT__CONFIG = DatasetConfig(
    dataset_name="BAAI/Infinity-Instruct",
    data_dir=ensure_dir(DATA_DIR / "infinity_instruct"),
    cache_dir=ensure_dir(DATA_DIR / "infinity_instruct" / ".cache"),
    revision="main",
    configs={"660k": "0625"},
)

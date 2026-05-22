"""Pytest path setup and shared fixtures for data_preparation tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

_DATA_PREP_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _DATA_PREP_ROOT.parents[1]

for path in (_DATA_PREP_ROOT, _REPO_ROOT):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)


@pytest.fixture(autouse=True)
def _disable_progress_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use plain apply in tests (no tqdm pandas hook required)."""
    monkeypatch.setattr(
        pd.DataFrame, "progress_apply", pd.DataFrame.apply, raising=False
    )
    monkeypatch.setattr(pd.Series, "progress_apply", pd.Series.apply, raising=False)


@pytest.fixture
def voicebench_like_df() -> pd.DataFrame:
    """Minimal VoiceBench-style table for preprocessing tests."""
    return pd.DataFrame({
        "prompt": [
            "First prompt here for testing.",
            "First prompt here for testing.",
            "Second prompt with more words inside.",
        ],
        "dataset": ["BBH", "WildVoice", "MMSU"],
        "sentences": [
            ["First prompt here for testing."],
            ["First prompt here for testing."],
            ["Second prompt with more words inside."],
        ],
        "sentences__num": [1, 1, 1],
    })

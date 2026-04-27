"""Tests for Jupyter utility helpers."""

import pandas as pd
import pytest
import torch
from pandas.io.formats.style import Styler

from mllm_shap.utils import jupyter as jupyter_utils


def test_audio_html_uses_display_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """audio_html should delegate to display_audio and return html string."""

    class _AudioStub:
        def _repr_html_(self) -> str:
            return "<audio>ok</audio>"

    monkeypatch.setattr(jupyter_utils, "display_audio", lambda _: _AudioStub())
    html = jupyter_utils.audio_html(b"abc")
    assert html == "<audio>ok</audio>"


def test_display_shap_colors_df_returns_styler() -> None:
    """display_shap_colors_df should return a Styler instance."""
    df = pd.DataFrame({"Shapley Value": [0.1, -0.2], "Token": ["a", "b"]})
    styled = jupyter_utils.display_shap_colors_df(df)
    assert isinstance(styled, Styler)


def test_display_shap_colors_df_audio_applies_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """display_shap_colors_df_audio should convert audio bytes to html strings."""
    monkeypatch.setattr(jupyter_utils, "audio_html", lambda _: "<audio>x</audio>")
    df = pd.DataFrame(
        {"Audio": [b"a1", b"a2"], "Shapley Value": [0.5, -0.1], "Token": ["x", "y"]}
    )
    styled = jupyter_utils.display_shap_colors_df_audio(df, audio_column_name="Audio")
    assert isinstance(styled, Styler)
    assert df["Audio"].tolist() == ["<audio>x</audio>", "<audio>x</audio>"]


def test_plot_distribution_calls_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """plot_distribution should call matplotlib plot primitives."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    calls: list[str] = []

    def _record(name: str):
        def inner(*args, **kwargs):
            del args, kwargs
            calls.append(name)

        return inner

    monkeypatch.setattr(plt, "hist", _record("hist"))
    monkeypatch.setattr(plt, "title", _record("title"))
    monkeypatch.setattr(plt, "xlabel", _record("xlabel"))
    monkeypatch.setattr(plt, "ylabel", _record("ylabel"))
    monkeypatch.setattr(plt, "show", _record("show"))

    jupyter_utils.plot_distribution(torch.tensor([0.1, 0.2, -0.1]), bins=10)

    assert calls == ["hist", "title", "xlabel", "ylabel", "show"]

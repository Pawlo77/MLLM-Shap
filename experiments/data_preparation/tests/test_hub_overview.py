"""Tests for Hub overview helpers."""

from src.hub_overview import show_text_content


def test_show_text_content_prefers_sentences(capsys) -> None:
    show_text_content({"sentences": ["a", "b"], "prompt": "legacy"})
    out = capsys.readouterr().out
    assert "a" in out
    assert "legacy" not in out


def test_show_text_content_falls_back_to_prompt(capsys) -> None:
    show_text_content({"prompt": "only prompt"})
    out = capsys.readouterr().out
    assert "only prompt" in out

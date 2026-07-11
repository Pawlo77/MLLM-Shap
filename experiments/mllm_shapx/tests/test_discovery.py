"""Tests for runtime discovery of explainers and connectors."""

from ..src.discovery import (
    discover_connector_types,
    get_available_explainer_types,
    get_mc_like_explainer_types,
)


def test_discovers_core_explainers() -> None:
    names = get_available_explainer_types()
    assert "exact" in names
    assert "limited_mc" in names
    assert "standard_mc" in names
    assert "limited_cc" in names
    assert "standard_cc" in names
    assert "limited_neyman" in names
    assert "standard_neyman" in names
    assert "hierarchical" in names


def test_mc_like_subset_discovery() -> None:
    mc_like = get_mc_like_explainer_types()
    assert "limited_mc" in mc_like
    assert "standard_mc" in mc_like
    assert "limited_cc" in mc_like
    assert "standard_cc" in mc_like
    assert "limited_neyman" in mc_like
    assert "standard_neyman" in mc_like
    assert "exact" not in mc_like
    assert "hierarchical" not in mc_like


def test_discovers_connector_aliases() -> None:
    names = discover_connector_types()
    assert "liquid_audio" in names
    assert "hf_text" in names
    assert "openai_compat_text" in names

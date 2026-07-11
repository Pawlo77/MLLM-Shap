"""Tests for runner module — variant expansion, stages, helper classes."""

from types import SimpleNamespace

import pandas as pd

from ..src.config import (
    ExperimentSet,
    ExplainerVariant,
    HierarchicalConfig,
)
from ..src.runner import (
    RowSelector,
    _LinearSampleScaler,
    _expand_exact,
    _expand_hierarchical,
    _expand_mc_like,
    _try_set_num_samples,
    expand_variants,
    pick_device,
)


class TestLinearSampleScaler:
    def test_even_result(self) -> None:
        s = _LinearSampleScaler(factor=0.5)
        # 0.5 * 4^2 = 8 (even)
        assert s.scale(n_pre=4) == 8

    def test_odd_rounds_up(self) -> None:
        s = _LinearSampleScaler(factor=0.12)
        # 0.12 * 5^2 = 3 (odd) → 4
        assert s.scale(n_pre=5) == 4

    def test_zero_n_pre(self) -> None:
        s = _LinearSampleScaler(factor=1.0)
        assert s.scale(n_pre=0) == 0

    def test_large_factor(self) -> None:
        s = _LinearSampleScaler(factor=2.0)
        # 2.0 * 3^2 = 18 (even)
        assert s.scale(n_pre=3) == 18


class TestTrySetNumSamples:
    def test_updates_supported(self, monkeypatch) -> None:
        from ..src import runner

        class FakeApprox:
            def __init__(self):
                self.num_samples = 1

        monkeypatch.setattr(runner, "BaseShapApproximation", FakeApprox)
        explainer = SimpleNamespace(shap_explainer=FakeApprox())
        assert _try_set_num_samples(explainer, 99) is True
        assert explainer.shap_explainer.num_samples == 99

    def test_rejects_unsupported(self) -> None:
        explainer = SimpleNamespace(shap_explainer=object())
        assert _try_set_num_samples(explainer, 10) is False


class TestExpandExact:
    def test_single_variant(self) -> None:
        v = ExplainerVariant(explainer_type="exact")
        result = _expand_exact(v)
        assert len(result) == 1
        assert result[0].run_slug == "exact"
        assert result[0].num_samples is None
        assert result[0].fraction is None

    def test_named_variant(self) -> None:
        v = ExplainerVariant(explainer_type="exact", name="my_exact")
        result = _expand_exact(v)
        assert result[0].run_slug == "my_exact"


class TestExpandMcLike:
    def test_num_samples_expansion(self) -> None:
        v = ExplainerVariant(explainer_type="limited_mc", num_samples=[10, 20, 30])
        result = _expand_mc_like(v)
        assert len(result) == 3
        assert result[0].num_samples == 10
        assert result[1].num_samples == 20
        assert result[2].num_samples == 30

    def test_fractions_expansion(self) -> None:
        v = ExplainerVariant(explainer_type="limited_cc", fractions=[0.5, 0.8])
        result = _expand_mc_like(v)
        assert len(result) == 2
        assert result[0].fraction == 0.5
        assert result[1].fraction == 0.8

    def test_linear_expansion(self) -> None:
        v = ExplainerVariant(explainer_type="standard_mc", linear=[0.1, 0.2])
        result = _expand_mc_like(v)
        assert len(result) == 2
        assert result[0].linear == 0.1
        assert result[1].linear == 0.2

    def test_combined_expansion(self) -> None:
        v = ExplainerVariant(
            explainer_type="limited_mc",
            num_samples=[10],
            fractions=[0.5],
            linear=[0.1],
        )
        result = _expand_mc_like(v)
        assert len(result) == 3  # 1 + 1 + 1

    def test_named_variant_slug(self) -> None:
        v = ExplainerVariant(
            explainer_type="limited_mc", num_samples=[10], name="custom"
        )
        result = _expand_mc_like(v)
        assert "custom" in result[0].run_slug


class TestExpandHierarchical:
    def test_default_config(self) -> None:
        v = ExplainerVariant(
            explainer_type="hierarchical",
            hierarchical=HierarchicalConfig(),
        )
        result = _expand_hierarchical(v)
        assert len(result) >= 1
        assert result[0].hier_k == 10
        assert result[0].hier_shap_type == "limited_neyman"

    def test_multiple_ks(self) -> None:
        v = ExplainerVariant(
            explainer_type="hierarchical",
            hierarchical=HierarchicalConfig(ks=[5, 10, 15]),
        )
        result = _expand_hierarchical(v)
        ks = [r.hier_k for r in result]
        assert 5 in ks
        assert 10 in ks
        assert 15 in ks

    def test_with_first_layer(self) -> None:
        v = ExplainerVariant(
            explainer_type="hierarchical",
            hierarchical=HierarchicalConfig(
                ks=[10],
                first_layer_type="precise",
            ),
        )
        result = _expand_hierarchical(v)
        assert result[0].hier_first_layer_type == "precise"


class TestExpandVariants:
    def test_exact(self) -> None:
        cfg = ExperimentSet.model_validate({
            "experiment_set_id": "test",
            "experiments": [{"explainer_type": "exact"}],
        })
        variants = expand_variants(cfg)
        assert len(variants) == 1

    def test_mc_with_samples(self) -> None:
        cfg = ExperimentSet.model_validate({
            "experiment_set_id": "test",
            "experiments": [
                {"explainer_type": "limited_mc", "num_samples": [10, 20]},
            ],
        })
        variants = expand_variants(cfg)
        assert len(variants) == 2

    def test_multiple_experiments(self) -> None:
        cfg = ExperimentSet.model_validate({
            "experiment_set_id": "test",
            "experiments": [
                {"explainer_type": "exact"},
                {"explainer_type": "limited_mc", "num_samples": [10]},
            ],
        })
        variants = expand_variants(cfg)
        assert len(variants) == 2


class TestPickDevice:
    def test_explicit_cpu(self) -> None:
        import torch

        d = pick_device("cpu")
        assert d == torch.device("cpu")

    def test_none_resolves(self) -> None:
        import torch

        d = pick_device(None)
        assert isinstance(d, torch.device)


class TestRowSelector:
    def test_basic_iteration(self) -> None:
        cfg = ExperimentSet.model_validate({
            "experiment_set_id": "test",
            "selection": {"max_samples": 3},
            "experiments": [{"explainer_type": "exact"}],
        })
        df = pd.DataFrame({"prompt": [f"text_{i}" for i in range(10)]})
        selector = RowSelector(cfg, df)
        rows = list(selector.iterate())
        # RowSelector yields all for the runner to handle max_samples
        assert len(rows) >= 3

    def test_with_filters(self) -> None:
        cfg = ExperimentSet.model_validate({
            "experiment_set_id": "test",
            "selection": {"filters": [{"column": "lang", "op": "==", "value": "en"}]},
            "experiments": [{"explainer_type": "exact"}],
        })
        df = pd.DataFrame({"prompt": ["a", "b", "c"], "lang": ["en", "de", "en"]})
        selector = RowSelector(cfg, df)
        rows = list(selector.iterate())
        assert len(rows) == 2

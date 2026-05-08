"""Tests for config module — Pydantic models, parsing, validation, inheritance."""

import json
from pathlib import Path

import pytest

from ..src.config import (
    NORMALIZER_MAP,
    REDUCER_MAP,
    SIMILARITY_MAP,
    ColumnMapping,
    DatasetConfig,
    ExperimentSet,
    ExplainerVariant,
    FilterPredicate,
    GenerationConfig,
    HierarchicalConfig,
    SelectionConfig,
    ShapConfig,
    validate_config,
)
from ..src.constants import (
    DatasetSource,
    ExplainerType,
    TokenFilterType,
)


class TestFilterPredicate:
    def test_valid_ops(self) -> None:
        for op in ("in", "not_in", "==", "!=", "<", "<=", ">", ">=", "between"):
            fp = FilterPredicate(column="x", op=op, value=1)
            assert fp.op == op

    def test_invalid_op_raises(self) -> None:
        with pytest.raises(ValueError, match="filter op"):
            FilterPredicate(column="x", op="contains", value="y")


class TestDatasetConfig:
    def test_defaults(self) -> None:
        cfg = DatasetConfig()
        assert cfg.source == DatasetSource.HF_PARQUET
        assert cfg.repo_id == "Pawlo77/mllm-shap"

    def test_local_parquet_requires_path(self) -> None:
        with pytest.raises(ValueError, match="path is required"):
            DatasetConfig.model_validate({"source": "local_parquet"})

    def test_local_csv_requires_path(self) -> None:
        with pytest.raises(ValueError, match="path is required"):
            DatasetConfig.model_validate({"source": "local_csv"})

    def test_local_parquet_with_path(self) -> None:
        cfg = DatasetConfig(
            source=DatasetSource.LOCAL_PARQUET, path="/tmp/data.parquet"
        )
        assert cfg.path == "/tmp/data.parquet"

    def test_use_parquet_backward_compat_true(self) -> None:
        # Simulating JSON deserialization with use_parquet field
        raw = {"use_parquet": True, "repo_id": "test/repo"}
        cfg = DatasetConfig.model_validate(raw)
        assert cfg.source == DatasetSource.HF_PARQUET

    def test_use_parquet_backward_compat_false(self) -> None:
        raw = {"use_parquet": False, "repo_id": "test/repo"}
        cfg = DatasetConfig.model_validate(raw)
        assert cfg.source == DatasetSource.HF_DATASETS


class TestColumnMapping:
    def test_defaults(self) -> None:
        cm = ColumnMapping()
        assert cm.text is None
        assert cm.audio is None
        assert cm.language == "language"
        assert cm.token_count == "token_count"


class TestSelectionConfig:
    def test_defaults(self) -> None:
        sel = SelectionConfig()
        assert sel.max_samples is None
        assert sel.start_index == 0
        assert sel.shuffle_seed == 0

    def test_negative_max_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SelectionConfig(max_samples=-1)

    def test_zero_max_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SelectionConfig(max_samples=0)

    def test_negative_start_index_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            SelectionConfig(start_index=-1)


class TestGenerationConfig:
    def test_defaults(self) -> None:
        gen = GenerationConfig()
        assert gen.max_new_tokens == 32
        assert gen.text_temperature == 0.2
        assert gen.text_top_k is None
        assert gen.audio_temperature is None
        assert gen.audio_top_k is None

    def test_custom_values(self) -> None:
        gen = GenerationConfig(
            max_new_tokens=100,
            text_temperature=0.7,
            text_top_k=50,
            audio_temperature=0.3,
            audio_top_k=10,
        )
        assert gen.max_new_tokens == 100
        assert gen.text_top_k == 50


class TestShapConfig:
    def test_defaults(self) -> None:
        shap = ShapConfig()
        assert shap.token_filter == TokenFilterType.EXCLUDE_PUNCTUATION
        assert shap.allow_mask_duplicates is False

    def test_invalid_normalizer_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown shap.normalizer"):
            ShapConfig(normalizer="FakeNormalizer")

    def test_invalid_reducer_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown shap.reducer"):
            ShapConfig(reducer="FakeReducer")


class TestExplainerVariant:
    def test_shorthand_mc(self) -> None:
        v = ExplainerVariant(explainer_type="mc", num_samples=[10])
        assert v.explainer_type == ExplainerType.LIMITED_MC

    def test_shorthand_cc(self) -> None:
        v = ExplainerVariant(explainer_type="cc", num_samples=[10])
        assert v.explainer_type == ExplainerType.LIMITED_CC

    def test_shorthand_neyman(self) -> None:
        v = ExplainerVariant(explainer_type="neyman", num_samples=[10])
        assert v.explainer_type == ExplainerType.LIMITED_NEYMAN

    def test_shorthand_complementary(self) -> None:
        v = ExplainerVariant(explainer_type="complementary", num_samples=[10])
        assert v.explainer_type == ExplainerType.LIMITED_CC

    def test_full_name(self) -> None:
        v = ExplainerVariant(explainer_type="limited_mc", num_samples=[10])
        assert v.explainer_type == ExplainerType.LIMITED_MC


class TestHierarchicalConfig:
    def test_defaults(self) -> None:
        h = HierarchicalConfig()
        assert h.ks == [10]
        assert h.shap_type == "limited_neyman"
        assert h.use_importance_sampling is True

    def test_shap_fraction_alias(self) -> None:
        raw = {"shap_fraction": [0.5, 0.7]}
        h = HierarchicalConfig.model_validate(raw)
        assert h.shap_fractions == [0.5, 0.7]


class TestExperimentSet:
    def test_minimal_config(self) -> None:
        raw = {
            "experiment_set_id": "test_exp",
            "experiments": [{"explainer_type": "exact"}],
        }
        cfg = ExperimentSet.model_validate(raw)
        assert cfg.experiment_set_id == "test_exp"
        assert len(cfg.experiments) == 1

    def test_from_json(self, tmp_path: Path) -> None:
        config_data = {
            "experiment_set_id": "json_test",
            "experiments": [{"explainer_type": "exact"}],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        cfg = ExperimentSet.from_json(config_file)
        assert cfg.experiment_set_id == "json_test"

    def test_config_inheritance(self, tmp_path: Path) -> None:
        base_data = {
            "experiment_set_id": "base_exp",
            "generation": {"max_new_tokens": 64},
            "experiments": [{"explainer_type": "exact"}],
        }
        base_file = tmp_path / "base.json"
        base_file.write_text(json.dumps(base_data))

        child_data = {
            "base": "base.json",
            "experiment_set_id": "child_exp",
            "generation": {"text_temperature": 0.9},
        }
        child_file = tmp_path / "child.json"
        child_file.write_text(json.dumps(child_data))

        cfg = ExperimentSet.from_json(child_file)
        assert cfg.experiment_set_id == "child_exp"
        assert cfg.generation.max_new_tokens == 64  # inherited
        assert cfg.generation.text_temperature == 0.9  # overridden

    def test_per_variant_shap_override(self) -> None:
        raw = {
            "experiment_set_id": "override_test",
            "shap": {"normalizer": "AbsSumNormalizer"},
            "experiments": [
                {
                    "explainer_type": "exact",
                    "shap_override": {"normalizer": "IdentityNormalizer"},
                }
            ],
        }
        cfg = ExperimentSet.model_validate(raw)
        effective = cfg.get_effective_shap(cfg.experiments[0])
        assert effective.normalizer == "IdentityNormalizer"

    def test_effective_shap_no_override(self) -> None:
        raw = {
            "experiment_set_id": "no_override",
            "shap": {"normalizer": "AbsSumNormalizer"},
            "experiments": [{"explainer_type": "exact"}],
        }
        cfg = ExperimentSet.model_validate(raw)
        effective = cfg.get_effective_shap(cfg.experiments[0])
        assert effective.normalizer == "AbsSumNormalizer"


class TestValidateConfig:
    def test_empty_experiments_error(self) -> None:
        cfg = ExperimentSet.model_validate(
            {"experiment_set_id": "x", "experiments": []}
        )
        errs = validate_config(cfg)
        assert any("at least one variant" in e for e in errs)

    def test_mc_without_samples_error(self) -> None:
        cfg = ExperimentSet.model_validate(
            {
                "experiment_set_id": "x",
                "experiments": [{"explainer_type": "limited_mc"}],
            }
        )
        errs = validate_config(cfg)
        assert any("num_samples" in e for e in errs)

    def test_valid_config_no_errors(self) -> None:
        cfg = ExperimentSet.model_validate(
            {
                "experiment_set_id": "x",
                "experiments": [{"explainer_type": "exact"}],
            }
        )
        errs = validate_config(cfg)
        assert errs == []

    def test_hf_text_with_audio_output_error(self) -> None:
        cfg = ExperimentSet.model_validate(
            {
                "experiment_set_id": "x",
                "connector": "hf_text",
                "modality": {"output_modality": "audio"},
                "experiments": [{"explainer_type": "exact"}],
            }
        )
        errs = validate_config(cfg)
        assert any("audio output" in e for e in errs)


class TestRegistries:
    def test_normalizer_map_has_expected_keys(self) -> None:
        assert "AbsSumNormalizer" in NORMALIZER_MAP
        assert "IdentityNormalizer" in NORMALIZER_MAP
        assert "PowerShiftNormalizer" in NORMALIZER_MAP

    def test_reducer_map_has_expected_keys(self) -> None:
        assert "MeanReducer" in REDUCER_MAP
        assert "MaxReducer" in REDUCER_MAP
        assert "SumReducer" in REDUCER_MAP

    def test_similarity_map_has_expected_keys(self) -> None:
        assert "CosineSimilarity" in SIMILARITY_MAP
        assert "TfIdfCosineSimilarity" in SIMILARITY_MAP

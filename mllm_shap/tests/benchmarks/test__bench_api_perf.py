"""Unit tests for benchmark helpers in bench_api_perf."""

from argparse import Namespace
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mllm_shap.benchmarks import bench_api_perf as bp


def test_time_many_collects_min_median_max() -> None:
    """_time_many should aggregate durations across repeats."""
    stamps = [0.0, 0.3, 1.0, 1.1, 2.0, 2.2]

    with patch(
        "mllm_shap.benchmarks.bench_api_perf.time.perf_counter", side_effect=stamps
    ):
        out = bp._time_many(lambda: None, repeats=3)

    assert out[0] == pytest.approx(0.1)
    assert out[1] == pytest.approx(0.2)
    assert out[2] == pytest.approx(0.3)


def test_fmt_uses_ms_and_seconds_units() -> None:
    """_fmt should format values according to requested time unit."""
    vals = (0.1, 0.2, 0.3)
    ms = bp._fmt("label", vals, unit="ms")
    sec = bp._fmt("label", vals, unit="s")

    assert "min= 100.000 ms" in ms
    assert "p50= 200.000 ms" in ms
    assert "max= 300.000 ms" in ms
    assert "min=   0.100 s" in sec


def test_bench_mask_hash_prints_formatted_result() -> None:
    """bench_mask_hash should call formatter and print section output."""
    with (
        patch(
            "mllm_shap.benchmarks.bench_api_perf._time_many",
            return_value=(0.1, 0.2, 0.3),
        ) as tm,
        patch("builtins.print") as mock_print,
    ):
        bp.bench_mask_hash(iters=3, mask_len=8, repeats=2)

    tm.assert_called_once()
    assert mock_print.call_count == 2


def test_bench_mask_hash_run_closure_executes_hash_loop() -> None:
    """Timed closure in bench_mask_hash should execute hash calls `iters` times."""

    def _fake_time_many(fn, repeats):
        del repeats
        fn()
        return (0.1, 0.2, 0.3)

    with (
        patch(
            "mllm_shap.benchmarks.bench_api_perf._time_many",
            side_effect=_fake_time_many,
        ),
        patch("mllm_shap.benchmarks.bench_api_perf.MasksManager.get_hash") as get_hash,
        patch("builtins.print"),
    ):
        bp.bench_mask_hash(iters=4, mask_len=8, repeats=1)

    assert get_hash.call_count == 4


def test_fake_chat_from_chat_ignores_inputs() -> None:
    """_FakeChat.from_chat should return a new fake chat instance."""
    out = bp._FakeChat.from_chat(torch.tensor([True]), bp._FakeChat(cache="x"))
    assert isinstance(out, bp._FakeChat)
    assert out.cache is None


def test_fake_model_generate_keep_history_toggles_response_chat() -> None:
    """_FakeModel.generate should return ModelResponse with optional chat history."""
    model = bp._FakeModel()

    with patch(
        "mllm_shap.benchmarks.bench_api_perf.ModelResponse",
        side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
    ):
        with_history = model.generate(chat=bp._FakeChat(), keep_history=True)
        without_history = model.generate(chat=bp._FakeChat(), keep_history=False)

    assert with_history.chat is not None
    assert without_history.chat is None
    assert with_history.generated_text_tokens.tolist() == [1, 2, 3]


def test_fake_cache_manager_contains_and_extract() -> None:
    """_FakeCacheManager should expose contains/extract over internal cache."""
    cache = bp._FakeCacheManager()
    response = bp._FakeModel().generate(chat=bp._FakeChat())
    cache._cache[7] = response

    assert cache.contains(7) is True
    assert cache.contains(9) is False
    assert cache.extract(7) is response


def test_mask_gen_yields_expected_number_and_shapes() -> None:
    """_mask_gen should yield boolean masks with the configured length."""
    items = list(bp._mask_gen(n_masks=4, mask_len=6))

    assert len(items) == 4
    for idx, (mask, mask_hash) in enumerate(items):
        assert mask.shape == (6,)
        assert mask.dtype == torch.bool
        assert isinstance(mask_hash, int)
        assert mask_hash == hash((idx, 6))


def test_bench_response_generation_invokes_generate_responses() -> None:
    """bench_response_generation should delegate orchestration to generate_responses."""

    def _fake_time_many(fn, repeats):
        del repeats
        fn()
        return (0.1, 0.2, 0.3)

    with (
        patch(
            "mllm_shap.benchmarks.bench_api_perf._time_many",
            side_effect=_fake_time_many,
        ) as tm,
        patch(
            "mllm_shap.benchmarks.bench_api_perf.generate_responses",
            return_value=([], []),
        ) as gen,
        patch("builtins.print") as mock_print,
    ):
        bp.bench_response_generation(
            n_masks=3,
            mask_len=5,
            repeats=2,
            n_jobs=1,
            verbose=True,
        )

    tm.assert_called_once()
    gen.assert_called_once()
    assert mock_print.call_count == 2


def test_dummy_approx_methods_cover_fallback_paths() -> None:
    """_DummyApprox should implement expected minimal approximation behaviors."""
    approx = bp._DummyApprox(num_samples=7, fraction=0.6)

    assert approx._get_num_splits(10) == 7
    assert approx._calculate_shap_values(
        masks=torch.ones(2, 4, dtype=torch.bool),
        similarities=torch.zeros(2),
        device=torch.device("cpu"),
    ).shape == (4,)
    assert (
        approx._get_next_split(
            n=5,
            device=torch.device("cpu"),
            generated_masks_num=0,
        )
        is None
    )


def test_bench_linear_num_samples_update_prints_results() -> None:
    """bench_linear_num_samples_update should print benchmark summary."""
    with (
        patch(
            "mllm_shap.benchmarks.bench_api_perf._time_many",
            return_value=(0.1, 0.2, 0.3),
        ) as tm,
        patch("builtins.print") as mock_print,
    ):
        bp.bench_linear_num_samples_update(iters=5, repeats=2)

    tm.assert_called_once()
    assert mock_print.call_count == 2


def test_bench_linear_num_samples_update_run_closure_executes_updates() -> None:
    """Timed closure in linear update benchmark should mutate num_samples repeatedly."""

    class _ApproxStub:
        def __init__(self, num_samples, fraction):
            self.num_samples = num_samples
            self.fraction = fraction

    def _fake_time_many(fn, repeats):
        del repeats
        fn()
        return (0.1, 0.2, 0.3)

    with (
        patch("mllm_shap.benchmarks.bench_api_perf._DummyApprox", _ApproxStub),
        patch(
            "mllm_shap.benchmarks.bench_api_perf._time_many",
            side_effect=_fake_time_many,
        ),
        patch("builtins.print"),
    ):
        bp.bench_linear_num_samples_update(iters=3, repeats=1)


def test_main_runs_all_benchmarks_when_bench_all() -> None:
    """main should run all benchmark groups when --bench=all."""
    args = Namespace(
        iters=2,
        repeats=1,
        mask_len=4,
        n_masks=3,
        n_stages=2,
        jobs=2,
        bench="all",
        output_json=None,
        output_csv=None,
        max_p50_ms=None,
        max_overhead_pct=None,
    )

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch(
            "mllm_shap.benchmarks.bench_api_perf.bench_mask_hash", return_value=[]
        ) as bh,
        patch(
            "mllm_shap.benchmarks.bench_api_perf.bench_response_generation",
            return_value=[],
        ) as br,
        patch(
            "mllm_shap.benchmarks.bench_api_perf.bench_sampling_adapter",
            return_value=[],
        ) as bs,
        patch(
            "mllm_shap.benchmarks.bench_api_perf.bench_pipeline_observability_overhead",
            return_value=[],
        ) as bo,
        patch(
            "mllm_shap.benchmarks.bench_api_perf.bench_linear_num_samples_update",
            return_value=[],
        ) as bl,
        patch("builtins.print") as mock_print,
    ):
        bp.main()

    bh.assert_called_once_with(iters=2, mask_len=4, repeats=1)
    br.assert_called_once_with(
        n_masks=3,
        mask_len=4,
        repeats=1,
        n_jobs=2,
        verbose=False,
    )
    bs.assert_called_once_with(
        n_masks=3,
        mask_len=4,
        repeats=1,
        n_jobs=2,
        verbose=False,
    )
    bo.assert_called_once_with(iters=2, repeats=1, n_stages=2)
    bl.assert_called_once_with(iters=2, repeats=1)
    assert mock_print.call_count == 4


def test_main_runs_selected_benchmark_only() -> None:
    """main should execute only selected benchmark group for specific --bench value."""
    args = Namespace(
        iters=2,
        repeats=1,
        mask_len=4,
        n_masks=3,
        n_stages=2,
        jobs=2,
        bench="responses",
        output_json=None,
        output_csv=None,
        max_p50_ms=None,
        max_overhead_pct=None,
    )

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch("mllm_shap.benchmarks.bench_api_perf.bench_mask_hash") as bh,
        patch(
            "mllm_shap.benchmarks.bench_api_perf.bench_response_generation",
            return_value=[],
        ) as br,
        patch(
            "mllm_shap.benchmarks.bench_api_perf.bench_linear_num_samples_update"
        ) as bl,
        patch("builtins.print") as mock_print,
    ):
        bp.main()

    bh.assert_not_called()
    br.assert_called_once()
    bl.assert_not_called()
    assert mock_print.call_count == 1


def test_main_mask_hash_branch_skips_responses_and_linear() -> None:
    """main with --bench=mask-hash should not call responses or linear-update branches."""
    args = Namespace(
        iters=2,
        repeats=1,
        mask_len=4,
        n_masks=3,
        n_stages=2,
        jobs=2,
        bench="mask-hash",
        output_json=None,
        output_csv=None,
        max_p50_ms=None,
        max_overhead_pct=None,
    )

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch(
            "mllm_shap.benchmarks.bench_api_perf.bench_mask_hash", return_value=[]
        ) as bh,
        patch("mllm_shap.benchmarks.bench_api_perf.bench_response_generation") as br,
        patch(
            "mllm_shap.benchmarks.bench_api_perf.bench_linear_num_samples_update"
        ) as bl,
        patch("builtins.print") as mock_print,
    ):
        bp.main()

    bh.assert_called_once()
    br.assert_not_called()
    bl.assert_not_called()
    assert mock_print.call_count == 1


def test_bench_response_generation_run_closure_executes_generator() -> None:
    """The timed closure in bench_response_generation should initialize lists and consume gen."""

    def _fake_time_many(fn, repeats):
        del repeats
        fn()
        return (0.1, 0.2, 0.3)

    with (
        patch(
            "mllm_shap.benchmarks.bench_api_perf._time_many",
            side_effect=_fake_time_many,
        ),
        patch(
            "mllm_shap.benchmarks.bench_api_perf.generate_responses",
            return_value=([], []),
        ) as gen,
        patch("builtins.print"),
    ):
        bp.bench_response_generation(
            n_masks=2,
            mask_len=3,
            repeats=1,
            n_jobs=1,
            verbose=False,
        )

    kwargs = gen.call_args.kwargs
    assert isinstance(kwargs["masks"], list)
    assert isinstance(kwargs["responses"], list)
    assert kwargs["progress_bar"] is False


def test_module_dunder_main_executes_main_function() -> None:
    """Executing the module as __main__ should trigger main via dunder guard."""
    args = Namespace(
        iters=1,
        repeats=1,
        mask_len=2,
        n_masks=1,
        n_stages=1,
        jobs=1,
        bench="mask-hash",
        output_json=None,
        output_csv=None,
        max_p50_ms=None,
        max_overhead_pct=None,
    )

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch("builtins.print"),
    ):
        module_name = "mllm_shap.benchmarks.bench_api_perf"
        previous = sys.modules.pop(module_name, None)
        try:
            runpy.run_module(module_name, run_name="__main__")
        finally:
            if previous is not None:
                sys.modules[module_name] = previous


def test_write_results_json_and_csv(tmp_path) -> None:
    """Structured result writers should persist benchmark records."""
    results = [
        bp.BenchResult(
            bench="mask-hash",
            label="sample",
            min_s=0.001,
            p50_s=0.002,
            max_s=0.003,
        )
    ]
    json_path = tmp_path / "bench" / "results.json"
    csv_path = tmp_path / "bench" / "results.csv"

    bp._write_results_json(str(json_path), results)
    bp._write_results_csv(str(csv_path), results)

    assert json_path.exists()
    assert csv_path.exists()
    assert "mask-hash" in json_path.read_text(encoding="utf-8")
    assert "p50_ms" in csv_path.read_text(encoding="utf-8")


def test_enforce_thresholds_reports_violations() -> None:
    """Threshold helper should count both latency and overhead violations."""
    results = [
        bp.BenchResult(
            bench="responses",
            label="slow",
            min_s=0.010,
            p50_s=0.020,
            max_s=0.030,
        ),
        bp.BenchResult(
            bench="pipeline-observability",
            label="with-sink",
            min_s=0.010,
            p50_s=0.020,
            max_s=0.030,
            overhead_p50_pct=15.0,
        ),
    ]

    with patch("builtins.print") as mock_print:
        violations = bp._enforce_thresholds(
            results=results,
            max_p50_ms=5.0,
            max_overhead_pct=10.0,
        )

    assert violations == 3
    assert mock_print.call_count == 3

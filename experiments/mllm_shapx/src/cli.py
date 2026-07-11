"""CLI entrypoint: validate, run, plan experiments with sharding support."""

import argparse
import glob
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

# Suppress torch warnings that fire at import time (before _setup_logging).
warnings.filterwarnings(  # noqa: E402
    "ignore",
    message=r"The pynvml package is deprecated\. Please install nvidia-ml-py instead\..*",
    category=FutureWarning,
    module=r"torch\.cuda",
)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
    logging.ERROR
)


_MLLM_SHAP_SRC = os.environ.get("MLLM_SHAP_SRC")
if _MLLM_SHAP_SRC and _MLLM_SHAP_SRC not in sys.path:
    sys.path.insert(0, _MLLM_SHAP_SRC)


@dataclass(frozen=True)
class OutputOptions:
    """Output mode flags for CLI rendering."""

    as_json: bool = False
    """Emit machine-readable JSON lines instead of human-readable text."""
    quiet: bool = False
    """Reduce console output to errors and summaries."""
    verbose: bool = False
    """Enable additional progress diagnostics."""


def _output_options_from_args(args: argparse.Namespace) -> OutputOptions:
    """Build output options from parsed CLI args."""
    return OutputOptions(
        as_json=bool(getattr(args, "json", False)),
        quiet=bool(getattr(args, "quiet", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _emit_json(event: str, payload: dict[str, object]) -> None:
    """Emit one JSON-line event for machine-readable output."""
    out = {"event": event, **payload}
    print(json.dumps(out, ensure_ascii=False))


def _print_line(
    output: OutputOptions, message: str,  quiet_sensitive: bool = True
) -> None:
    """Print one plain-text line respecting output mode flags."""
    if output.as_json:
        return
    if output.quiet and quiet_sensitive:
        return
    print(message)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Return a grammatically correct noun form for a count."""
    if count == 1:
        return singular
    return plural or f"{singular}s"


def _print_status(output: OutputOptions, level: str, message: str) -> None:
    """Print a compact status line with a normalized level prefix."""
    if output.as_json:
        _emit_json(
            "status",
            {
                "level": level.lower(),
                "message": message,
            },
        )
        return
    if output.quiet and level not in {"ERROR"}:
        return
    if level == "DEBUG" and not output.verbose:
        return
    print(f"[{level}] {message}")


def _print_header(output: OutputOptions, title: str) -> None:
    """Print a simple ASCII section header."""
    if output.as_json:
        return
    if output.quiet:
        return
    line = "=" * max(20, min(80, len(title) + 8))
    print(f"\n{line}\n{title}\n{line}")


def _print_config_errors(
    output: OutputOptions,
    errors: Iterable[str],
    config_path: str | None = None,
) -> None:
    """Print config validation errors in a compact, scan-friendly list."""
    error_list = list(errors)
    target = f" ({config_path})" if config_path else ""
    if output.as_json:
        _emit_json(
            "config_error",
            {
                "ok": False,
                "config": config_path,
                "errors": error_list,
            },
        )
        return
    _print_status(output, "ERROR", f"Config validation failed{target}.")
    for error in error_list:
        print(f"  - {error}")


def _setup_logging() -> None:
    """Configure root logger from environment."""
    level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"An output with one or more elements was resized since it had shape "
            r"\[\], which does not match the required output shape .*"
        ),
        category=UserWarning,
        module=r"torch\.functional",
    )

    # Suppress benign 404 logs from httpx when HuggingFace Hub probes for non-existent files
    # (e.g., additional_chat_templates) during model metadata fetching.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate config and optionally check dataset availability."""
    from .config import ExperimentSet, validate_config
    from .data import load_df

    output = _output_options_from_args(args)

    cfg = ExperimentSet.from_json(args.config)
    errs = validate_config(cfg)
    if errs:
        _print_config_errors(output, errs, args.config)
        sys.exit(2)

    dataset_ok = None
    if args.check_dataset:
        try:
            _ = load_df(cfg.dataset)
            dataset_ok = True
            _print_status(output, "OK", "Dataset is reachable and readable.")
        except Exception as ex:
            dataset_ok = False
            _print_status(output, "ERROR", f"Dataset fetch failed: {ex}")
            sys.exit(2)

    if output.as_json:
        _emit_json(
            "validate",
            {
                "ok": True,
                "config": args.config,
                "check_dataset": bool(args.check_dataset),
                "dataset_ok": dataset_ok,
            },
        )

    _print_status(output, "OK", "Configuration looks valid.")


def cmd_plan(args: argparse.Namespace) -> None:
    """Preview what variants will be generated and how many rows selected."""
    from .config import ExperimentSet, validate_config
    from .data import apply_filters, choose_prompt_text_column, load_df
    from .runner import expand_variants

    output = _output_options_from_args(args)

    cfg = ExperimentSet.from_json(args.config)
    errs = validate_config(cfg)
    if errs:
        _print_config_errors(output, errs, args.config)
        sys.exit(2)

    variants = expand_variants(cfg)

    dataset_rows: int | None = None
    text_col: str | None = None
    dataset_error: str | None = None

    if output.as_json:
        if not args.skip_data:
            try:
                df = load_df(cfg.dataset)
                if cfg.selection.filters:
                    df = apply_filters(df, cfg.selection.filters)
                dataset_rows = len(df)
                text_col = choose_prompt_text_column(
                    df, cfg.dataset.column_mapping.text
                )
            except Exception as ex:
                dataset_error = str(ex)

        variant_payload: list[dict[str, object]] = []
        for v in variants:
            entry: dict[str, object] = {
                "run_slug": v.run_slug,
                "explainer_type": v.variant.explainer_type,
            }
            if v.num_samples is not None:
                entry["num_samples"] = v.num_samples
            if v.fraction is not None:
                entry["fraction"] = v.fraction
            if v.linear is not None:
                entry["linear"] = v.linear
            if v.hier_k is not None:
                entry["hier_k"] = v.hier_k
            variant_payload.append(entry)

        payload: dict[str, object] = {
            "ok": True,
            "config": args.config,
            "experiment_set_id": cfg.experiment_set_id,
            "connector": cfg.connector,
            "device": cfg.device or "auto",
            "variant_count": len(variants),
            "variants": variant_payload,
            "dataset": {
                "source": cfg.dataset.source,
                "path": cfg.dataset.path,
                "repo_id": cfg.dataset.repo_id,
                "subset": cfg.dataset.subset,
                "split": cfg.dataset.split,
            },
            "selection": {
                "max_samples": cfg.selection.max_samples,
            },
            "shap": {
                "mode": cfg.shap.mode,
                "normalizer": cfg.shap.normalizer,
                "reducer": cfg.shap.reducer,
                "similarity": cfg.shap.similarity,
                "token_filter": cfg.shap.token_filter,
                "allow_mask_duplicates": cfg.shap.allow_mask_duplicates,
            },
            "generation": {
                "max_new_tokens": cfg.generation.max_new_tokens,
                "text_temperature": cfg.generation.text_temperature,
                "text_top_k": cfg.generation.text_top_k,
                "audio_temperature": cfg.generation.audio_temperature,
                "audio_top_k": cfg.generation.audio_top_k,
            },
        }
        if dataset_rows is not None:
            payload["dataset_rows"] = dataset_rows
        if text_col is not None:
            payload["text_column"] = text_col
        if dataset_error is not None:
            payload["dataset_error"] = dataset_error
        _emit_json("plan", payload)
        return

    _print_header(output, "Execution Plan")
    plan_header_fields = [
        ("set_id", cfg.experiment_set_id),
        ("connector", cfg.connector),
        ("device", cfg.device or "auto"),
        ("variants", f"{len(variants)} {_plural(len(variants), 'variant')}"),
    ]
    for key, value in plan_header_fields:
        _print_line(output, f"{key:<12}: {value}")
    _print_line(output, "\nPlanned variants:")
    for i, v in enumerate(variants, 1):
        parts = [f"type={v.variant.explainer_type}"]
        if v.num_samples is not None:
            parts.append(f"ns={v.num_samples}")
        if v.fraction is not None:
            parts.append(f"frac={v.fraction}")
        if v.linear is not None:
            parts.append(f"lin={v.linear}")
        if v.hier_k is not None:
            parts.append(f"k={v.hier_k}")
        _print_line(output, f"  {i:3d}. {v.run_slug}  [{', '.join(parts)}]")

    # Dataset info
    _print_line(output, "\nDataset:")
    _print_line(output, f"  {'source':<12}: {cfg.dataset.source}")
    if cfg.dataset.source.startswith("hf"):
        _print_line(
            output,
            f"  {'repo/split':<12}: {cfg.dataset.repo_id} / {cfg.dataset.subset} / {cfg.dataset.split}",
        )
    else:
        _print_line(output, f"  {'path':<12}: {cfg.dataset.path}")

    if not args.skip_data:
        try:
            df = load_df(cfg.dataset)
            if cfg.selection.filters:
                df = apply_filters(df, cfg.selection.filters)
            _print_line(output, f"  {'rows':<12}: {len(df)}")
            text_col = choose_prompt_text_column(df, cfg.dataset.column_mapping.text)
            _print_line(output, f"  {'text_column':<12}: {text_col}")
            if cfg.selection.max_samples:
                _print_line(
                    output, f"  {'max_samples':<12}: {cfg.selection.max_samples}"
                )
        except Exception as ex:
            _print_status(
                output, "WARN", f"Could not inspect dataset during plan: {ex}"
            )

    # Shap config summary
    _print_line(output, "\nSHAP config:")
    shap_fields = [
        ("mode", cfg.shap.mode),
        ("normalizer", cfg.shap.normalizer),
        ("reducer", cfg.shap.reducer),
        ("similarity", cfg.shap.similarity),
        ("token_filter", cfg.shap.token_filter),
        ("duplicates", cfg.shap.allow_mask_duplicates),
    ]
    for key, value in shap_fields:
        _print_line(output, f"  {key:<12}: {value}")

    # Generation config
    _print_line(output, "\nGeneration:")
    generation_fields: list[tuple[str, object]] = [
        ("max_new_tokens", cfg.generation.max_new_tokens),
        ("text_temp", cfg.generation.text_temperature),
    ]
    if cfg.generation.text_top_k is not None:
        generation_fields.append(("text_top_k", cfg.generation.text_top_k))
    if cfg.generation.audio_temperature is not None:
        generation_fields.append(("audio_temp", cfg.generation.audio_temperature))
    if cfg.generation.audio_top_k is not None:
        generation_fields.append(("audio_top_k", cfg.generation.audio_top_k))
    for key, value in generation_fields:
        _print_line(output, f"  {key:<14}: {value}")
    _print_line(output, "")


def cmd_run(args: argparse.Namespace) -> None:
    """Execute all configured variants, with optional sharding."""
    from .config import ExperimentSet, validate_config
    from .data import load_df
    from .runner import expand_variants, run_single_sentence_variant

    output = _output_options_from_args(args)

    configs = _resolve_configs(args, output)
    total_variants = 0
    variants_skipped = 0
    variants_executed = 0

    _print_status(
        output,
        "INFO",
        f"Starting run for {len(configs)} {_plural(len(configs), 'config')}.",
    )

    for cfg_idx, config_path in enumerate(configs, 1):
        _print_line(
            output,
            f"\n[Config {cfg_idx}/{len(configs)}] {os.path.basename(config_path)}",
        )
        cfg = ExperimentSet.from_json(config_path)
        if args.max_samples is not None:
            cfg.selection.max_samples = int(args.max_samples)

        # n_generator_jobs: CLI > env > config
        n_jobs = (
            args.n_generator_jobs
            if args.n_generator_jobs is not None
            else os.environ.get("MLLM_SHAP_N_GENERATOR_JOBS")
        )
        if n_jobs is not None:
            cfg.runtime.n_generator_jobs = int(n_jobs)

        # LM Studio host: CLI > env > config
        lms_host = (
            args.lm_studio_host
            if args.lm_studio_host is not None
            else os.environ.get("MLLM_SHAP_LM_STUDIO_HOST")
        )
        if lms_host is not None:
            cfg.lm_studio.api_host = lms_host

        # Apply sharding to selection
        if args.shard_index is not None and args.num_shards is not None:
            shard_idx = int(args.shard_index)
            num_shards = int(args.num_shards)
            total = cfg.selection.max_samples or 1000
            shard_size = (total + num_shards - 1) // num_shards
            cfg.selection.start_index = shard_idx * shard_size
            cfg.selection.max_samples = min(
                shard_size, total - cfg.selection.start_index
            )
            if cfg.selection.max_samples <= 0:
                _print_status(
                    output,
                    "INFO",
                    (
                        f"Shard {shard_idx} has no work "
                        f"(total={total}, num_shards={num_shards})."
                    ),
                )
                continue

        errs = validate_config(cfg)
        if errs:
            _print_config_errors(output, errs, config_path)
            sys.exit(2)

        variants = expand_variants(cfg)
        total_variants += len(variants)
        if not variants:
            _print_status(output, "INFO", f"No variants to run for {config_path}.")
            continue

        # LM Studio lifecycle: download + load before variants, unload after
        lm_studio_mgr = None
        if cfg.lm_studio.enabled:
            from .lm_studio import LmStudioManager, build_lm_studio_config

            max_prompt_tokens = cfg.selection.max_prompt_tokens
            lms_cfg = build_lm_studio_config(cfg, max_prompt_tokens=max_prompt_tokens)
            lm_studio_mgr = LmStudioManager(lms_cfg)
            _print_status(output, "INFO", "LM Studio: ensuring model is downloaded...")
            lm_studio_mgr.ensure_downloaded()
            _print_status(output, "INFO", "LM Studio: loading model...")
            lm_studio_mgr.load()
            _print_status(output, "INFO", "LM Studio: model ready.")

            # Inject OpenAI-compat connector kwargs so the connector uses LM Studio
            if cfg.connector in ("openai_compat_text", "lm_studio_text"):
                cfg.connector_kwargs.setdefault("base_url", lm_studio_mgr.api_base_url)
                cfg.connector_kwargs.setdefault(
                    "chat_model", lm_studio_mgr.model_identifier
                )

        try:
            df: pd.DataFrame | None = None
            for var_idx, run in enumerate(variants, 1):
                # Check how many samples are already done for this variant
                already_done = (
                    _get_completed_count_from_mlflow(cfg, run.run_slug)
                    if args.resume
                    else 0
                )
                desired = cfg.selection.max_samples
                status = f"{already_done}/{desired}" if desired else f"{already_done}/?"

                if desired and already_done >= desired:
                    variants_skipped += 1
                    _print_line(
                        output,
                        f"  [Variant {var_idx}/{len(variants)}] {run.run_slug} "
                        f"[{status} done] skipped",
                    )
                    continue

                _print_line(
                    output,
                    f"  [Variant {var_idx}/{len(variants)}] {run.run_slug} [{status} done]",
                )
                if df is None:
                    _print_status(
                        output, "INFO", "Loading dataset once for this config."
                    )
                    df = load_df(cfg.dataset)
                run_single_sentence_variant(cfg, run, df, resume=args.resume)
                variants_executed += 1
        finally:
            if lm_studio_mgr is not None:
                _print_status(output, "INFO", "LM Studio: unloading model...")
                lm_studio_mgr.close()
                _print_status(output, "INFO", "LM Studio: model unloaded.")

    if output.as_json:
        _emit_json(
            "run_summary",
            {
                "ok": True,
                "configs": len(configs),
                "variants_total": total_variants,
                "variants_run": variants_executed,
                "variants_skip": variants_skipped,
            },
        )
        return

    _print_header(output, "Run Summary")
    _print_line(output, f"configs       : {len(configs)}", quiet_sensitive=False)
    _print_line(output, f"variants_total: {total_variants}", quiet_sensitive=False)
    _print_line(output, f"variants_run  : {variants_executed}", quiet_sensitive=False)
    _print_line(output, f"variants_skip : {variants_skipped}", quiet_sensitive=False)
    _print_status(output, "OK", "All variants complete.")


def _get_completed_count_from_mlflow(cfg: Any, run_slug: str) -> int:
    """Query MLflow to find how many samples are completed for a run."""
    from .mlflow_tracker import _find_existing_run, _get_completed_indices_from_run

    run_name = f"{cfg.experiment_set_id}__{run_slug}"
    run_id = _find_existing_run(cfg.mlflow.experiment_name, run_name)
    if run_id is None:
        return 0
    return len(_get_completed_indices_from_run(run_id))


def _resolve_configs(args: argparse.Namespace, output: OutputOptions) -> list[str]:
    """Resolve config paths from --config (glob) or --config-list."""
    if args.config_list:
        with open(args.config_list, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    config_arg = args.config
    # Support glob patterns
    if "*" in config_arg or "?" in config_arg:
        paths = sorted(glob.glob(config_arg, recursive=True))
        if not paths:
            _print_status(output, "ERROR", f"No configs matched pattern: {config_arg}")
            sys.exit(2)
        return paths
    return [config_arg]


def build_argparser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    p = argparse.ArgumentParser(
        description="mllm_shap experiment runner (validate, plan, run)"
    )

    output_group = p.add_mutually_exclusive_group()
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output to errors and summaries.",
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable additional progress diagnostics.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON lines instead of human-readable text.",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- validate
    v = sub.add_parser(
        "validate", help="Validate a config (and optionally test dataset)."
    )
    v.add_argument("--config", required=True, help="Path to JSON config.")
    v.add_argument(
        "--check-dataset", action="store_true", help="Try to download and read dataset."
    )
    v.set_defaults(func=cmd_validate)

    # ---- plan
    pl = sub.add_parser(
        "plan", help="Preview variants, dataset info, and parameters (dry run)."
    )
    pl.add_argument("--config", required=True, help="Path to JSON config.")
    pl.add_argument(
        "--skip-data", action="store_true", help="Skip dataset loading in plan output."
    )
    pl.set_defaults(func=cmd_plan)

    # ---- run
    r = sub.add_parser("run", help="Run experiments defined in config.")
    r.add_argument(
        "--config",
        default=None,
        help="Path to JSON config (supports globs like 'configs/*.json').",
    )
    r.add_argument(
        "--config-list",
        default=None,
        help="Path to file listing config paths (one per line).",
    )
    r.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint/output if present.",
    )
    r.add_argument(
        "--max-samples", type=int, default=None, help="Override selection.max_samples."
    )
    r.add_argument(
        "--n-generator-jobs",
        type=int,
        default=None,
        help=(
            "Number of parallel model calls. "
            "Env: MLLM_SHAP_N_GENERATOR_JOBS (default: from config or 1)."
        ),
    )
    r.add_argument(
        "--lm-studio-host",
        type=str,
        default=None,
        help=(
            "LM Studio API host (e.g. 127.0.0.1:1234). "
            "Env: MLLM_SHAP_LM_STUDIO_HOST (default: from config)."
        ),
    )
    r.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Shard index (0-based) for distributed execution.",
    )
    r.add_argument(
        "--num-shards", type=int, default=None, help="Total number of shards."
    )
    r.set_defaults(func=cmd_run)

    return p


def main() -> None:
    """CLI main."""
    _setup_logging()
    ap = build_argparser()
    args = ap.parse_args()

    # Validate run args
    if args.cmd == "run":
        if not args.config and not args.config_list:
            ap.error("run requires --config or --config-list")
        if args.shard_index is not None and args.num_shards is None:
            ap.error("--shard-index requires --num-shards")
        if args.num_shards is not None and args.shard_index is None:
            ap.error("--num-shards requires --shard-index")

    args.func(args)


if __name__ == "__main__":
    main()

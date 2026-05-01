"""CLI entrypoint: validate, run, plan experiments with sharding support."""

import argparse
import glob
import logging
import os
import sys
import warnings

import pandas as pd


_MLLM_SHAP_SRC = os.environ.get("MLLM_SHAP_SRC")
if _MLLM_SHAP_SRC and _MLLM_SHAP_SRC not in sys.path:
    sys.path.insert(0, _MLLM_SHAP_SRC)


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
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated\. Please install nvidia-ml-py instead\..*",
        category=FutureWarning,
        module=r"torch\.cuda\.__init__",
    )


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate config and optionally check dataset availability."""
    from .config import ExperimentSet, validate_config
    from .data import load_df

    cfg = ExperimentSet.from_json(args.config)
    errs = validate_config(cfg)
    if errs:
        print("❌ Config problems:")
        for e in errs:
            print("  -", e)
        sys.exit(2)

    if args.check_dataset:
        try:
            _ = load_df(cfg.dataset)
            print("✅ Dataset reachable & readable.")
        except Exception as ex:
            print("❌ Dataset fetch failed:", ex)
            sys.exit(2)

    print("✅ Config looks good.")


def cmd_plan(args: argparse.Namespace) -> None:
    """Preview what variants will be generated and how many rows selected."""
    from .config import ExperimentSet, validate_config
    from .data import apply_filters, choose_prompt_text_column, load_df
    from .runner import expand_variants

    cfg = ExperimentSet.from_json(args.config)
    errs = validate_config(cfg)
    if errs:
        print("❌ Config problems:")
        for e in errs:
            print("  -", e)
        sys.exit(2)

    variants = expand_variants(cfg)
    print(f"\n{'=' * 60}")
    print(f"Experiment set: {cfg.experiment_set_id}")
    print(f"Connector:      {cfg.connector}")
    print(f"Device:         {cfg.device or 'auto'}")
    print(f"{'=' * 60}")
    print(f"\nVariants to run: {len(variants)}")
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
        print(f"  {i:3d}. {v.run_slug} ({', '.join(parts)})")

    # Dataset info
    print("\nDataset:")
    print(f"  Source: {cfg.dataset.source}")
    if cfg.dataset.source.startswith("hf"):
        print(
            f"  Repo:   {cfg.dataset.repo_id} / {cfg.dataset.subset} / {cfg.dataset.split}"
        )
    else:
        print(f"  Path:   {cfg.dataset.path}")

    if not args.skip_data:
        try:
            df = load_df(cfg.dataset)
            if cfg.selection.filters:
                df = apply_filters(df, cfg.selection.filters)
            print(f"  Rows:   {len(df)} total")
            text_col = choose_prompt_text_column(df, cfg.dataset.column_mapping.text)
            print(f"  Text column: '{text_col}'")
            if cfg.selection.max_samples:
                print(f"  Max samples: {cfg.selection.max_samples}")
        except Exception as ex:
            print(f"  ⚠️ Could not load dataset: {ex}")

    # Shap config summary
    print("\nSHAP config:")
    print(f"  Mode:       {cfg.shap.mode}")
    print(f"  Normalizer: {cfg.shap.normalizer}")
    print(f"  Reducer:    {cfg.shap.reducer}")
    print(f"  Similarity: {cfg.shap.similarity}")
    print(f"  Filter:     {cfg.shap.token_filter}")
    print(f"  Duplicates: {cfg.shap.allow_mask_duplicates}")

    # Generation config
    print("\nGeneration:")
    print(f"  max_new_tokens:    {cfg.generation.max_new_tokens}")
    print(f"  text_temperature:  {cfg.generation.text_temperature}")
    if cfg.generation.text_top_k is not None:
        print(f"  text_top_k:        {cfg.generation.text_top_k}")
    if cfg.generation.audio_temperature is not None:
        print(f"  audio_temperature: {cfg.generation.audio_temperature}")
    if cfg.generation.audio_top_k is not None:
        print(f"  audio_top_k:       {cfg.generation.audio_top_k}")
    print()


def cmd_run(args: argparse.Namespace) -> None:
    """Execute all configured variants, with optional sharding."""
    from .config import ExperimentSet, validate_config
    from .data import load_df
    from .runner import expand_variants, run_single_sentence_variant

    configs = _resolve_configs(args)

    for config_path in configs:
        cfg = ExperimentSet.from_json(config_path)
        if args.max_samples is not None:
            cfg.selection.max_samples = int(args.max_samples)

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
                print(
                    f"Shard {shard_idx} has no work (total={total}, shards={num_shards})."
                )
                continue

        errs = validate_config(cfg)
        if errs:
            print(f"❌ Config problems ({config_path}):")
            for e in errs:
                print("  -", e)
            sys.exit(2)

        df: pd.DataFrame = load_df(cfg.dataset)
        variants = expand_variants(cfg)
        if not variants:
            print(f"Nothing to run for {config_path}.")
            continue

        for run in variants:
            print(
                f"\n=== Running variant: {run.run_slug} (type={run.variant.explainer_type}, "
                f"num_samples={run.num_samples}, fraction={run.fraction}) ==="
            )
            run_single_sentence_variant(cfg, run, df, resume=args.resume)

    print("\nAll variants complete.")


def _resolve_configs(args: argparse.Namespace) -> list[str]:
    """Resolve config paths from --config (glob) or --config-list."""
    if args.config_list:
        with open(args.config_list, "r") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    config_arg = args.config
    # Support glob patterns
    if "*" in config_arg or "?" in config_arg:
        paths = sorted(glob.glob(config_arg, recursive=True))
        if not paths:
            print(f"❌ No configs matched pattern: {config_arg}")
            sys.exit(2)
        return paths
    return [config_arg]


def build_argparser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    p = argparse.ArgumentParser(
        description="mllm_shap experiment runner (validate, plan, run)"
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

"""CLI entrypoint for validation and running experiments."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

# pylint: disable=wrong-import-position
_MLLM_SHAP_SRC = os.environ.get("MLLM_SHAP_SRC")
if _MLLM_SHAP_SRC and _MLLM_SHAP_SRC not in sys.path:
    sys.path.insert(0, _MLLM_SHAP_SRC)

from .config import ExperimentSet, validate_config  # noqa: E402
from .data import load_df, load_single_sentence_df  # noqa: E402
from .runner import expand_variants, run_single_sentence_variant  # noqa: E402


def _setup_logging() -> None:
    """Configure root logger from environment."""
    level = os.environ.get("LOG_LEVEL", "DEBUG").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate config and optionally check dataset availability."""
    cfg = ExperimentSet.from_json(args.config)
    errs = validate_config(cfg)
    if errs:
        print("❌ Config problems:")
        for e in errs:
            print("  -", e)
        sys.exit(2)

    if args.check_dataset:
        try:
            _ = load_single_sentence_df(
                cfg.dataset.repo_id,
                cfg.dataset.subset,
                cfg.dataset.split,
                cfg.dataset.revision,
            )
            print("✅ Dataset shard reachable & readable.")
        except Exception as ex:  # pylint: disable=broad-except
            print("❌ Dataset fetch failed:", ex)
            sys.exit(2)

    print("✅ Config looks good.")


def cmd_run(args: argparse.Namespace) -> None:
    """Execute all configured variants."""
    cfg = ExperimentSet.from_json(args.config)
    errs = validate_config(cfg)
    if errs:
        print("❌ Config problems:")
        for e in errs:
            print("  -", e)
        sys.exit(2)

    df: pd.DataFrame = load_df(
        cfg.dataset.repo_id,
        cfg.dataset.subset,
        cfg.dataset.split,
        cfg.dataset.revision,
        use_parquet=cfg.dataset.use_parquet,
        trust_remote_code=cfg.dataset.trust_remote_code,
    )
    variants = expand_variants(cfg)
    if not variants:
        print("Nothing to run.")
        return

    for run in variants:
        print(
            f"\n=== Running variant: {run.run_slug} (type={run.variant.explainer_type}, "
            f"num_samples={run.num_samples}, fraction={run.fraction}) ==="
        )
        run_single_sentence_variant(cfg, run, df, resume=args.resume)

    print("\nAll variants complete.")


def build_argparser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    p = argparse.ArgumentParser(
        description="mllm_shap single_sentence experiment runner (exact & MC)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser(
        "validate", help="Validate a config (and optionally test dataset availability)."
    )
    v.add_argument("--config", required=True, help="Path to JSON config.")
    v.add_argument(
        "--check-dataset",
        action="store_true",
        help="Try to download and read a parquet shard.",
    )
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="Run experiments defined in config.")
    r.add_argument("--config", required=True, help="Path to JSON config.")
    r.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint/output if present.",
    )
    r.set_defaults(func=cmd_run)

    return p


def main() -> None:
    """CLI main."""
    _setup_logging()
    ap = build_argparser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

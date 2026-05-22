#!/usr/bin/env python3
"""CLI: upload local data-preparation parquets to Pawlo77/mllm-shap on the Hub."""

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return the root directory of the repository."""
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> None:
    """Add the repository root to sys.path for imports."""
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Upload builder parquets to the Hugging Face dataset repo. "
            "Each file is placed at {config}/{split}/0000.parquet, matching "
            "experiments/mllm_shapx parquet loading."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured Hub configs and local parquet paths.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        metavar="NAME",
        help="Hub config to upload (repeatable). Default: all configured targets.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload every configured target that has a local parquet file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without calling the Hub API.",
    )
    parser.add_argument(
        "--message",
        "-m",
        default=None,
        help="Custom commit message for the Hub upload.",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Open a pull request on the dataset repo instead of pushing to main.",
    )
    parser.add_argument(
        "--show-revision",
        action="store_true",
        help="Print the latest Hub revision and exit (no upload).",
    )
    parser.add_argument(
        "--readme",
        action="store_true",
        help=(
            "Upload experiments/data_preparation/hf/README.md to the Hub repo root. "
            "Combine with --all or --config to include README in the same commit."
        ),
    )
    return parser.parse_args()


def main() -> int:
    _ensure_import_path()

    from src.constants import (
        HUB_PUBLISH_TARGETS,
        HUB_REPO_ID,
    )
    from src.constants import HUB_README_PATH
    from src.hub_upload import (
        build_upload_plan,
        format_upload_plan,
        get_latest_hub_revision,
        list_publish_targets,
        resolve_hub_readme_path,
        upload_readme_to_hub,
        upload_targets_to_hub,
    )

    args = _parse_args()

    if args.show_revision:
        revision = get_latest_hub_revision(HUB_REPO_ID)
        print(f"{HUB_REPO_ID} @ {revision}")
        return 0

    if args.list:
        readme_status = "ok" if HUB_README_PATH.is_file() else "missing"
        print(f"[{readme_status}] README <- {HUB_README_PATH}")
        for target in HUB_PUBLISH_TARGETS:
            status = "ok" if target.parquet_path.is_file() else "missing"
            print(
                f"[{status}] {target.hub_config} <- {target.parquet_path}"
                + (f"  ({target.description})" if target.description else "")
            )
        return 0

    if args.readme and not args.all and not args.configs:
        if args.dry_run:
            readme = resolve_hub_readme_path()
            print(format_upload_plan([], readme_path=readme))
            return 0
        result = upload_readme_to_hub(
            commit_message=args.message,
            create_pr=args.create_pr,
        )
        print(f"Uploaded README to {result.repo_id}")
        print(f"Revision: {result.revision}")
        if result.commit_url:
            print(f"Commit:   {result.commit_url}")
        return 0

    if not args.all and not args.configs:
        print("Specify --all, --config NAME, --readme, or --list.", file=sys.stderr)
        return 2

    hub_configs = set(args.configs) if args.configs else None
    targets = list_publish_targets(hub_configs=hub_configs)
    if not targets:
        print("No matching publish targets.", file=sys.stderr)
        return 2

    unknown = (hub_configs or set()) - {t.hub_config for t in list_publish_targets()}
    if unknown:
        known = ", ".join(t.hub_config for t in list_publish_targets())
        print(
            f"Unknown config(s): {', '.join(sorted(unknown))}. Known: {known}",
            file=sys.stderr,
        )
        return 2

    readme_path = HUB_README_PATH if args.readme else None
    if args.dry_run:
        plan = build_upload_plan(targets, require_files=False)
        if not plan and readme_path is None:
            print("Nothing to upload (no parquet files on disk).", file=sys.stderr)
            return 1
        if readme_path is not None:
            resolve_hub_readme_path(readme_path)
        print(format_upload_plan(plan, readme_path=readme_path))
        return 0

    result = upload_targets_to_hub(
        targets,
        commit_message=args.message,
        create_pr=args.create_pr,
        readme_path=readme_path,
    )
    uploaded_count = len(result.uploaded) + (1 if args.readme else 0)
    print(f"Uploaded {uploaded_count} file(s) to {result.repo_id}")
    print(f"Revision: {result.revision}")
    if result.commit_url:
        print(f"Commit:   {result.commit_url}")
    print()
    print("Pin this revision in:")
    print("  - experiments/data_preparation/overview.ipynb (REVISION)")
    print("  - experiments/mllm_shapx configs (dataset.revision)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

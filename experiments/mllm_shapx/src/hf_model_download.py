"""Securely download a pinned Hugging Face model snapshot to a local directory.

Usage:
    python -m mllm_shap.utils.hf_download \
        --repo-id intfloat/e5-base-v2 \
        --revision <40_hex_commit_sha> \
        --dest ./local_models/e5-base-v2

You can also restrict files via --allow-patterns to save space, e.g.:
    --allow-patterns tokenizer.* config.json *.safetensors
"""

import argparse
import re
from pathlib import Path
from huggingface_hub import snapshot_download


def _is_commit_sha(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", s))


def download_model(
    repo_id: str,
    revision: str,
    allow_patterns: (list[str] | str) | None = None,
    local_files_only: bool = False,
) -> str:
    """Download a pinned HF snapshot and return the local path."""
    if not _is_commit_sha(revision):
        raise ValueError("`revision` must be a 40-character hex commit SHA.")

    out = snapshot_download(  # nosec: B615 - pinned to immutable commit; validated above
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
        tqdm_class=None,
    )
    return str(Path(out).resolve())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download a pinned Hugging Face model snapshot locally."
    )
    p.add_argument(
        "--repo-id", required=True, help="HF repo id, e.g. intfloat/e5-base-v2"
    )
    p.add_argument("--revision", required=True, help="40-hex commit SHA (immutable)")
    p.add_argument(
        "--dest", default=None, help="Destination directory (created if missing)"
    )
    p.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="Optional file-globs to include (e.g. tokenizer.* config.json *.safetensors)",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local cache; do not hit the network.",
    )
    p.add_argument(
        "--use-symlinks",
        action="store_true",
        help="Keep cache layout via symlinks instead of copying files.",
    )
    return p


def main() -> None:
    """Main CLI entry point."""
    args = _build_parser().parse_args()
    local_dir = download_model(
        repo_id=args.repo_id,
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        local_files_only=args.local_files_only,
    )
    print(local_dir)


if __name__ == "__main__":
    main()

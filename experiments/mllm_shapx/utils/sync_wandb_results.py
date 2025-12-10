"""Sync W&B Artifacts to Local Experiments Folder"""

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:
    import wandb
except ImportError:
    print("[!] wandb not installed. Please install wandb to use this script.")
    wandb = None

if TYPE_CHECKING:
    from wandb.apis.public import Run

# Regex to match the artifact naming convention defined in runner.py
# Format: {experiment_set_id}__{run_slug}-{type}
ARTIFACT_REGEX = re.compile(
    r"^(?P<set_id>.+?)__(?P<slug>.+?)-(?P<kind>samples|summary)$"
)


def reconstruct_spec(run: "Run", target_path: Path) -> None:
    """
    Reconstructs spec.json from the W&B Run configuration.
    This is crucial because runner.py expects this file to exist.
    """
    if target_path.exists():
        return

    print("  -> Reconstructing spec.json from W&B Run config...")
    try:
        # W&B flattens config, but your spec structure in runner.py is nested.
        # Fortunately, wandb.config usually preserves the structure if passed as a dict.
        spec_content = run.config

        # Remove internal wandb keys
        spec_content = {k: v for k, v in spec_content.items() if not k.startswith("_")}

        # Ensure directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(spec_content, f, indent=2)
    except (OSError, json.JSONDecodeError, AttributeError) as e:
        print(f"  [!] Failed to reconstruct spec.json: {e}")


# pylint: disable=too-many-locals
def sync_artifacts(
    entity: str, project: str, output_root: str, experiment_filter: Optional[str] = None
) -> None:
    """Sync W&B Artifacts to Local Experiments Folder."""

    api: Any = wandb.Api()
    print(f"--- Syncing from {entity}/{project} to '{output_root}' ---")

    # We look for both 'samples' and 'summary' artifact types
    artifact_types = ["samples", "summary"]

    for art_type_name in artifact_types:
        try:
            # Fetch artifact type collection
            art_type = api.artifact_type(
                type_name=art_type_name, project=f"{entity}/{project}"
            )
        except wandb.Error:
            print(f"No artifacts of type '{art_type_name}' found.")
            continue

        for collection in art_type.collections():
            # Match the artifact name against your naming convention
            match = ARTIFACT_REGEX.match(collection.name)
            if not match:
                continue

            set_id = match.group("set_id")
            run_slug = match.group("slug")
            kind = match.group("kind")

            # Optional filtering
            if experiment_filter and set_id != experiment_filter:
                continue

            # Get the latest version of this artifact
            try:
                artifact = api.artifact(f"{entity}/{project}/{collection.name}:latest")
            except wandb.Error:
                print(f"  [!] Could not fetch latest artifact for {collection.name}")
                continue

            # Construct Local Path: output_root / experiment_set_id / run_slug / {samples|summary}
            # This mirrors the structure in storage.py::make_run_dir
            run_dir = Path(output_root) / set_id / run_slug
            target_dir = run_dir / kind

            print(f"Syncing: {set_id} / {run_slug} [{kind}]")

            # 1. Download the content
            # 'root' ensures files are placed directly into samples/ or summary/
            artifact.download(root=str(target_dir))

            # 2. Attempt to reconstruct spec.json if this is a 'summary' or 'samples' download
            # We only need to do this once per run, but doing it safely on every artifact is fine.
            spec_path = run_dir / "spec.json"
            if not spec_path.exists():
                # Find the run that logged this artifact to get the config
                logged_by = artifact.logged_by()
                if logged_by:
                    reconstruct_spec(logged_by, spec_path)
                else:
                    # Fallback: Create empty spec if run no longer exists
                    print("  [!] Run not found. Creating empty spec.json.")
                    with open(spec_path, "w", encoding="utf-8") as f:
                        json.dump({}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync W&B artifacts to local experiments_output folder."
    )
    parser.add_argument("--entity", required=True, help="W&B Entity (username or team)")
    parser.add_argument("--project", required=True, help="W&B Project name")
    parser.add_argument(
        "--output",
        default="experiments_output",
        help="Local root folder for experiments",
    )
    parser.add_argument(
        "--filter", default=None, help="Optional: Filter by experiment_set_id"
    )

    args = parser.parse_args()

    sync_artifacts(args.entity, args.project, args.output, args.filter)

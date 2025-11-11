"""Simple helper used for backfilling artifacts in past runs."""

import importlib

wb = importlib.import_module("wandb")
init = getattr(wb, "init")
artifact = getattr(wb, "Artifact")
with init(entity="<entity>",
          project="<project>",
          id="<id>",
          resume="allow") as run:
    art = artifact("ss_package_test_exact_mc_2025_11_10__exact_baseline-samples",
                   type="samples",
                   metadata={"source": "backfill"})
    art.add_dir("experiments_output/ss_package_test_exact_mc_2025_11_10/exact_baseline/samples")
    run.log_artifact(art, aliases=["latest", "backfill"])

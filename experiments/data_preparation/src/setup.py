"""Shared notebook setup: device detection and model initialization.

Small helpers used by the notebooks to configure the runtime environment
(``get_device``, seed RNGs, enable ``tqdm.pandas``) and create commonly
used helper objects like a token-counting model wrapper.
"""

import os
from pathlib import Path

import torch
from mllm_shap.connectors import LiquidAudio
from mllm_shap.connectors.enums import ModelHistoryTrackingMode


def get_device(include_mps: bool = True) -> torch.device:
    """Detect the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if include_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_token_model(device: torch.device) -> LiquidAudio:
    """Create a LiquidAudio model configured for token counting."""
    return LiquidAudio(
        device=device, history_tracking_mode=ModelHistoryTrackingMode.TEXT
    )


def configure_notebook_environment(
    chdir_levels: int = 2,
    set_plot_style: bool = False,
) -> Path:
    """Chdir to repo root and set common notebook environment variables."""
    from tqdm.auto import tqdm
    import numpy as np

    np.random.seed(0)

    repo_root = Path.cwd()
    for _ in range(chdir_levels):
        repo_root = repo_root.parent
    os.chdir(repo_root)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tqdm.pandas()

    if set_plot_style:
        import seaborn as sns

        sns.set_style("whitegrid")

    return repo_root

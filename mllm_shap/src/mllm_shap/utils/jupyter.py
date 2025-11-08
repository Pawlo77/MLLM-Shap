"""Utility functions for Jupyter Notebook visualization."""

from typing import Any

import pandas as pd
import torch
from pandas.io.formats.style import Styler
from torch import Tensor

from .audio import display_audio


def _audio_html(content: bytes) -> str:
    """
    Generate HTML representation for audio content.

    Args:
        content: Audio content in bytes.
    Returns:
        str: HTML representation of the audio.
    """
    a = display_audio(content)
    return str(a._repr_html_())  # type: ignore[no-untyped-call] # pylint: disable=protected-access


def display_shap_colors_df(
    df: pd.DataFrame,
    shap_column_name: str = "Shapley Value",
    cmap: str = "coolwarm",
    low: float = 0.0,
    high: float = 1.0,
    **kwargs: Any,
) -> Styler:
    """Set background gradient colors for SHAP values in a DataFrame.

    Args:
        df: DataFrame containing SHAP values.
        shap_column_name: Name of the column with SHAP values.
        cmap: Colormap to use for the gradient.
        low: Minimum value for the gradient.
        high: Maximum value for the gradient.
        **kwargs: Additional arguments for pandas Styler.background_gradient.
    Returns:
        pd.Styler: Styled DataFrame with background gradient.
    """
    return df.style.background_gradient(subset=[shap_column_name], cmap=cmap, low=low, high=high, **kwargs)


def display_shap_colors_df_audio(df: pd.DataFrame, audio_column_name: str = "Audio", **kwargs: Any) -> Styler:
    """
    Set background gradient colors for SHAP values in a DataFrame with audio.
    Render audio in the specified audio column for jupyter notebooks.

    Args:
        df: DataFrame containing SHAP values and audio.
        audio_column_name: Name of the column with audio.
        **kwargs: Additional arguments for display_shap_colors_df.
    Returns:
        pd.Styler: Styled DataFrame with background gradient.
    """
    df[audio_column_name] = df[audio_column_name].apply(_audio_html)

    return display_shap_colors_df(df, **kwargs)


def plot_distribution(values: Tensor, bins: int = 50, **kwargs: Any) -> None:
    """
    Plot histogram of SHAP values distribution.

    Args:
        values: Tensor of SHAP values.
        bins: Number of bins for the histogram.
        **kwargs: Additional arguments for matplotlib.pyplot.hist.
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    # Move to CPU & flatten for plotting
    values_np = values.detach().cpu().to(torch.float32).numpy().flatten()

    # Plot histogram
    plt.hist(values_np, bins=bins, **kwargs)
    plt.title("SHAP Values Distribution")
    plt.xlabel("SHAP Value")
    plt.ylabel("Frequency")
    plt.show()

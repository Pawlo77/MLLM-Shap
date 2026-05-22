"""SGPA alignment demo helpers for the audio notebook.

Small helpers to load the Spectrogram Guided Aligner, tokenize and render
token-level alignment tables that include short audio clips for each token.
"""

import numpy as np
import pandas as pd
import torch
from IPython.display import HTML, display
from transformers import PreTrainedTokenizerBase

from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import display_audio
from mllm_shap.utils.jupyter import audio_html


def create_sgpa_demo(
    device: torch.device | None = None,
) -> tuple[SpectrogramGuidedAligner, PreTrainedTokenizerBase]:
    """Load aligner and tokenizer used in the audio QA notebook."""
    from transformers import AutoTokenizer

    if device is None:
        device = torch.device("cpu")
    aligner = SpectrogramGuidedAligner(device=device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")  # nosec B615
    return aligner, tokenizer


def display_token_alignment(
    text: str,
    audio: np.ndarray,
    aligner: SpectrogramGuidedAligner,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Align *audio* to *text* and render the per-token HTML table."""
    tokens = [tokenizer.decode([token_id]) for token_id in tokenizer.encode(text)]

    print(f"Aligning to transcript: '{text}'")
    display(display_audio(audio))

    print(f"Aligning to {len(tokens)} tokens: {tokens}")
    segments = aligner(audio_content=audio, transcript=tokens)

    table = pd.DataFrame({
        "Token": [s.token for s in segments],
        "Start Time (s)": [s.start_time for s in segments],
        "End Time (s)": [s.end_time for s in segments],
        "Duration (s)": [s.end_time - s.start_time for s in segments],
        "Confidence": [s.confidence for s in segments],
        "Audio": [s.audio if s.audio else None for s in segments],
    })
    table["Audio"] = table["Audio"].apply(audio_html).str.replace("\n", "")
    display(HTML(table.to_html(escape=False)))


def display_random_alignment_example(
    df: pd.DataFrame,
    aligner: SpectrogramGuidedAligner,
    tokenizer: PreTrainedTokenizerBase,
    audio_col: str = "audio__male",
) -> None:
    """Pick a random row and show SGPA alignment for its first sentence clip."""
    row = df.sample(1).iloc[0]
    display_token_alignment(
        row.sentences[0],
        row[audio_col][0],
        aligner=aligner,
        tokenizer=tokenizer,
    )

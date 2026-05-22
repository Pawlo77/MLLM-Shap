"""Google Cloud TTS exploration helpers for ``google_tts_demo.ipynb``.

Helpers to list available voices, filter them for demos and synthesise
small sample DataFrames used interactively in the demo notebook.
"""

from typing import Any

import pandas as pd

from .constants import TTS_CONFIGS, TTSConfig
from .nlp import TTS


async def list_voices_dataframe(tts: TTS) -> pd.DataFrame:
    """Fetch all voices from the API as a DataFrame."""
    return pd.DataFrame([
        {
            "name": voice.name,
            "language_codes": voice.language_codes,
            "gender": voice.ssml_gender.name,
        }
        for voice in (await tts.client.list_voices()).voices
    ])


def filter_voices_for_demo(
    voices_df: pd.DataFrame,
    language_codes: tuple[str, ...],
) -> pd.DataFrame:
    """Keep selected languages and model names shared across all of them."""
    out = voices_df.explode("language_codes")
    out = out[out["language_codes"].isin(language_codes)]
    out = out.sort_values(by=["language_codes", "gender"]).reset_index(drop=True)
    out["model_name"] = (
        out["name"].str.split("-HD-").apply(lambda x: x[1] if len(x) > 1 else None)
    )
    out.dropna(subset=["model_name"], inplace=True)

    groups = out.groupby(["language_codes"])
    common = set.intersection(*[
        set(group["model_name"].unique()) for _, group in groups
    ])
    return out[out["model_name"].isin(common)].reset_index(drop=True)


async def build_configured_samples_dataframe(
    tts: TTS,
    samples: dict[str, str],
    tts_configs: dict[str, dict[str, TTSConfig]] | None = None,
) -> pd.DataFrame:
    """Synthesize one clip per language × gender from *tts_configs*."""
    if tts_configs is None:
        tts_configs = TTS_CONFIGS

    rows: list[dict[str, Any]] = []
    for language, gender_config in tts_configs.items():
        for _gender, config in gender_config.items():
            audio = await tts.synthesize_text(
                text=samples[language],
                language_code=config.language_code,
                voice_name=config.voice_name,
                gender=config.gender,
            )
            rows.append({
                "language_code": config.language_code,
                "gender": config.gender,
                "voice_name": config.voice_name,
                "audio": audio,
            })
    return pd.DataFrame(rows)


def display_tts_sample_row(row: pd.Series, tts: TTS | None = None) -> None:
    """Print metadata and play one synthesized sample row."""
    print(
        f"Language: {row['language_code']}, Gender: {row['gender']}, "
        f"Voice: {row['voice_name']}"
    )
    player = tts or TTS()
    player.display_audio(row["audio"])


def display_all_tts_samples(sample_df: pd.DataFrame, tts: TTS | None = None) -> None:
    """Play every row in a samples DataFrame."""
    for _, row in sample_df.iterrows():
        display_tts_sample_row(row, tts=tts)

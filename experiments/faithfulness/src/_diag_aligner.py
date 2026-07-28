"""Ad-hoc diagnostic: does the SGPA aligner segment a real clip locally?"""

import glob
import traceback

import pandas as pd
import torch

from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner


def _find_parquet() -> str:
    hits = glob.glob(
        "/home/mvishiu11/.cache/huggingface/**/single_sentence_500/test/*.parquet",
        recursive=True,
    )
    if not hits:
        hits = glob.glob(
            "/home/mvishiu11/.cache/huggingface/**/*.parquet", recursive=True
        )
    return hits[0]


def main() -> None:
    pq = _find_parquet()
    print("parquet:", pq)
    df = pd.read_parquet(pq)
    print("columns:", list(df.columns))
    print("rows:", len(df))

    # find transcript-ish + audio-ish columns
    text_col = next(
        (c for c in ("sentences", "text", "transcript", "sentence") if c in df.columns),
        None,
    )
    print("text_col:", text_col)
    row = df.iloc[0]
    transcript = row[text_col]
    if isinstance(transcript, (list, tuple)):
        transcript = " ".join(map(str, transcript))
    print("transcript:", repr(transcript)[:200])

    audio_val = row.get("audio__original")
    print("audio__original type:", type(audio_val).__name__)
    import numpy as _np

    if isinstance(audio_val, _np.ndarray) and audio_val.dtype == object:
        audio_bytes = bytes(audio_val[0])
    elif isinstance(audio_val, dict):
        audio_bytes = audio_val.get("bytes")
    elif isinstance(audio_val, (bytes, bytearray)):
        audio_bytes = bytes(audio_val)
    else:
        audio_bytes = None
    fmt_hint = "wav" if audio_bytes and audio_bytes[:4] == b"RIFF" else "mp3"
    print(
        "audio_bytes len:",
        len(audio_bytes) if audio_bytes else None,
        "| fmt:",
        fmt_hint,
    )

    print("\n--- building aligner (cpu) ---")
    aligner = SpectrogramGuidedAligner(torch.device("cpu"))
    print("blank_id:", aligner.blank_id, "| vocab size:", len(aligner.vocab))

    for fmt in (fmt_hint,):
        print(f"\n--- aligner(transcript, audio, format={fmt}) ---")
        try:
            segs = aligner(transcript, audio_content=audio_bytes, audio_format=fmt)
            print("SEGMENTS:", len(segs))
            for s in segs[:12]:
                print(
                    "   token=%r start=%s end=%s refined=%s"
                    % (
                        getattr(s, "token", None),
                        getattr(s, "start_sample", None),
                        getattr(s, "end_sample", None),
                        getattr(s, "boundary_refined", None),
                    )
                )
            break
        except Exception:  # noqa: BLE001
            traceback.print_exc()


if __name__ == "__main__":
    main()

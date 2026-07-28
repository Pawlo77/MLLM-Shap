"""Aggregate per-condition log-prob endpoint summaries into a paper-ready table.

.venv/bin/python -m experiments.faithfulness.src.logprob_summarize
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("experiments/faithfulness/outputs/logprob_qwen_voxtral")
MODELS = {"qwen": "Qwen2-Audio-7B", "voxtral": "Voxtral-Mini-3B"}
CONDS = {"librispeech_original": "LibriSpeech", "tts_original": "TTS orig."}


def main() -> None:
    rows = []
    for m, mlab in MODELS.items():
        for c, clab in CONDS.items():
            f = OUT / f"{m}_{c}_logprob_summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            rows.append(
                {
                    "model": mlab,
                    "condition": clab,
                    "top_drop": round(s.get("mean_top_drop", float("nan")), 3),
                    "nontop_drop": round(s.get("mean_nontop_drop", float("nan")), 3),
                    "dz": round(s.get("dz") or float("nan"), 2),
                    "pos_rate": round(s.get("pos_rate", float("nan")), 2),
                    "pooled_rho": round(s.get("pooled_spearman", float("nan")), 2),
                    "within_median_rho": round(
                        s.get("within_sample_spearman_median") or float("nan"), 2
                    ),
                    "wilcoxon_p": s.get("wilcoxon_p_greater"),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "logprob_main.csv", index=False)

    def md(x):
        cols = list(x.columns)
        out = "| " + " | ".join(cols) + " |\n"
        out += "|" + "|".join(["---"] * len(cols)) + "|\n"
        for _, r in x.iterrows():
            out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
        return out

    (OUT / "logprob_main.md").write_text(md(df))
    print(md(df))
    print(f"wrote {OUT / 'logprob_main.csv'} and {OUT / 'logprob_main.md'}")


if __name__ == "__main__":
    main()

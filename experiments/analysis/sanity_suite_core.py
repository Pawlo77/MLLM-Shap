"""Core computations for the multilingual sanity + insight test suite.

This module is designed to be imported both from notebooks and scripts.
"""

# pylint: disable=missing-function-docstring
# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-lines
# pylint: disable=magic-value-comparison

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as SMF  # type: ignore

    HAS_STATSMODELS = True
except ModuleNotFoundError:
    SMF = None
    HAS_STATSMODELS = False


SEED: int = 1337
FIG_DIR: str = "figures_sanity_suite"
DEFAULT_WORD_MODE: str = "auto"  # 'auto'|'sentencepiece'|'roberta'|'bert'|'whitespace'

# Optional filters (leave empty/None to analyze everything present)
CASES: Optional[Sequence[str]] = None
LANGUAGES: Optional[Sequence[str]] = None

POS_CANONICAL: tuple[str, ...] = (
    "Noun",
    "PropN",
    "Verb",
    "Aux",
    "Adj",
    "Det",
    "Adv",
    "Pron",
    "Adp",
    "CConj",
    "SConj",
    "Num",
    "Punct",
    "Other",
)


@dataclass(frozen=True)
class ColumnMap:
    """Column names used by the suite."""

    # pylint: disable=too-many-instance-attributes
    # Core
    sample_id: str = "sample_id"
    case: str = "case"
    language: str = "language"
    modality: str = "modality"
    turn: str = "turn"
    is_history: str = "is_history"
    position: str = "position"
    token: str = "token"
    sv: str = "sv"
    pos: str = "pos"

    # Message-level inputs (units_df-style)
    tokens_list: str = "tokens"
    sv_list: str = "sv"
    pos_list: str = "pos"
    turn_index: str = "turn_index"
    msg_index: str = "msg_index"

    # Repo-specific convenience
    row_index: str = "row_index"
    unit_type: str = "unit_type"


def _require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}. Available: {list(df.columns)}")


def _canonicalize_case(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "UNKNOWN"
    return str(x)


def _canonicalize_language(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    return s if s else None


def _canonicalize_modality(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "text"
    s = str(x).strip().lower()
    if s in {"text", "audio"}:
        return s
    return "text"


def _canonicalize_turn(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"history", "current"}:
        return s
    return None


def canonicalize_pos(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None

    mapping = {
        "NOUN": "Noun",
        "PROPN": "PropN",
        "VERB": "Verb",
        "AUX": "Aux",
        "ADJ": "Adj",
        "DET": "Det",
        "ADV": "Adv",
        "PRON": "Pron",
        "ADP": "Adp",
        "CCONJ": "CConj",
        "SCONJ": "SConj",
        "NUM": "Num",
        "PUNCT": "Punct",
    }

    out = mapping.get(s.upper())
    if out is not None:
        return out
    if s in POS_CANONICAL:
        return s
    return "Other"


def load_data() -> Dict[str, pd.DataFrame]:
    """Load raw data using the existing repo loaders.

    Env vars:
      - SANITY_DATASET in {'multilingual','multi_sentence'}
      - SANITY_RUN (run name override)
      - SANITY_CASES comma-separated (case list override)

    Returns: {'raw_units': units_df}
    """

    cwd = Path(os.getcwd()).resolve()
    repo_root: Optional[Path] = None

    for p in [cwd, *cwd.parents]:
        if (p / "experiments" / "analysis" / "multi_sentence.py").exists():
            repo_root = p
            break

    if repo_root is None:
        raise RuntimeError(
            "Could not locate repo root containing experiments/analysis/multi_sentence.py. "
            "Run notebook from the repo workspace."
        )

    analysis_dir = repo_root / "experiments" / "analysis"
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    dataset = os.environ.get("SANITY_DATASET", "multilingual").strip().lower()
    run_override = os.environ.get("SANITY_RUN")
    cases_env = os.environ.get("SANITY_CASES")
    cases_override = [c.strip() for c in cases_env.split(",") if c.strip()] if cases_env else None

    if dataset == "multilingual":
        import multilingual as ml  # type: ignore  # pylint: disable=import-outside-toplevel

        run = run_override or ml.DEFAULT_RUN
        cases = cases_override or ["T2T", "SM2T", "SF2T"]
        units = pd.concat([ml.build_units_df(c, run=run) for c in cases], ignore_index=True)
        return {"raw_units": units}

    if dataset == "multi_sentence":
        import multi_sentence as ms  # type: ignore  # pylint: disable=import-outside-toplevel

        run = run_override or ms.DEFAULT_RUN
        cases = cases_override or ["T2T", "I_TF_M", "I_AF_M"]
        units = pd.concat([ms.build_units_df(c, run=run) for c in cases], ignore_index=True)
        return {"raw_units": units}

    raise ValueError(f"Unknown SANITY_DATASET={dataset!r}. Expected 'multilingual' or 'multi_sentence'.")


def adapt_token_level_raw(raw: pd.DataFrame, col: ColumnMap) -> pd.DataFrame:
    """Adapt a token-per-row dataframe into canonical df_tokens."""

    _require_columns(raw, [col.sample_id, col.case, col.position, col.token, col.sv], "token-level raw")

    df = raw.copy()
    df = df.rename(
        columns={
            col.sample_id: "sample_id",
            col.case: "case",
            col.language: "language",
            col.modality: "modality",
            col.turn: "turn",
            col.is_history: "is_history",
            col.position: "position",
            col.token: "token",
            col.sv: "sv",
            col.pos: "pos",
        }
    )

    df["case"] = df["case"].map(_canonicalize_case)
    df["language"] = df["language"].map(_canonicalize_language) if "language" in df.columns else None

    df["modality"] = df["modality"].map(_canonicalize_modality) if "modality" in df.columns else "text"
    df["turn"] = df["turn"].map(_canonicalize_turn) if "turn" in df.columns else None

    if "is_history" not in df.columns:
        df["is_history"] = df["turn"].map(lambda t: bool(t == "history")) if "turn" in df.columns else False
    else:
        df["is_history"] = df["is_history"].astype(bool)

    df["position"] = df["position"].astype(int)
    df["token"] = df["token"].astype(str)
    df["sv"] = df["sv"].astype(float)
    df["abs_sv"] = df["sv"].abs()
    df["pos"] = df["pos"].map(canonicalize_pos) if "pos" in df.columns else None

    df["word_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df["word"] = None

    keep = [
        "sample_id",
        "case",
        "language",
        "modality",
        "turn",
        "is_history",
        "position",
        "token",
        "sv",
        "abs_sv",
        "pos",
        "word_id",
        "word",
    ]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]


def _explode_units_row(
    *,
    case: str,
    sample_id: Any,
    language: Any,
    modality: str,
    turn: Optional[str],
    is_history: bool,
    turn_index: Any,
    msg_index: Any,
    tokens: Any,
    sv: Any,
    pos: Any = None,
) -> List[Dict[str, Any]]:
    toks = list(tokens) if isinstance(tokens, (list, tuple, np.ndarray)) else []
    sv_a = np.asarray(sv, dtype=float) if isinstance(sv, (list, tuple, np.ndarray)) else np.asarray([], dtype=float)
    m = min(len(toks), len(sv_a))
    toks = toks[:m]
    sv_a = sv_a[:m]

    pos_l: Optional[List[Any]] = None
    if pos is not None and isinstance(pos, (list, tuple, np.ndarray)):
        pos_l = list(pos)[:m]

    rows: List[Dict[str, Any]] = []
    for i in range(m):
        rows.append(
            {
                "sample_id": sample_id,
                "case": case,
                "language": language,
                "modality": modality,
                "turn": turn,
                "is_history": bool(is_history),
                "position": int(i),
                "token": str(toks[i]),
                "sv": float(sv_a[i]),
                "abs_sv": float(abs(sv_a[i])),
                "pos": canonicalize_pos(pos_l[i]) if pos_l is not None else None,
                "turn_index": turn_index,
                "msg_index": msg_index,
            }
        )
    return rows


def adapt_units_level_raw(raw_units: pd.DataFrame, col: ColumnMap) -> pd.DataFrame:
    """Adapt a message-per-row dataframe (units_df-style) into canonical df_tokens."""

    if col.sample_id in raw_units.columns:
        sid_col = col.sample_id
    elif col.row_index in raw_units.columns:
        sid_col = col.row_index
    else:
        raise ValueError(f"units-level raw: expected {col.sample_id} or {col.row_index} column")

    _require_columns(raw_units, [col.case, sid_col, col.tokens_list, col.sv_list], "units-level raw")

    dfu = raw_units.copy()
    dfu["_case"] = dfu[col.case].map(_canonicalize_case)
    dfu["_sample_id"] = dfu[sid_col]
    dfu["_language"] = dfu[col.language].map(_canonicalize_language) if col.language in dfu.columns else None

    if col.unit_type in dfu.columns:
        dfu["_modality"] = (
            dfu[col.unit_type].astype(str).str.lower().map(lambda x: "audio" if "audio" in x else "text")
        )
    elif col.modality in dfu.columns:
        dfu["_modality"] = dfu[col.modality].map(_canonicalize_modality)
    else:
        dfu["_modality"] = "text"

    dfu["_turn_index"] = dfu[col.turn_index] if col.turn_index in dfu.columns else None
    dfu["_msg_index"] = dfu[col.msg_index] if col.msg_index in dfu.columns else None
    dfu["_sort_turn"] = pd.to_numeric(dfu["_turn_index"], errors="coerce")
    dfu["_sort_msg"] = pd.to_numeric(dfu["_msg_index"], errors="coerce")
    dfu["_orig_i"] = np.arange(len(dfu), dtype=int)

    text = dfu[dfu["_modality"].eq("text")].copy()
    text = text.sort_values(["_case", "_sample_id", "_sort_turn", "_sort_msg", "_orig_i"], na_position="last")
    text = text.reset_index(drop=True)
    text["message_rank"] = text.groupby(["_case", "_sample_id"]).cumcount()
    text["is_history"] = text.groupby(["_case", "_sample_id"])["message_rank"].transform(lambda s: s < s.max())
    text["turn"] = text["is_history"].map(lambda b: "history" if bool(b) else "current")

    audio = dfu[~dfu["_modality"].eq("text")].copy()
    audio["is_history"] = False
    audio["turn"] = "current"

    dfu2 = pd.concat([text, audio], ignore_index=True)

    out_rows: List[Dict[str, Any]] = []
    for _, row in dfu2.iterrows():
        out_rows.extend(
            _explode_units_row(
                case=str(row["_case"]),
                sample_id=row["_sample_id"],
                language=row.get("_language"),
                modality=str(row["_modality"]),
                turn=_canonicalize_turn(row.get("turn")),
                is_history=bool(row.get("is_history", False)),
                turn_index=row.get("_turn_index"),
                msg_index=row.get("_msg_index"),
                tokens=row.get(col.tokens_list, []),
                sv=row.get(col.sv_list, []),
                pos=row.get(col.pos_list, None) if col.pos_list in dfu2.columns else None,
            )
        )

    df = pd.DataFrame(out_rows)
    df["case"] = df["case"].map(_canonicalize_case)
    df["language"] = df["language"].map(_canonicalize_language)
    df["modality"] = df["modality"].map(_canonicalize_modality)
    df["turn"] = df["turn"].map(_canonicalize_turn)
    df["is_history"] = df["is_history"].astype(bool)
    df["position"] = df["position"].astype(int)
    df["sv"] = df["sv"].astype(float)
    df["abs_sv"] = df["abs_sv"].astype(float)
    df["pos"] = df["pos"].map(canonicalize_pos) if "pos" in df.columns else None

    df["word_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df["word"] = None

    df = df.sort_values(["case", "sample_id", "modality", "is_history", "position"]).reset_index(drop=True)
    return df


def entropy_base2_from_abs(abs_values: np.ndarray) -> float:
    abs_values = np.asarray(abs_values, dtype=float)
    s = float(abs_values.sum())
    if s <= 0:
        return float("nan")
    p = abs_values / s
    return float(stats.entropy(p, base=2))


def _detect_bpe_scheme(tokens: Sequence[str]) -> str:
    toks = [str(t) for t in tokens]
    has_sp = any(t.startswith("▁") for t in toks)
    has_roberta = any(t.startswith("Ġ") for t in toks)
    has_bert = any(t.startswith("##") for t in toks)
    has_leading_space = any((len(t) > 0 and t[0].isspace()) for t in toks)
    if has_sp:
        return "sentencepiece"
    if has_roberta:
        return "roberta"
    if has_bert:
        return "bert"
    if has_leading_space:
        return "leading_space"
    return "unknown"


def merge_subwords(tokens: Sequence[str], mode: str = "auto") -> Tuple[np.ndarray, List[str]]:
    toks = [str(t) for t in tokens]
    n = len(toks)
    if n == 0:
        return np.asarray([], dtype=int), []
    if mode == "none":
        return np.arange(n, dtype=int), toks

    scheme = _detect_bpe_scheme(toks) if mode == "auto" else "unknown"
    word_ids = np.zeros(n, dtype=int)
    words: List[str] = [""] * n
    current_word = ""
    current_id = -1

    def _start(piece: str) -> None:
        nonlocal current_id, current_word
        current_id += 1
        current_word = piece

    def _append(piece: str) -> None:
        nonlocal current_word
        current_word = current_word + piece

    for i, raw in enumerate(toks):
        if scheme == "sentencepiece":
            if raw.startswith("▁"):
                _start(raw[1:])
            elif current_id < 0:
                _start(raw)
            else:
                _append(raw)
        elif scheme == "roberta":
            if raw.startswith("Ġ"):
                _start(raw[1:])
            elif current_id < 0:
                _start(raw)
            else:
                _append(raw)
        elif scheme == "bert":
            if raw.startswith("##"):
                piece = raw[2:]
                if current_id < 0:
                    _start(piece)
                else:
                    _append(piece)
            else:
                _start(raw)
        elif raw[:1].isspace():
            _start(raw.lstrip())
        elif current_id < 0:
            _start(raw)
        else:
            _append(raw)

        word_ids[i] = current_id
        words[i] = current_word

    return word_ids, words


def add_word_features(df_tokens: pd.DataFrame, merge_mode: str = "auto") -> pd.DataFrame:
    df = df_tokens.copy()
    mask = (df["modality"].eq("text")) & (df["turn"].eq("current"))
    df.loc[~mask, "word_id"] = pd.NA
    df.loc[~mask, "word"] = None

    parts: List[pd.DataFrame] = []
    for (_case, _sample_id), group in df[mask].groupby(["case", "sample_id"], sort=False):
        group = group.sort_values("position").copy()
        wids, wtxt = merge_subwords(group["token"].astype(str).tolist(), mode=merge_mode)
        group["word_id"] = pd.Series(wids, index=group.index, dtype="Int64")
        group["word"] = pd.Series(wtxt, index=group.index, dtype=object)
        parts.append(group)

    if parts:
        upd = pd.concat(parts).sort_index()
        df.loc[upd.index, "word_id"] = upd["word_id"]
        df.loc[upd.index, "word"] = upd["word"]

    return df


def compute_sample_metrics(df_tokens: pd.DataFrame, *, word_level: bool = False) -> pd.DataFrame:
    df = df_tokens.copy()
    for c in ["sample_id", "case", "modality", "sv", "abs_sv"]:
        if c not in df.columns:
            raise ValueError(f"df_tokens missing required column: {c}")

    is_text = df["modality"].eq("text")
    is_audio = df["modality"].eq("audio")
    is_current = df["turn"].eq("current")
    is_history = df["turn"].eq("history") | df["is_history"].astype(bool)
    grp_cols = ["sample_id", "case", "language"]

    def _count(mask: pd.Series) -> pd.Series:
        return df[mask].groupby(grp_cols)["token"].size()

    def _sum_abs(mask: pd.Series) -> pd.Series:
        return df[mask].groupby(grp_cols)["abs_sv"].sum()

    out = (
        pd.DataFrame(
            {
                "token_count_text_current": _count(is_text & is_current),
                "token_count_text_history": _count(is_text & is_history),
                "token_count_audio_current": _count(is_audio & is_current),
                "token_count_audio_history": _count(is_audio & is_history),
                "token_count_audio": _count(is_audio),
                "token_count_all_current": _count(is_current),
                "total_abs_sv_text_current": _sum_abs(is_text & is_current),
                "total_abs_sv_text_history": _sum_abs(is_text & is_history),
                "total_abs_sv_audio_current": _sum_abs(is_audio & is_current),
                "total_abs_sv_audio_history": _sum_abs(is_audio & is_history),
                "total_abs_sv_audio": _sum_abs(is_audio),
                "total_abs_sv_all_current": _sum_abs(is_current),
            }
        )
        .fillna(0.0)
        .reset_index()
    )

    denom_all = (
        out["total_abs_sv_text_history"] + out["total_abs_sv_text_current"] + out["total_abs_sv_audio"]
    ).replace(0.0, np.nan)
    denom_text = (out["total_abs_sv_text_history"] + out["total_abs_sv_text_current"]).replace(0.0, np.nan)
    out["history_retention_pct_all"] = 100.0 * (out["total_abs_sv_text_history"] / denom_all)
    out["history_retention_pct_textonly"] = 100.0 * (out["total_abs_sv_text_history"] / denom_text)

    cur_all = df[is_current].copy()
    if cur_all.empty:
        out["shapley_entropy_all_current"] = np.nan
        out["density_all_current"] = np.nan
    else:
        ent_all = (
            cur_all.groupby(grp_cols)["abs_sv"]
            .apply(lambda s: entropy_base2_from_abs(s.to_numpy(dtype=float)))
            .rename("shapley_entropy_all_current")
            .reset_index()
        )
        dens_all = (
            cur_all.groupby(grp_cols)["abs_sv"]
            .apply(lambda s: float(np.mean(s.to_numpy(dtype=float))) if len(s) else float("nan"))
            .rename("density_all_current")
            .reset_index()
        )
        out = out.merge(ent_all, on=grp_cols, how="left")
        out = out.merge(dens_all, on=grp_cols, how="left")

    cur_text = df[is_text & is_current].copy()
    if cur_text.empty:
        out["shapley_entropy_text_current"] = np.nan
        out["density_text_current"] = np.nan
        out["shapley_entropy_word_current"] = np.nan
        return out

    ent_tok = (
        cur_text.groupby(grp_cols)["abs_sv"]
        .apply(lambda s: entropy_base2_from_abs(s.to_numpy(dtype=float)))
        .rename("shapley_entropy_text_current")
        .reset_index()
    )
    dens_tok = (
        cur_text.groupby(grp_cols)["abs_sv"]
        .apply(lambda s: float(np.mean(s.to_numpy(dtype=float))) if len(s) else float("nan"))
        .rename("density_text_current")
        .reset_index()
    )
    out = out.merge(ent_tok, on=grp_cols, how="left")
    out = out.merge(dens_tok, on=grp_cols, how="left")

    if word_level:
        if "word_id" not in cur_text.columns:
            raise ValueError("word_level=True requires word_id/word columns; call add_word_features first")
        curw = cur_text.dropna(subset=["word_id"]).copy()
        if curw.empty:
            out["shapley_entropy_word_current"] = np.nan
        else:
            w = curw.groupby(grp_cols + ["word_id"], as_index=False)["abs_sv"].sum()
            ent_word = (
                w.groupby(grp_cols)["abs_sv"]
                .apply(lambda s: entropy_base2_from_abs(s.to_numpy(dtype=float)))
                .rename("shapley_entropy_word_current")
                .reset_index()
            )
            out = out.merge(ent_word, on=grp_cols, how="left")
    else:
        out["shapley_entropy_word_current"] = np.nan

    cur_text["pos"] = cur_text["pos"].map(canonicalize_pos).fillna("Other")
    pos_sum = cur_text.groupby(grp_cols + ["pos"], as_index=False)["abs_sv"].sum()
    pos_tot = cur_text.groupby(grp_cols, as_index=False)["abs_sv"].sum().rename(columns={"abs_sv": "_pos_total"})
    pos_sum = pos_sum.merge(pos_tot, on=grp_cols, how="left")
    pos_sum["pos_share"] = pos_sum["abs_sv"] / pos_sum["_pos_total"].replace(0.0, np.nan)
    pos_wide = pos_sum.pivot_table(index=grp_cols, columns="pos", values="pos_share", aggfunc="first").reset_index()
    pos_wide.columns = [c if c in grp_cols else f"pos_share_{c}" for c in pos_wide.columns]
    out = out.merge(pos_wide, on=grp_cols, how="left")

    for p in POS_CANONICAL:
        c = f"pos_share_{p}"
        if c not in out.columns:
            out[c] = np.nan

    return out


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    r = np.random.default_rng(seed)
    n = values.size
    boots = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        samp = values[r.integers(0, n, size=n)]
        boots[i] = float(statistic(samp))

    alpha = (1 - ci) / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    est = float(statistic(values))
    return est, lo, hi


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")

    nx, ny = x.size, y.size
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    denom = nx + ny - 2
    if denom <= 0:
        return float("nan")

    sp = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / denom)
    if sp <= 0 or not np.isfinite(sp):
        return float("nan")

    return float((np.mean(x) - np.mean(y)) / sp)


def cliffs_delta(x: np.ndarray, y: np.ndarray, max_n: int = 2000, seed: int = SEED) -> float:
    r = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")

    if x.size > max_n:
        x = r.choice(x, size=max_n, replace=False)
    if y.size > max_n:
        y = r.choice(y, size=max_n, replace=False)

    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))

    denom = x.size * y.size
    return float((gt - lt) / denom) if denom else float("nan")


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p

    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks

    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]

    out = np.empty(n, dtype=float)
    out[order] = np.clip(q_sorted, 0.0, 1.0)
    return out


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.all(x == 0):
        return 0.0

    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def bridge_from_abs(hist_abs: np.ndarray, cur_abs: np.ndarray) -> np.ndarray:
    hist_abs = np.asarray(hist_abs, dtype=float)
    cur_abs = np.asarray(cur_abs, dtype=float)
    if hist_abs.size == 0 or cur_abs.size == 0:
        return np.zeros((hist_abs.size, cur_abs.size), dtype=float)
    return np.outer(hist_abs, cur_abs)


def hubness_metrics_from_h(h_mat: np.ndarray, topk: Sequence[int] = (1, 3)) -> Dict[str, float]:
    h_mat = np.asarray(h_mat, dtype=float)
    if h_mat.size == 0:
        return {
            "total_bridge_mass": float("nan"),
            "hubness": float("nan"),
            "col_gini": float("nan"),
            "col_entropy": float("nan"),
            "top1_col_share": float("nan"),
            "top3_col_share": float("nan"),
        }

    col = h_mat.sum(axis=0)
    total = float(col.sum())
    if total <= 0:
        return {
            "total_bridge_mass": 0.0,
            "hubness": float("nan"),
            "col_gini": 0.0,
            "col_entropy": float("nan"),
            "top1_col_share": float("nan"),
            "top3_col_share": float("nan"),
        }

    out: Dict[str, float] = {
        "total_bridge_mass": total,
        "hubness": float(np.max(col) / total),
        "col_gini": gini(col),
        "col_entropy": entropy_base2_from_abs(col),
    }

    for k in topk:
        kk = int(k)
        share = float(np.sum(np.sort(col)[-kk:]) / total) if kk > 0 else float("nan")
        out[f"top{kk}_col_share"] = share

    out.setdefault("top1_col_share", float("nan"))
    out.setdefault("top3_col_share", float("nan"))
    return out


def compute_bridge_metrics(df_tokens: pd.DataFrame) -> pd.DataFrame:
    text = df_tokens[df_tokens["modality"].eq("text")].copy()
    keys = ["sample_id", "case", "language"]
    if text.empty:
        return pd.DataFrame()

    if "turn_index" in text.columns and "msg_index" in text.columns:
        msg = text.groupby(keys + ["turn_index", "msg_index"], as_index=False).agg(
            {"abs_sv": list, "is_history": "max"}
        )
        msg["_turn_i"] = pd.to_numeric(msg["turn_index"], errors="coerce")
        msg["_msg_i"] = pd.to_numeric(msg["msg_index"], errors="coerce")
        msg = msg.sort_values(keys + ["_turn_i", "_msg_i"], na_position="last")
        msg["message_rank"] = msg.groupby(keys).cumcount()

        out_rows: List[Dict[str, Any]] = []
        for _, group in msg.groupby(keys, sort=False):
            group = group.sort_values("message_rank")
            if len(group) < 2:
                continue
            prev = group.iloc[-2]
            cur = group.iloc[-1]
            h_mat = bridge_from_abs(np.asarray(prev["abs_sv"], dtype=float), np.asarray(cur["abs_sv"], dtype=float))
            m = hubness_metrics_from_h(h_mat)
            out_rows.append(
                {
                    **{k: prev[k] for k in keys},
                    **m,
                    "n_hist_tokens": int(len(prev["abs_sv"])),
                    "n_cur_tokens": int(len(cur["abs_sv"])),
                }
            )
        return pd.DataFrame(out_rows)

    out_rows = []
    for (sample_id, case, language), group in text.groupby(keys, sort=False):
        hist = group[group["turn"].eq("history") | group["is_history"].astype(bool)]
        cur = group[group["turn"].eq("current")]
        if hist.empty or cur.empty:
            continue

        h_mat = bridge_from_abs(hist["abs_sv"].to_numpy(dtype=float), cur["abs_sv"].to_numpy(dtype=float))
        m = hubness_metrics_from_h(h_mat)
        out_rows.append(
            {
                "sample_id": sample_id,
                "case": case,
                "language": language,
                **m,
                "n_hist_tokens": int(len(hist)),
                "n_cur_tokens": int(len(cur)),
            }
        )

    return pd.DataFrame(out_rows)


def test_denominator_artifact(sample_metrics: pd.DataFrame, *, by: str = "case") -> pd.DataFrame:
    df = sample_metrics.copy()
    needed = ["history_retention_pct_all", "history_retention_pct_textonly", by]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    df = df.dropna(subset=["history_retention_pct_all", "history_retention_pct_textonly"])
    out_rows = []

    for key, group in df.groupby(by, dropna=False, sort=False):
        x = group["history_retention_pct_all"].to_numpy(dtype=float)
        y = group["history_retention_pct_textonly"].to_numpy(dtype=float)
        d = y - x
        est, lo, hi = bootstrap_ci(d, np.mean)
        t_res = stats.ttest_rel(y, x, nan_policy="omit")
        sd = float(np.nanstd(d, ddof=1))
        d_eff = float(np.nanmean(d) / sd) if sd > 0 else float("nan")
        out_rows.append(
            {
                by: key,
                "n": int(len(group)),
                "mean_all": float(np.nanmean(x)),
                "mean_textonly": float(np.nanmean(y)),
                "mean_delta_textonly_minus_all": float(est),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "p_ttest_rel": float(t_res.pvalue) if np.isfinite(t_res.pvalue) else float("nan"),
                "effect_d_paired": d_eff,
            }
        )

    out = pd.DataFrame(out_rows)
    print("[Denominator artifact] Large +Δ means audio denominator suppresses retention_all.")
    return out


def test_length_confound(
    sample_metrics: pd.DataFrame,
    *,
    metric: str,
    controls: Sequence[str] = ("token_count_text_current",),
    group_cols: Sequence[str] = ("case",),
    alpha: float = 0.05,
) -> pd.DataFrame:
    df = sample_metrics.copy()
    if metric not in df.columns:
        raise ValueError(f"metric not found: {metric}")

    df = df.dropna(subset=[metric, "language"])
    rows = []

    for key, group in df.groupby(list(group_cols), dropna=False, sort=False):
        if group["language"].nunique() < 2:
            continue
        for c in controls:
            if c not in group.columns:
                raise ValueError(f"missing control: {c}")

        formula = f"{metric} ~ " + " + ".join(list(controls)) + " + C(language)"

        if HAS_STATSMODELS and SMF is not None:
            model = SMF.ols(formula=formula, data=group).fit(cov_type="HC3")
            df_resid = max(int(model.df_resid), 1)
            tcrit = float(stats.t.ppf(1 - alpha / 2, df=df_resid))

            for term, coef in model.params.items():
                if term == "Intercept" or term.startswith("C(language)") or term in controls:
                    se = float(model.bse.get(term, np.nan))
                    ci_lo = float(coef - tcrit * se) if np.isfinite(se) else float("nan")
                    ci_hi = float(coef + tcrit * se) if np.isfinite(se) else float("nan")
                    p_val = float(model.pvalues.get(term, np.nan))
                    rec = {
                        group_cols[i]: (key[i] if isinstance(key, tuple) else key) for i in range(len(group_cols))
                    }
                    rec.update(
                        {
                            "metric": metric,
                            "term": term,
                            "coef": float(coef),
                            "ci_lo": ci_lo,
                            "ci_hi": ci_hi,
                            "p": p_val,
                            "n": int(len(group)),
                        }
                    )
                    rows.append(rec)
        else:
            rec = {group_cols[i]: (key[i] if isinstance(key, tuple) else key) for i in range(len(group_cols))}
            rec.update(
                {
                    "metric": metric,
                    "term": "statsmodels_missing",
                    "coef": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "p": float("nan"),
                    "n": int(len(group)),
                }
            )
            rows.append(rec)

    out = pd.DataFrame(rows)
    print(f"[Length confound] metric={metric} rows={len(out)}")
    return out


def permutation_test_pos_share(
    df_tokens: pd.DataFrame,
    *,
    language: str,
    pos: str = "Adj",
    n_perm: int = 500,
    seed: int = SEED,
) -> Dict[str, Any]:
    r = np.random.default_rng(seed)
    df = df_tokens[(df_tokens["modality"].eq("text")) & (df_tokens["turn"].eq("current"))].copy()
    df["pos"] = df["pos"].map(canonicalize_pos).fillna("Other")
    df = df[df["language"].astype(str) == str(language)]

    if df.empty:
        return {"language": language, "pos": pos, "n_tokens": 0, "observed": float("nan"), "p_perm": float("nan")}

    keys = ["sample_id", "case", "language"]

    obs = df.groupby(keys + ["pos"], as_index=False)["abs_sv"].sum()
    tot = df.groupby(keys, as_index=False)["abs_sv"].sum().rename(columns={"abs_sv": "total"})
    obs = obs.merge(tot, on=keys, how="left")
    obs["share"] = obs["abs_sv"] / obs["total"].replace(0.0, np.nan)
    observed = float(np.nanmean(obs.loc[obs["pos"].eq(pos), "share"].to_numpy(dtype=float)))

    pos_labels = df["pos"].to_numpy(dtype=object)
    null = []
    for _ in range(int(n_perm)):
        perm = pos_labels.copy()
        r.shuffle(perm)
        dfp = df.copy()
        dfp["pos_perm"] = perm
        s = dfp.groupby(keys + ["pos_perm"], as_index=False)["abs_sv"].sum()
        s = s.merge(tot, on=keys, how="left")
        s["share"] = s["abs_sv"] / s["total"].replace(0.0, np.nan)
        null.append(float(np.nanmean(s.loc[s["pos_perm"].eq(pos), "share"].to_numpy(dtype=float))))

    null_a = np.asarray(null, dtype=float)
    null_mean = float(np.nanmean(null_a))
    p_val = float(
        (np.sum(np.abs(null_a - null_mean) >= np.abs(observed - null_mean)) + 1)
        / (np.sum(np.isfinite(null_a)) + 1)
    )

    return {
        "language": language,
        "pos": pos,
        "n_tokens": int(len(df)),
        "observed_mean_share": observed,
        "null_mean_share": null_mean,
        "p_perm_two_sided": p_val,
    }


def null_bridge_hubness(
    df_tokens: pd.DataFrame,
    *,
    n_sim: int = 200,
    alpha_dirichlet: float = 1.0,
    seed: int = SEED,
) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    bm = compute_bridge_metrics(df_tokens)
    if bm is None or bm.empty:
        return pd.DataFrame()

    out_rows = []
    for _, row in bm.iterrows():
        n_cur = int(row.get("n_cur_tokens", 0))
        hub_obs = float(row.get("hubness", np.nan))
        if n_cur <= 1 or not np.isfinite(hub_obs):
            continue

        sims = []
        for _ in range(int(n_sim)):
            p = r.dirichlet(alpha=np.full(n_cur, float(alpha_dirichlet)))
            sims.append(float(np.max(p)))

        sims_a = np.asarray(sims, dtype=float)
        out_rows.append(
            {
                "sample_id": row["sample_id"],
                "case": row["case"],
                "language": row.get("language", None),
                "hubness_observed": hub_obs,
                "hubness_null_mean": float(np.mean(sims_a)),
                "hubness_null_ci_lo": float(np.quantile(sims_a, 0.025)),
                "hubness_null_ci_hi": float(np.quantile(sims_a, 0.975)),
                "delta_hubness_obs_minus_nullmean": float(hub_obs - np.mean(sims_a)),
                "n_cur_tokens": n_cur,
            }
        )

    out = pd.DataFrame(out_rows)
    print("[Null bridge] Δhubness>0 => more hubness than Dirichlet baseline")
    return out


def test_case_effect_controlling_length(
    sample_metrics: pd.DataFrame,
    *,
    metric: str = "shapley_entropy_text_current",
    length_col: str = "token_count_text_current",
    alpha: float = 0.05,
) -> pd.DataFrame:
    df = sample_metrics.copy()
    needed = ["case", metric, length_col]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    df = df.dropna(subset=[metric, length_col, "case"]).copy()
    if df.empty:
        return pd.DataFrame()

    has_lang = "language" in df.columns and df["language"].notna().any() and df["language"].nunique(dropna=True) >= 2

    if HAS_STATSMODELS and SMF is not None:
        formula = f"{metric} ~ {length_col} + C(case)" + (" + C(language)" if has_lang else "")
        model = SMF.ols(formula=formula, data=df).fit(cov_type="HC3")

        df_resid = max(int(model.df_resid), 1)
        tcrit = float(stats.t.ppf(1 - alpha / 2, df=df_resid))

        rows = []
        for term, coef in model.params.items():
            if term == "Intercept" or term == length_col or term.startswith("C(case)"):
                se = float(model.bse.get(term, np.nan))
                ci_lo = float(coef - tcrit * se) if np.isfinite(se) else float("nan")
                ci_hi = float(coef + tcrit * se) if np.isfinite(se) else float("nan")
                p_val = float(model.pvalues.get(term, np.nan))
                rows.append(
                    {
                        "metric": metric,
                        "term": term,
                        "coef": float(coef),
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "p": p_val,
                        "n": int(len(df)),
                        "n_languages": int(df["language"].nunique(dropna=True)) if "language" in df.columns else 0,
                    }
                )

        out = pd.DataFrame(rows)
        return out.sort_values(["term"]).reset_index(drop=True)

    # NumPy HC3 OLS fallback (kept from notebook implementation)
    y = df[metric].to_numpy(dtype=float)
    x_len = df[length_col].to_numpy(dtype=float)

    case_levels = sorted(df["case"].astype(str).unique().tolist())
    case_ref = case_levels[0] if case_levels else ""
    lang_levels = sorted(df["language"].dropna().astype(str).unique().tolist()) if has_lang else []
    lang_ref = lang_levels[0] if lang_levels else ""

    cols = ["Intercept", length_col]
    x_parts = [np.ones((len(df), 1), dtype=float), x_len.reshape(-1, 1)]

    for lvl in case_levels:
        if lvl == case_ref:
            continue
        cols.append(f"C(case)[T.{lvl}]")
        x_parts.append((df["case"].astype(str).to_numpy() == lvl).astype(float).reshape(-1, 1))

    if has_lang:
        for lvl in lang_levels:
            if lvl == lang_ref:
                continue
            cols.append(f"C(language)[T.{lvl}]")
            x_parts.append((df["language"].astype(str).to_numpy() == lvl).astype(float).reshape(-1, 1))

    x_mat = np.concatenate(x_parts, axis=1)
    xtx = x_mat.T @ x_mat
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x_mat.T @ y)
    yhat = x_mat @ beta
    resid = y - yhat

    hat_diag = np.sum((x_mat @ xtx_inv) * x_mat, axis=1)
    denom = np.clip(1.0 - hat_diag, 1e-12, np.inf)

    w = (resid**2) / (denom**2)
    meat = (x_mat.T * w) @ x_mat
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    df_resid = max(int(len(df) - x_mat.shape[1]), 1)
    tvals = beta / np.where(se > 0, se, np.nan)
    pvals = 2 * stats.t.sf(np.abs(tvals), df=df_resid)
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=df_resid))

    rows = []
    for j, term in enumerate(cols):
        if term == "Intercept" or term == length_col or term.startswith("C(case)"):
            ci_lo = float(beta[j] - tcrit * se[j]) if np.isfinite(se[j]) else float("nan")
            ci_hi = float(beta[j] + tcrit * se[j]) if np.isfinite(se[j]) else float("nan")
            rows.append(
                {
                    "metric": metric,
                    "term": term,
                    "coef": float(beta[j]),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p": float(pvals[j]) if np.isfinite(pvals[j]) else float("nan"),
                    "n": int(len(df)),
                    "n_languages": int(df["language"].nunique(dropna=True)) if "language" in df.columns else 0,
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["term"]).reset_index(drop=True)


def compare_groups(
    df: pd.DataFrame,
    metric: str,
    group_cols: Sequence[str],
    *,
    min_n: int = 20,
    n_boot: int = 2000,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metric not in df.columns:
        raise ValueError(f"metric not found: {metric}")

    d = df.dropna(subset=[metric]).copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    g_rows = []
    for key, group in d.groupby(list(group_cols), dropna=False, sort=False):
        vals = group[metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < int(min_n):
            continue
        est, lo, hi = bootstrap_ci(vals, np.mean, n_boot=n_boot, seed=seed)
        rec = {group_cols[i]: (key[i] if isinstance(key, tuple) else key) for i in range(len(group_cols))}
        rec.update({"metric": metric, "mean": est, "ci95_lo": lo, "ci95_hi": hi, "n": int(vals.size)})
        g_rows.append(rec)

    group_summary = pd.DataFrame(g_rows)

    pw_rows = []
    if not group_summary.empty:
        keys = group_summary[group_cols].drop_duplicates().to_dict("records")
        r = np.random.default_rng(seed)
        for i, a_key in enumerate(keys):
            for b_key in keys[i + 1:]:

                ga = d
                gb = d
                for c in group_cols:
                    ga = ga[ga[c].astype(str) == str(a_key[c])]
                    gb = gb[gb[c].astype(str) == str(b_key[c])]

                xa = ga[metric].to_numpy(dtype=float)
                xb = gb[metric].to_numpy(dtype=float)
                xa = xa[np.isfinite(xa)]
                xb = xb[np.isfinite(xb)]
                if xa.size < int(min_n) or xb.size < int(min_n):
                    continue

                diff = float(np.mean(xa) - np.mean(xb))
                tt = stats.ttest_ind(xa, xb, equal_var=False, nan_policy="omit")

                boots = []
                for _ in range(2000):
                    sa = xa[r.integers(0, xa.size, size=xa.size)]
                    sb = xb[r.integers(0, xb.size, size=xb.size)]
                    boots.append(float(np.mean(sa) - np.mean(sb)))

                boots_a = np.asarray(boots, dtype=float)
                lo = float(np.quantile(boots_a, 0.025))
                hi = float(np.quantile(boots_a, 0.975))

                pw_rows.append(
                    {
                        "metric": metric,
                        "group_a": a_key,
                        "group_b": b_key,
                        "mean_diff_a_minus_b": diff,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "p_welch_t": float(tt.pvalue),
                        "cohens_d": cohens_d(xa, xb),
                        "cliffs_delta": cliffs_delta(xa, xb, seed=seed),
                        "n_a": int(xa.size),
                        "n_b": int(xb.size),
                    }
                )

    pairwise = pd.DataFrame(pw_rows)
    if not pairwise.empty:
        pairwise["q_bh"] = bh_fdr(pairwise["p_welch_t"].to_numpy(dtype=float))

    return group_summary, pairwise


def build_findings(
    *,
    sample_tok: pd.DataFrame,
    sample_word: pd.DataFrame,
    bridge: pd.DataFrame,
    denom: pd.DataFrame,
    q_max: float = 0.10,
    min_abs_effect: float = 0.05,
    top_n: int = 25,
) -> pd.DataFrame:
    findings_frames = []

    scan_specs = [
        (sample_tok, "shapley_entropy_text_current", ["language"], "token::entropy::by_lang"),
        (sample_tok, "shapley_entropy_text_current", ["case"], "token::entropy::by_case"),
        (sample_tok, "density_text_current", ["language"], "token::density::by_lang"),
        (sample_tok, "density_text_current", ["case"], "token::density::by_case"),
        (sample_tok, "history_retention_pct_all", ["case"], "token::ret_all::by_case"),
        (sample_tok, "history_retention_pct_textonly", ["case"], "token::ret_text::by_case"),
        (sample_word, "shapley_entropy_word_current", ["language"], "word::entropy::by_lang"),
        (bridge, "hubness", ["case"], "bridge::hubness::by_case"),
    ]

    for df, metric, group_cols, label in scan_specs:
        if df is None or df.empty or metric not in df.columns:
            continue
        _, pw = compare_groups(df, metric=metric, group_cols=group_cols, min_n=15)
        if pw.empty:
            continue
        pw = pw.copy()
        pw["scan"] = label
        pw["effect_mag"] = pw["mean_diff_a_minus_b"].abs()
        pw["passes_ci"] = (pw["ci95_lo"] > 0) | (pw["ci95_hi"] < 0)
        pw["passes_q"] = pw["q_bh"] <= q_max
        pw["passes_effect"] = pw["effect_mag"] >= min_abs_effect
        pw["tokenization_artifact"] = False
        pw["denominator_artifact"] = False
        findings_frames.append(pw)

    if not findings_frames:
        return pd.DataFrame()

    f = pd.concat(findings_frames, ignore_index=True)

    if denom is not None and not denom.empty and "case" in denom.columns:
        denom_map = {str(r["case"]): float(r.get("mean_delta_textonly_minus_all", 0.0)) for _, r in denom.iterrows()}

        def _denom_flag(row: pd.Series) -> bool:
            if row["metric"] != "history_retention_pct_all":
                return False
            ga = row["group_a"]
            if isinstance(ga, dict) and "case" in ga:
                return float(denom_map.get(str(ga["case"]), 0.0)) > 5.0
            return False

        f["denominator_artifact"] = f.apply(_denom_flag, axis=1)

    tok = f[(f["metric"] == "shapley_entropy_text_current") & (f["scan"].str.contains("by_lang"))].copy()
    word = f[(f["metric"] == "shapley_entropy_word_current") & (f["scan"].str.contains("by_lang"))].copy()

    if not tok.empty and not word.empty:
        word_q = {}
        for _, r in word.iterrows():
            ga, gb = r["group_a"], r["group_b"]
            if isinstance(ga, dict) and isinstance(gb, dict) and "language" in ga and "language" in gb:
                word_q[(str(ga["language"]), str(gb["language"]))] = float(r.get("q_bh", 1.0))

        def _tok_art(row: pd.Series) -> bool:
            ga, gb = row["group_a"], row["group_b"]
            if not (isinstance(ga, dict) and isinstance(gb, dict) and "language" in ga and "language" in gb):
                return False
            q_word = float(word_q.get((str(ga["language"]), str(gb["language"])), 1.0))
            q_tok = float(row.get("q_bh", 1.0))
            tok_passes = q_tok <= q_max
            word_fails = q_word > q_max
            return tok_passes and word_fails

        f.loc[tok.index, "tokenization_artifact"] = tok.apply(_tok_art, axis=1).to_numpy(dtype=bool)

    f["robustness_score"] = (
        f["passes_ci"].astype(int)
        + f["passes_q"].astype(int)
        + f["passes_effect"].astype(int)
        - f["tokenization_artifact"].astype(int)
        - f["denominator_artifact"].astype(int)
    )

    f = f.sort_values(["robustness_score", "effect_mag"], ascending=[False, False]).reset_index(drop=True)
    return f.head(int(top_n))

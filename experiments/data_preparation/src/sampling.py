"""Stratified sampling utilities."""

from __future__ import annotations

import pandas as pd


def stratified_sample(
    pool: pd.DataFrame,
    n_target: int,
    strat_col: str = "token_count",
    random_state: int = 0,
) -> pd.DataFrame:
    """Stratified sample of *n_target* rows from *pool*, balanced across *strat_col*.

    If the pool is smaller than *n_target*, all rows are returned.
    Remaining quota is filled from the un-sampled portion if some groups are too small.

    Parameters
    ----------
    pool         : Source DataFrame to sample from.
    n_target     : Desired number of rows in the output.
    strat_col    : Column to stratify on.
    random_state : Random seed for reproducibility.
    """
    if len(pool) <= n_target:
        return pool.reset_index(drop=True)

    groups = sorted(pool[strat_col].unique())
    base_n = n_target // len(groups)
    remainder = n_target % len(groups)
    parts: list[pd.DataFrame] = []
    selected_idx: list[int] = []

    for i, g in enumerate(groups):
        n = base_n + (1 if i < remainder else 0)
        group_df = pool[pool[strat_col] == g]
        take = min(n, len(group_df))
        if take > 0:
            samp = group_df.sample(n=take, random_state=random_state)
            parts.append(samp)
            selected_idx.extend(samp.index.tolist())

    result = pd.concat(parts, ignore_index=True)

    if len(result) < n_target:
        remaining = pool.drop(index=selected_idx)
        need = min(n_target - len(result), len(remaining))
        if need > 0:
            filler = remaining.sample(n=need, random_state=random_state)
            result = pd.concat([result, filler], ignore_index=True)

    return result.reset_index(drop=True)

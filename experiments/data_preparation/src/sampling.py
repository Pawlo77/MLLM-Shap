"""Stratified sampling utilities.

Functions to perform stratified and group-wise sampling used during dataset
construction and downsampling.
"""

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


def sample_fraction_by_group(
    df: pd.DataFrame,
    group_col: str,
    frac: float,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample *frac* of rows within each group in *group_col*."""
    return (
        df.groupby(group_col, group_keys=False)
        .apply(
            lambda x: x.sample(frac=frac, random_state=random_state),
            include_groups=False,
        )
        .reset_index(drop=True)
    )


def sample_fraction_by_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    frac: float,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample *frac* of rows within each combination of *group_cols*."""
    return (
        df.groupby(group_cols, group_keys=False)
        .apply(
            lambda x: x.sample(frac=frac, random_state=random_state),
            include_groups=False,
        )
        .reset_index(drop=True)
    )


def sample_n_per_group(
    df: pd.DataFrame,
    group_col: str,
    n: int,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample up to *n* rows per group."""
    return (
        df.groupby(group_col, as_index=False)
        .apply(lambda g: g.sample(n=min(n, len(g)), random_state=random_state))
        .reset_index(drop=True)
    )

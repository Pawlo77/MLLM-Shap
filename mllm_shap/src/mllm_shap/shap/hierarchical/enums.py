"""Configuration for possible modes."""

from enum import Enum


class Mode(int, Enum):
    """
    Defines supported grouping modes.

    Warning:
        Depending on mode, first level groups might be large -
        only `TEXT` mode guarantees small first level groups, as
        only then it is divided by size corresponding to
        log(number of tokens, k) formula. Other modes
        do not divide first level groups at all, therefore
        first level call will have cost of up to 2^(number of groups),
        depending on shap explainer algorithm used.m
    """

    TEXT = 0
    """
    Splits first level groups just by size, requires single modality.
    As most models always include system prompts, this restricts
    usage to text-only mode.
    """

    MULTI_MODAL = 1
    """
    Splits first level groups according to size and modalities.
    Results in moderate first level group sizes, but might mix
    different users' inputs together. Cheapest mode for multimodal inputs.
    """

    MULTI_MODAL_MULTI_USER = 2
    """
    Splits first level groups according to size, modalities, and users.
    Results in larger first level groups, but better preserves logical structure.
    """

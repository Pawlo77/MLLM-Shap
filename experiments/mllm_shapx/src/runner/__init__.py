"""Runner package: variant expansion, staged execution, and result writing."""

from mllm_shap.shap.base.approx import BaseShapApproximation

from ..audio_utils import serialize_result_with_audio
from .execution import run_single_sentence_variant
from .helpers import _LinearSampleScaler, _try_set_num_samples
from .io_utils import compute_modality_summary, serialize_conversation
from .stages import ChatBuilder, ExplainerRunner, ResultWriter, RowSelector
from .types import ExpandedVariant
from .variants import (
    _expand_exact,
    _expand_hierarchical,
    _expand_mc_like,
    expand_variants,
    pick_device,
)


_compute_modality_summary = compute_modality_summary
_serialize_conversation = serialize_conversation


__all__ = [
    "BaseShapApproximation",
    "ExpandedVariant",
    "RowSelector",
    "ChatBuilder",
    "ExplainerRunner",
    "ResultWriter",
    "_LinearSampleScaler",
    "_try_set_num_samples",
    "_expand_exact",
    "_expand_mc_like",
    "_expand_hierarchical",
    "expand_variants",
    "pick_device",
    "run_single_sentence_variant",
    "serialize_result_with_audio",
    "compute_modality_summary",
    "serialize_conversation",
]

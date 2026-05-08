"""Core composable building blocks for SHAP internals."""

from .contracts import (
    EstimationResult,
    Estimator,
    SamplingStrategy,
    StopDecision,
    StoppingPolicy,
)
from .engine import SamplingEngine, SamplingStats
from .estimation import (
    CallableEstimator,
    CallableStoppingPolicy,
    FixedThresholdStoppingPolicy,
)
from .sampling import CallableAdapterStrategy
from .telemetry import (
    CacheMetrics,
    JSONProbeSink,
    LogProbeSink,
    MaskMetrics,
    ProbeSink,
    StageTimer,
    TelemetryData,
    TelemetryProbe,
    TimingMetrics,
)

__all__ = [
    "SamplingStrategy",
    "Estimator",
    "StoppingPolicy",
    "EstimationResult",
    "StopDecision",
    "SamplingEngine",
    "SamplingStats",
    "CallableAdapterStrategy",
    "CallableEstimator",
    "CallableStoppingPolicy",
    "FixedThresholdStoppingPolicy",
    "TelemetryProbe",
    "ProbeSink",
    "LogProbeSink",
    "JSONProbeSink",
    "StageTimer",
    "TelemetryData",
    "CacheMetrics",
    "MaskMetrics",
    "TimingMetrics",
]

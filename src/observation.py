"""Bounded observation semantics shared by all policies.

The native C++ side now emits features in [0, 1]. Delay features are encoded
with log1p(seconds)/log1p(120 seconds), preserving ranking while limiting the
scale gap between radio/queue delay and CPU/battery/memory features.
"""
import numpy as np

OBSERVATION_VERSION = 3
MAX_DELAY_SECONDS = 120.0
_LOG_DELAY_DENOMINATOR = float(np.log1p(MAX_DELAY_SECONDS))


def encode_delay_feature(seconds: float) -> float:
    value = float(np.clip(seconds, 0.0, MAX_DELAY_SECONDS))
    return float(np.log1p(value) / _LOG_DELAY_DENOMINATOR)


def decode_delay_feature(feature: float) -> float:
    value = float(np.clip(feature, 0.0, 1.0))
    return float(np.expm1(value * _LOG_DELAY_DENOMINATOR))


def normalize_observation(raw_observation, config=None) -> np.ndarray:
    """Validate and bound the already normalized 60-value C++ observation."""
    obs = np.asarray(raw_observation, dtype=np.float32).copy()
    if obs.shape != (60,):
        raise ValueError(f"Expected 60 observation values, received {obs.shape}")
    if not np.isfinite(obs).all():
        raise ValueError("ns-3 observation contains NaN or infinity")
    # All 60 fields are explicitly normalized by the C++ source. Clipping is a
    # defensive guard against small floating-point overshoots only.
    return np.clip(obs, 0.0, 1.0).astype(np.float32, copy=False)

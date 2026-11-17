import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.observation import encode_delay_feature, decode_delay_feature, normalize_observation


def test_log_delay_encode_decode_and_saturation():
    for seconds in (0.0, 0.5, 2.0, 20.0, 120.0):
        assert decode_delay_feature(encode_delay_feature(seconds)) == pytest.approx(seconds)
    assert encode_delay_feature(500.0) == 1.0


def test_observation_validator_bounds_all_features():
    raw = np.linspace(-0.1, 1.1, 60, dtype=np.float32)
    normalized = normalize_observation(raw, {})
    assert normalized.shape == (60,)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_observation_validator_rejects_non_finite():
    raw = np.zeros(60, dtype=np.float32)
    raw[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        normalize_observation(raw, {})

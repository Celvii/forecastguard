"""Quantile forecast postprocessing for ForecastGuard.

Non-crossing adjustment is a display/decision guard, not calibration.
"""
import numpy as np


def non_crossing_quantiles(q05, q80, q95):
    """Return nonnegative (lower, q80, upper, repaired) for matching arrays.

    Preserve the estimated Q80 inventory policy and use a conservative
    envelope if independently trained quantiles cross.
    """
    arrays = [np.asarray(x, dtype=float) for x in (q05, q80, q95)]
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("Quantile predictions must have matching shapes")
    if not all(np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("Quantile predictions must be finite")

    stacked = np.maximum(np.stack(arrays, axis=-1), 0.0)
    repaired = (stacked[..., 0] > stacked[..., 1]) | (stacked[..., 1] > stacked[..., 2])
    return (
        np.min(stacked, axis=-1),
        stacked[..., 1],
        np.max(stacked, axis=-1),
        repaired,
    )

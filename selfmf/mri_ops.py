"""Measurement-domain utilities for complex MRI k-space."""
import numpy as np

def enforce_acquired_samples(predicted, measured, mask):
    """Return predicted k-space with acquired entries replaced by measurements."""
    predicted = np.asarray(predicted)
    measured = np.asarray(measured)
    mask = np.asarray(mask, dtype=bool)
    if predicted.shape != measured.shape or predicted.shape != mask.shape:
        raise ValueError("predicted, measured and mask must have identical shapes")
    return np.where(mask, measured, predicted)

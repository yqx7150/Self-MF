"""Numerical checks for acquired-measurement enforcement."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "selfmf"))
from mri_ops import enforce_acquired_samples

class AcquiredSamplesTest(unittest.TestCase):
    def test_acquired_samples_unchanged(self):
        rng = np.random.default_rng(42)
        measured = rng.normal(size=(32, 32)) + 1j*rng.normal(size=(32,32))
        estimate = rng.normal(size=(32, 32)) + 1j*rng.normal(size=(32,32))
        mask = rng.random((32,32)) < 0.35
        reconstructed = enforce_acquired_samples(estimate, measured, mask)
        np.testing.assert_array_equal(reconstructed[mask], measured[mask])
        np.testing.assert_array_equal(reconstructed[~mask], estimate[~mask])

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            enforce_acquired_samples(np.ones((16, 16)), np.ones((8,8)), np.ones((16,16)))

if __name__ == "__main__":
    unittest.main()

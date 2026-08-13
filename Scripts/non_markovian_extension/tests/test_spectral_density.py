import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.observables import trace_distance_qubit
from src.spectral_density import structured_bosonic_spectral_density


class NonMarkovianUtilityTests(unittest.TestCase):
    def test_spectral_density_is_zero_at_negative_frequency(self):
        value = structured_bosonic_spectral_density(-1.0, 0.1, 1.0, 5.0)
        self.assertAlmostEqual(float(value), 0.0)

    def test_ohmic_density_is_positive_for_positive_frequency(self):
        value = structured_bosonic_spectral_density(1.0, 0.1, 1.0, 5.0)
        self.assertGreater(float(value), 0.0)

    def test_trace_distance_identical_states_is_zero(self):
        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        self.assertAlmostEqual(trace_distance_qubit(rho, rho), 0.0)


if __name__ == "__main__":
    unittest.main()


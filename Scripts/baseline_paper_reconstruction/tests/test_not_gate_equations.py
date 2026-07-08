import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parameters import NotGateParameters
from src.logic_gates import (
    bounded_neuron_response,
    generate_xor_surface,
    majority_virtual_temperature,
    nor_virtual_temperature,
)
from src.not_gate import (
    bounded_not_response,
    collector_current,
    modulator_current,
    virtual_temperature,
)
from src.thermal_functions import fermi_occupation, inverse_fermi_occupation
from src.virtual_qubit import virtual_temperature_two_qubit


class NotGateEquationTests(unittest.TestCase):
    def test_fermi_inverse_round_trip(self):
        beta = 1.7
        epsilon = 0.4
        probability = fermi_occupation(beta, epsilon)
        self.assertAlmostEqual(
            inverse_fermi_occupation(probability, epsilon),
            beta,
            places=12,
        )

    def test_virtual_temperature_matches_eq_16(self):
        beta_v = virtual_temperature(beta0=1.5, beta1=1.0, epsilon1=2.0, epsilon_z=0.1)
        expected = ((2.0 + 0.1) / 0.1) * 1.5 - (2.0 / 0.1) * 1.0
        self.assertAlmostEqual(beta_v, expected, places=12)

    def test_collector_current_zero_at_virtual_temperature(self):
        current = collector_current(beta_z=1.25, beta_v=1.25, epsilon_z=0.1, mu=1.0e-4)
        self.assertAlmostEqual(current, 0.0, places=18)

    def test_modulator_current_zero_at_reference_temperature(self):
        current = modulator_current(beta_z=2.0, beta_r=2.0, epsilon_z=0.1, mu_prime=1.0e-4)
        self.assertAlmostEqual(current, 0.0, places=18)

    def test_bounded_not_response_stays_in_logic_range(self):
        params = NotGateParameters(beta_hot=1.0, beta_cold=2.0, epsilon_z=0.1)
        hot_output = bounded_not_response(beta_v=100.0, params=params)
        cold_output = bounded_not_response(beta_v=-100.0, params=params)

        self.assertGreaterEqual(hot_output, params.beta_hot)
        self.assertLessEqual(hot_output, params.beta_cold)
        self.assertGreaterEqual(cold_output, params.beta_hot)
        self.assertLessEqual(cold_output, params.beta_cold)

    def test_two_qubit_virtual_temperature_matches_eq_11(self):
        beta_v = virtual_temperature_two_qubit(
            beta0=1.5,
            beta1=2.0,
            epsilon0=2.0,
            epsilon1=1.5,
        )
        expected = (2.0 / 0.5) * 1.5 - (1.5 / 0.5) * 2.0
        self.assertAlmostEqual(beta_v, expected, places=12)

    def test_nor_hyperplane_classifies_truth_table(self):
        epsilon_z = 0.1
        alpha = 8.0
        beta_hot = 0.0
        beta_cold = 1.0
        cases = [
            (beta_hot, beta_hot, True),
            (beta_hot, beta_cold, False),
            (beta_cold, beta_hot, False),
            (beta_cold, beta_cold, False),
        ]
        for beta1, beta2, should_be_high in cases:
            beta_v = nor_virtual_temperature(beta1, beta2, epsilon_z, alpha)
            beta_out = bounded_neuron_response(beta_v, beta_hot, beta_cold, epsilon_z)
            if should_be_high:
                self.assertGreater(beta_out, 0.5)
            else:
                self.assertLess(beta_out, 0.5)

    def test_majority_hyperplane_classifies_truth_table(self):
        epsilon_z = 0.1
        alpha = 8.0
        beta_hot = 0.0
        beta_cold = 1.0
        cases = [
            (0.0, 0.0, 0.0, True),
            (1.0, 0.0, 0.0, True),
            (1.0, 1.0, 0.0, False),
            (1.0, 1.0, 1.0, False),
        ]
        for beta1, beta2, beta3, should_be_high in cases:
            beta_v = majority_virtual_temperature(beta1, beta2, beta3, epsilon_z, alpha)
            beta_out = bounded_neuron_response(beta_v, beta_hot, beta_cold, epsilon_z)
            if should_be_high:
                self.assertGreater(beta_out, 0.5)
            else:
                self.assertLess(beta_out, 0.5)

    def test_xor_network_classifies_truth_table(self):
        class Params:
            beta_hot = 0.0
            beta_cold = 1.0
            epsilon_z = 0.1

        import numpy as np

        beta1, beta2 = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        rows = generate_xor_surface(beta1, beta2, Params(), alpha=8.0)
        observed = {
            (row["beta1"], row["beta2"]): row["xor_beta_z"] > 0.5
            for row in rows
        }
        self.assertFalse(observed[(0.0, 0.0)])
        self.assertTrue(observed[(1.0, 0.0)])
        self.assertTrue(observed[(0.0, 1.0)])
        self.assertFalse(observed[(1.0, 1.0)])


if __name__ == "__main__":
    unittest.main()

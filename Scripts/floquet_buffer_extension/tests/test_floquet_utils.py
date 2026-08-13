import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thermal_rates import qubit_excited_population, thermal_jump_rates


class FloquetUtilityTests(unittest.TestCase):
    def test_excited_population_is_bounded(self):
        value = qubit_excited_population(beta=1.0, epsilon=1.0)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)

    def test_jump_rates_are_positive(self):
        down, up = thermal_jump_rates(beta=1.0, epsilon=1.0, gamma=0.1)
        self.assertGreater(down, 0.0)
        self.assertGreater(up, 0.0)
        self.assertGreater(down, up)


if __name__ == "__main__":
    unittest.main()


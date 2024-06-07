import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from measures.measures import Dirac, Id, Measure, OneMeasure
from models.selection_mutation_model import SelectionMutation


def get_measures(interval=[0, 1]):
    m = []
    m0 = Measure(lambda x: x + 1, interval)
    m.append(m0)
    m1 = Measure(lambda x: 3 * x, interval)
    m.append(m1)
    m2 = Measure(lambda x: np.exp(x) + 1, interval)
    m.append(m2)
    m3 = Measure(lambda x: 3 * x + np.log(x + 1), interval)
    m.append(m3)

    return m


class TestMeasures(unittest.TestCase):
    def testAddition(self):
        m = get_measures(interval=[0, 1])
        self.assertEqual(
            m[0] + m[1],
            Measure(lambda x: 4 * x + 1),
            "(x+1) + (3*x) must be equals to (4*x + 1)",
        )

        self.assertEqual(
            m[2] + m[3],
            Measure(lambda x: 3 * x + 1 + np.exp(x) + np.log(x + 1)),
            "(e^x+1) + (3*x + log(x+1)) must be equals to (e^x + log(x+1) + 3*x + 1)",
        )

    def testSubstraction(self):
        m = get_measures([0, 1])
        self.assertEqual(
            m[0] - m[1],
            Measure(lambda x: 1 - 2 * x),
            "(x+1) + (3*x) must be equals to (1 - 2*x)",
        )

        self.assertEqual(
            m[2] - m[3],
            Measure(lambda x: -3 * x + 1 + np.exp(x) - np.log(x + 1)),
            "(e^x+1) + (3*x + log(x+1)) must be equals to (e^x - log(x+1) - 3*x + 1)",
        )

    def testMultiplication(self):
        m = get_measures([0, 1])
        self.assertEqual(
            m[0] * m[1],
            Measure(lambda x: 3 * x**2 + 3 * x),
            "(x+1) * (3*x) must be equals to (3*x^2 + 3*x)",
        )

        self.assertEqual(
            m[2] * m[3],
            Measure(
                lambda x: 3 * x * np.exp(x)
                + 3 * x
                + np.exp(x) * np.log(x + 1)
                + np.log(x + 1)
            ),
            "(e^x+1) * (3*x + log(x+1)) must be equals to (3*x*e^x + log(x+1)*e^x + 3*x + log(x+1))",
        )

    def testDivision(self):
        m = get_measures([0, 1])
        self.assertEqual(
            m[1] / m[0],
            Measure(lambda x: (3 * x) / (x + 1)),
            "(3*x) / (x+1)  must be equals to (3*x) / (x+1)",
        )

        self.assertEqual(
            m[3] / m[2],
            Measure(lambda x: (3 * x + np.log(x + 1)) / (np.exp(x) + 1)),
            "(3*x + log(x+1)) / (e^x+1)  must be equals to (3*x + log(x+1)) / (e^x+1)",
        )

    def testDirac(self):
        self.assertEqual(Dirac(0.5)(0), 0, "Dirac0.5 must be quals to 0 in 0")
        self.assertEqual(Dirac(0.5)(0.5), 1, "Dirac0.5 must be quals to 1 in 0.5")
        self.assertEqual(Dirac(0.5)(1), 0, "Dirac0.5 must be quals to 0 in 1")

        m = Measure(lambda x: x) * (Dirac(0) + Dirac(0.5) + Dirac(1))
        self.assertEqual(
            m(0), 0, "x*(Dirac0 + Dirac0.5 + Dirac1) must be quals to 0 in 0"
        )
        self.assertEqual(
            m(0.5), 0.5, "x*(Dirac0 + Dirac0.5 + Dirac1) must be quals to 0.5 in 0.5"
        )
        self.assertEqual(
            m(1), 1, "x*(Dirac0 + Dirac0.5 + Dirac1) must be quals to 1 in 1"
        )
        self.assertEqual(
            m(0.75), 0, "x*(Dirac0 + Dirac0.5 + Dirac1) must be quals to 0 in 0.75"
        )


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
import unittest
import pynumpy as pnp
from .helper import skip_if_no_numpy, np

def _clean_repr(a):
    return "".join(repr(a).split())

@skip_if_no_numpy
class TestPyNDArrayCompat(unittest.TestCase):
    def setUp(self):
        a = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
        self.t0 = pnp.array(a)
        self.n0 = np.array(a)

    def test_repr(self):
        self.assertEqual(_clean_repr(self.t0), _clean_repr(self.n0))

    def test_index(self):
        self.assertEqual(float(self.t0[1, 1]), float(self.n0[1, 1]))

    def test_slice(self):
        self.assertEqual(_clean_repr(self.t0[1:]), _clean_repr(self.n0[1:]))
        self.assertEqual(
            _clean_repr(self.t0[1:, 1:]), _clean_repr(self.n0[1:, 1:])
        )
        self.assertEqual(
            _clean_repr(self.t0[-1:,]), _clean_repr(self.n0[-1:,])
        )
        self.assertEqual(
            _clean_repr(self.t0[-1:, 1]), _clean_repr(self.n0[-1:, 1])
        )

    def test_double_slice(self):
        self.assertEqual(
            _clean_repr(self.t0[1:][2::2]), _clean_repr(self.n0[1:][2::2])
        )

    def test_len(self):
        self.assertEqual(len(self.t0), len(self.n0))

    def test_newaxis(self):
        self.assertEqual(self.t0[pnp.newaxis, 2:].shape, (1, 2, 4))
        self.assertEqual(self.n0[np.newaxis, 2:].shape, (1, 2, 4))
        self.assertEqual(
            _clean_repr(self.t0[pnp.newaxis, 2:]),
            _clean_repr(self.n0[np.newaxis, 2:]),
        )

@skip_if_no_numpy
class TestNDIterCompat(unittest.TestCase):
    def setUp(self):
        self.t0 = pnp.array([[1.0, 2.0], [3.0, 4.0]])
        self.n0 = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_basic(self):
        self.assertEqual(
            [float(i) for i in pnp.nditer(self.t0)],
            [float(i) for i in np.nditer(self.n0)],
        )

@skip_if_no_numpy
class TestFunctionsCompat(unittest.TestCase):
    def setUp(self):
        a1 = [[1, 2], [3, 4]]
        self.t1 = pnp.array(a1)
        self.n1 = np.array(a1)

    def test_eye(self):
        self.assertEqual(_clean_repr(pnp.eye(3)), _clean_repr(np.eye(3)))

    def test_var(self):
        self.assertAlmostEqual(float(self.t1.var()), float(self.n1.var()))

    def test_std(self):
        self.assertAlmostEqual(float(self.t1.std()), float(self.n1.std()))

if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
import unittest
import pynumpy as pnp

def _clean_repr(a):
    return "".join(repr(a).split())

class TestFunctions(unittest.TestCase):
    def setUp(self):
        a0 = [
            [[1, [1, 1, 1, 1], [2, 2], [2, 2]], 1],
            [2, 2, 1, 1],
            [3, 3],
            [3, 3],
        ]
        self.t0 = pnp.array(a0)

        a1 = [[1, 2], [3, 4]]
        self.t1 = pnp.array(a1)

    def test_array(self):
        self.assertEqual(self.t0.shape, (4, 4, 4, 4))

    def test_view_operator_safety(self):
        # Create a base array
        a = pnp.zeros((10,), dtype="int32")
        # Create a view
        v = a[2:5]
        # Perform in-place addition on the view
        v += 1
        
        # Verify that ONLY the view's elements were modified
        expected = pnp.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype="int32")
        self.assertTrue((a == expected).all())

    def test_rsub_scalar(self):
        a = pnp.array([1, 2, 3])
        b = 10 - a
        self.assertTrue((b == pnp.array([9, 8, 7])).all())

    def test_operator_type_error(self):
        a = pnp.array([1, 2, 3])
        with self.assertRaises(TypeError):
            # This should raise TypeError because we return NotImplemented
            # and strings don't know how to subtract arrays.
            a - "string"

    def test_shape_mismatch_error(self):
        a = pnp.array([1, 2, 3])
        b = pnp.array([1, 2])
        with self.assertRaises(ValueError):
            a + b

    def test_sum(self):
        self.assertEqual(self.t1.sum(), 10)

    def test_mean(self):
        self.assertEqual(pnp.array([-5, 5]).mean(), 0)

    def test_argmax(self):
        self.assertEqual(pnp.array([1, 2, 3]).argmax(), 2)

    def test_argmin(self):
        self.assertEqual(pnp.array([1, 2, -3]).argmin(), 2)

    def test_cumsum(self):
        self.assertEqual(
            _clean_repr(pnp.array([1, 2, 3]).cumsum()),
            _clean_repr(pnp.array([1, 3, 6])),
        )

    def test_cumprod(self):
        self.assertEqual(
            _clean_repr(pnp.array([1, 2, 3]).cumprod()),
            _clean_repr(pnp.array([1, 2, 6])),
        )

    def test_all(self):
        self.assertEqual(pnp.array([1, 0, 1]).all(), False)

    def test_any(self):
        self.assertEqual(pnp.array([1, 0, 1]).any(), True)

    def test_sum_negative(self):
        self.assertEqual(pnp.array([1, 2, -3]).sum(), 0)

    def test_max(self):
        self.assertEqual(self.t1.max(), 4)

    def test_min(self):
        self.assertEqual(self.t1.min(), 1)

    def test_prod(self):
        self.assertEqual(pnp.array([1, 2, 3, 10]).prod(), 60)

    def test_ptp(self):
        self.assertEqual(pnp.array([-1, 2, 3]).ptp(), 4)

    def test_fill(self):
        t = pnp.zeros((4, 4, 4))
        t.fill(1)
        self.assertEqual(t.sum(), 4 * 4 * 4)

    def test_copy(self):
        t = self.t1.copy()
        t[0, 0] = 0
        self.assertEqual(self.t1.sum(), 10)
        self.assertEqual(t.sum(), 9)

    def test_flatten(self):
        self.assertEqual(
            _clean_repr(self.t1.flatten()),
            _clean_repr(pnp.array([1, 2, 3, 4])),
        )

    def test_var_basic(self):
        self.assertEqual(pnp.array([1, 1, 1, 1]).var(), 0)

    def test_std_basic(self):
        self.assertEqual(pnp.array([1, 1, 1, 1]).std(), 0)

    def test_reshape_variadic(self):
        a = pnp.array([1, 2, 3, 4])
        # Test variadic args
        b = a.reshape(2, 2)
        self.assertEqual(b.shape, (2, 2))
        # Test tuple arg
        c = a.reshape((2, 2))
        self.assertEqual(c.shape, (2, 2))
        # Test top-level reshape variadic
        d = pnp.reshape(a, 2, 2)
        self.assertEqual(d.shape, (2, 2))
        
    def test_linalg_alias(self):
        self.assertTrue(hasattr(pnp, 'linalg'))
        self.assertTrue(hasattr(pnp.linalg, 'det'))

if __name__ == "__main__":
    unittest.main()

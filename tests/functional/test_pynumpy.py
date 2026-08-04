# -*- coding: utf-8 -*-
import unittest
import pynumpy as pnp

class TestPyNumPyInternal(unittest.TestCase):
    def test_dtype_exposure(self):
        # Verify that common dtypes are available at the top level
        self.assertEqual(pnp.float64, "float64")
        self.assertEqual(pnp.float32, "float32")
        self.assertEqual(pnp.int64, "int64")
        self.assertEqual(pnp.int32, "int32")
        self.assertEqual(pnp.bool, "bool")

    def test_creating_functions(self):
        b1 = pnp.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(b1.shape, (2, 3))

    def test_cross(self):
        x = [1, 2, 3]
        y = [4, 5, 6]
        z = pnp.cross(x, y)
        self.assertEqual(z, [-3, 6, -3])

    def test_2dim_cross(self):
        x = [1, 2]
        y = [4, 5]
        z = pnp.cross(x, y)
        self.assertEqual(z, [-3])

    def test_dot(self):
        x = [1, 2, 3]
        y = [4, 5, 6]
        z = pnp.dot(x, y)
        self.assertEqual(z, 32)

    def test_det(self):
        x = [5, -2, 1]
        y = [0, 3, -1]
        z = [2, 0, 7]
        mat = [x, y, z]
        a = pnp.linalg.det(mat)
        self.assertEqual(a, 103)

    def test_add(self):
        x = [5, -2, 1]
        y = [0, 3, -1]
        a = pnp.add(x, y)
        self.assertTrue((a == pnp.array([5, 1, 0], dtype="int64")).all())

    def test_subtract(self):
        x = [5, -2, 1]
        y = [0, 3, -1]
        a = pnp.subtract(x, y)
        self.assertTrue((a == pnp.array([5, -5, 2], dtype="int64")).all())

    def test_divide(self):
        x = [15, -12, 3]
        y = 3
        a = pnp.divide(x, y)
        self.assertTrue((a == pnp.array([5, -4, 1], dtype="int64")).all())

    def test_multiply(self):
        x = [5, -2, 1]
        y = [0, 3, -1]
        a = pnp.multiply(x, y)
        self.assertTrue((a == pnp.array([0, -6, -1], dtype="int64")).all())

if __name__ == "__main__":
    unittest.main()

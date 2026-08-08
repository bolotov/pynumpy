# -*- coding: utf-8 -*-
import unittest
import pynumpy as pnp
from .helper import skip_if_no_numpy, np


@skip_if_no_numpy
class TestPyNumPyCompat(unittest.TestCase):
    def test_shapes_and_strides(self):
        for shape in [
            (9,),
            (109,),
            (9, 4),
            (109, 104),
            (9, 4, 5),
            (109, 104, 105),
        ]:
            # Test shape and strides
            a = np.empty(shape)
            b = pnp.empty(shape)
            self.assertEqual(a.ndim, b.ndim)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.strides, b.strides)
            self.assertEqual(a.size, b.size)

    def test_dtype(self):
        for shape in [(9,), (9, 4), (9, 4, 5)]:
            for dtype in [
                "bool",
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "float32",
                "float64",
            ]:
                a = np.empty(shape, dtype=dtype)
                b = pnp.empty(shape, dtype=dtype)
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(a.itemsize, b.itemsize)

    def test_reshape(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="int32")
        b = pnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="int32")

        for shape in [(2, 4), (4, 2), (2, 2, 2), (8,)]:
            a = a.reshape(shape)
            b.shape = shape
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.strides, b.strides)

    def test_from_and_to_numpy(self):
        for dtype in [
            "float32",
            "float64",
            "int32",
            "uint32",
            "uint8",
            "int8",
        ]:
            for data in [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [[1, 2], [3, 4], [5, 6], [7, 8]],
            ]:
                a1 = np.array(data, dtype)
                b1 = pnp.array(a1)
                self.assertEqual(a1.shape, b1.shape)
                self.assertEqual(a1.dtype, b1.dtype)
                self.assertTrue((a1 == b1).all())


if __name__ == "__main__":
    unittest.main()

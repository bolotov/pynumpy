# -*- coding: utf-8 -*-
import unittest
import pynumpy as pnp
from .helper import (
    skip_if_no_numpy, skip_if_no_hypothesis, np, hypothesis, HAS_HYPOTHESIS
)

if HAS_HYPOTHESIS:
    from hypothesis import given, strategies as st


@skip_if_no_numpy
@skip_if_no_hypothesis
class TestLinAlgCompat(unittest.TestCase):
    @given(
        st.lists(
            st.lists(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
                min_size=2,
                max_size=2,
            ),
            min_size=2,
            max_size=2,
        )
    )
    def test_det_2x2(self, data):
        a = np.array(data)
        b = pnp.array(data)
        
        det_a = np.linalg.det(a)
        det_b = pnp.linalg.det(b)
        
        # For large values, absolute difference is not a good metric.
        # We use a relative tolerance check.
        if det_a == 0:
            self.assertAlmostEqual(det_a, det_b, places=5)
        else:
            rel_diff = abs(det_a - det_b) / abs(det_a)
            self.assertLess(rel_diff, 1e-12)

    @given(
        st.lists(
            st.lists(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
                min_size=3,
                max_size=3,
            ),
            min_size=3,
            max_size=3,
        )
    )
    def test_det_3x3(self, data):
        a = np.array(data)
        b = pnp.array(data)
        
        det_a = np.linalg.det(a)
        det_b = pnp.linalg.det(b)
        
        if det_a == 0:
            self.assertAlmostEqual(det_a, det_b, places=5)
        else:
            rel_diff = abs(det_a - det_b) / abs(det_a)
            self.assertLess(rel_diff, 1e-12)


if __name__ == "__main__":
    unittest.main()

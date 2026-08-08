import unittest

try:
    import numpy as np

    # Attempt a small operation to catch Bus Error or other runtime import failures
    _tmp = np.array([1, 2, 3])
    HAS_NUMPY = True
except (ImportError, Exception):
    HAS_NUMPY = False
    np = None

try:
    import hypothesis
    HAS_HYPOTHESIS = True
except (ImportError, Exception):
    HAS_HYPOTHESIS = False
    hypothesis = None

skip_if_no_numpy = unittest.skipIf(
    not HAS_NUMPY, "numpy is not available or is broken"
)

skip_if_no_hypothesis = unittest.skipIf(
    not HAS_HYPOTHESIS, "hypothesis is not available"
)

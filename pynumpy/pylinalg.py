# Copyright (c) 2014 Eric Allen Youngson
# pynumpy is distributed under the terms of the MIT License.

"""This module is used to implement native Python functions to replace those
called from numpy, when not available"""

from pynumpy.pynumpy import ndarray


# Written by Eric Youngson eric@scneco.com / eayoungs@gmail.com
# Succession Ecological Services: Portland, Oregon


class LinAlgError(Exception):
    pass


def det(A):
    # type: (ndarray) -> float
    """
    Compute the determinant of an array.
    """
    n = len(A)
    if n == 2 and all(len(vec) == 2 for vec in A):
        # http://mathworld.wolfram.com/Determinant.html
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    elif n == 3 and all(len(vec) == 3 for vec in A):
        try:
            # http://mathworld.wolfram.com/Determinant.html
            det_val = (
                A[0][0] * A[1][1] * A[2][2]
                + A[0][1] * A[1][2] * A[2][0]
                + A[0][2] * A[1][0] * A[2][1]
                - (
                    A[0][2] * A[1][1] * A[2][0]
                    + A[0][1] * A[1][0] * A[2][2]
                    + A[0][0] * A[1][2] * A[2][1]
                )
            )
            return float(det_val)
        except Exception:
            raise LinAlgError("Error computing 3x3 determinant")
    else:
        # todo: implement general LU-based determinant
        raise IndexError("Vector has invalid dimensions or not implemented")

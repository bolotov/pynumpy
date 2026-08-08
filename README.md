pynumpy
=======

[INFO] A lightweight, pure Python, NumPy-compliant ndarray library.

`pynumpy` is a fork of **tinynumpy** (originally created by Wade Brainerd). It is designed to provide a subset of NumPy functionality in environments where C extensions are unavailable or undesirable (e.g., PyPy, Jython, or constrained systems).

History and Lineage
-------------------

This project was forked in 2026 to continue development and maintenance.
* **Original Project:** [tinynumpy](https://github.com/wadetb/tinynumpy)
* **Current Maintainer:** Oleksandr Bolotov

Links
-----

* **GitHub:** [https://github.com/bolotov/pynumpy](https://github.com/bolotov/pynumpy)
* **Documentation:** [https://pynumpy.readthedocs.io/](https://pynumpy.readthedocs.io/)


Features
--------

* The ndarray class has all the same properties as the numpy ndarray
  class.
* Pretty good compliance with numpy in terms of behavior (such as views).
* Can be converted to a numpy array (with shared memory).
* Can get views of real numpy arrays (with shared memory).
* Support for wrapping ctypes arrays, or provide ctypes pointer to data.
* Pretty fast for being pure Python.
* Works on Python 2.5+, Python 3.x, Pypy and Jython.

Caveats
-------

* ndarray.flat iterator cannot be indexed (it is a generator).
* No support for Fortran order.
* Support for data types limited to bool, uin8, uint16, uint32, uint64,
  int8, int16, int32, int64, float32, float64.
* Functions that calculate statistics on the data are much slower, since
  the iteration takes place in Python.
* Assigning via slicing is usually pretty fast, but can be slow if the
  striding is unfortunate.


Examples
--------

```python
>>> from pynumpy import pynumpy as pnp
>>> a = pnp.array([[1, 2, 3, 4],[5, 6, 7, 8]])

>>> a
array([[ 1.,  2.,  3.,  4.],
    [ 5.,  6.,  7.,  8.]], dtype='float64')

>>> a[:, 2:]
array([[ 3.,  4.],
    [ 7.,  8.]], dtype='float64')

>>> a[:, ::2]
array([[ 1.,  3.],
    [ 5.,  7.]], dtype='float64')

>>> a.shape
(2, 4)

>>> a.shape = 4, 2

>>> a
array([[ 1.,  2.],
    [ 3.,  4.],
    [ 5.,  6.],
    [ 7.,  8.]], dtype='float64')

>>> b = a.ravel()

>>> a[0, 0] = 100

>>> b
array([ 100.,  2.,  3.,  4.,  5.,  6.,  7.,  8.], dtype='float64')
```

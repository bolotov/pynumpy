
Do not expect much performance because actual
NumPy useds BLAS/LAPACK libraries
(such as OpenBLAS, MKL, or Apple Accelerate).

These libraries use standard O(n^3) algorithms but also
hardware-level loop unrolling, cache blocking, and
SIMD vectorization which is all C stuff which is
not present in this C-free version.


Other somewhat related things:

- Free Book on NumPy by Wes McKinney:
  https://wesmckinney.com/book/

- NumPy-like json serialization
  https://pypi.org/project/numpy2/

- NumPy-like thingie for micropython, which USES C
  unlike this library
  https://github.com/v923z/micropython-ulab

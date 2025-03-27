"""
$ export MKL_NUM_THREADS=12
$ export NUMEXPR_NUM_THREADS=12
$ export OMP_NUM_THREADS=12
$ numactl --cpunodebind=0 --localalloc python linalg_solve.py
Computation with CPU
[ 0.03040997  0.01440329  0.06772684 ...  0.00026573 -0.0293215
  0.01875962]
computation time: 0.94 sec.
$ python linalg_solve.py --gpu
Computation with CUDA GPU
[ 0.0675002   0.03450186 -0.03794135 ...  0.01511969 -0.01006322
 -0.0252226 ]
computation time: 0.01 sec.
"""

import argparse
import time
def main(use_gpu: bool) -> None:
    if use_gpu:
        print("Computation with CUDA GPU")
        import cupy as np
    else:
        print("Computation with CPU")
        import numpy as np
    n = 8000
    A = np.random.uniform(size=(n, n)).astype(np.float64)  # $BA4MWAG(B [0, 1) $B$N(Bn$B<!@5J}9TNs(B
    b = np.ones(n, dtype=np.float64)                       # $BA4MWAG(B 1 $B$N4{CN%Y%/%H%k(B
    """
    warmup: $B=i4|2=$J$I$N%*!<%P!<%X%C%I$r7WB,$+$i=|30(B
    CuPy$B$O:G=i$N%+!<%M%k<B9T$G(BCUDA$B%3!<%I$N:GE,2=$H$$$C$?@-G=$K4X$9$k=i4|@_Dj$,Av$k$h$&$G$9(B
    """
    x = np.linalg.solve(A, b)  # https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    tbeg = time.clock_gettime(time.CLOCK_MONOTONIC)
    x = np.linalg.solve(A, b)  # Ax = b $B$r2r$/!"7W;;$K$O(BLAPACK$B$N(B?gesv$B$,;H$o$l$k(B
    tend = time.clock_gettime(time.CLOCK_MONOTONIC)
    print(x)
    print(f"computation time: {tend - tbeg:.2f} sec.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", action="store_true", default=False, dest="use_gpu")
    args = parser.parse_args()
    main(use_gpu=args.use_gpu)

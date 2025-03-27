"""
ライブラリのインストール
$ sudo yum install python2-pip
$ sudo yum install python3-pip
$ echo "alias python='python3'" >> ~/.bash_profile
$ python -m pip install numpy
$ python -m pip install cupy-cuda11x
$ python -m cupyx.tools.install_library --cuda 11.x --library cutensor

環境変数
$ export MKL_NUM_THREADS=12
$ export NUMEXPR_NUM_THREADS=12
$ export OMP_NUM_THREADS=12
$ python linalg_solve.py

実行 CPU
Computation with CPU
[-0.01070097  0.03764457  0.01875006 ... -0.02791825  0.01571928 -0.0341835 ]
computation time: 4.11 sec.

実行GPU
$ python linalg_solve.py -g
Computation with CUDA GPU
[-0.00837371  0.01567632  0.01950274 ...  0.00291613  0.01032924 -0.01412289]
computation time: 1.53 sec.

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
    A = np.random.uniform(size=(n, n)).astype(np.float64)  # 全要素 [0, 1) のn次正方行列
    b = np.ones(n, dtype=np.float64)                       # 全要素 1 の既知ベクトル
    """
    warmup: 初期化などのオーバーヘッドを計測から除外
    CuPyは最初のカーネル実行でCUDAコードの最適化といった性能に関する初期設定が走るようです
    """
    x = np.linalg.solve(A, b)  # https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    tbeg = time.clock_gettime(time.CLOCK_MONOTONIC)
    x = np.linalg.solve(A, b)  # Ax = b を解く、計算にはLAPACKの?gesvが使われる
    tend = time.clock_gettime(time.CLOCK_MONOTONIC)
    print(x)
    print(f"computation time: {tend - tbeg:.2f} sec.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", action="store_true", default=False, dest="use_gpu")
    args = parser.parse_args()
    main(use_gpu=args.use_gpu)

# https://docs.exaloop.io/parallel/gpu/
#
# GPU kernel版（できるだけ元コードの形を維持）
#

import numpy as np
import gpu

MAX = 1000
N = 4096

# 元は pixels=np.empty((N,N),dtype=int) でしたが、
# GPUからNumPy配列へ直接書くと落ちやすいので、
# まずは1次元バッファに書いて、最後にNumPyへ整形します。
pixels_buf = [0 for _ in range(N * N)]

def write_pgm_p2(path: str, img: np.ndarray):
    h, w = img.shape
    with open(path, "w") as f:
        f.write("P2\n")
        f.write(str(w)); f.write(" "); f.write(str(h)); f.write("\n")
        f.write("255\n")

        for i in range(h):
            chunk = ""
            row = img[i]
            for j in range(w):
                chunk += str(int(row[j]))
                chunk += " "
                if len(chunk) >= 8192:
                    f.write(chunk)
                    chunk = ""
            if chunk:
                f.write(chunk)
            f.write("\n")

def scale(x, a, b):
    # 元の形を維持（ただし float 除算を明示）
    return a + (float(x) / float(N)) * (b - a)

@gpu.kernel
def execKernel(pixels_buf, N: int, MAX: int) -> None:
    idx = gpu.block.x * gpu.block.dim.x + gpu.thread.x
    if idx >= N * N:
        return

    i = idx // N
    j = idx - i * N

    # 元: c=complex(scale(j,...), scale(i,...))
    # GPUでは complex/abs が不安定なので、実数2本で等価に計算
    cr = -2.00 + (float(j) / float(N)) * (0.47 - (-2.00))
    ci = -1.12 + (float(i) / float(N)) * (1.12 - (-1.12))

    zr = 0.0
    zi = 0.0
    iteration = 0

    # 元: while abs(z)<=2 and iteration<MAX:
    # abs(z)<=2  <=>  zr^2+zi^2 <= 4
    while (zr*zr + zi*zi) <= 4.0 and iteration < MAX:
        # 元: z=z**2+c を実数で
        zr2 = zr*zr - zi*zi + cr
        zi2 = 2.0*zr*zi + ci
        zr = zr2
        zi = zi2
        iteration += 1

    pixels_buf[idx] = (255 * iteration) // MAX

block = 256
grid = (N * N + block - 1) // block
execKernel(gpu.raw(pixels_buf), N, MAX, grid=grid, block=block)

# 元の pixels=np.empty((N,N),dtype=int) に戻す（CPU側で整形）
pixels = np.array(pixels_buf, dtype=np.uint8).reshape((N, N))

print(pixels[0, :64])                 # 左上は0になりがち
print(int(pixels.min()), int(pixels.max()))
mid = N // 2
print(pixels[mid, mid-10:mid+10])     # 中心付近の確認

write_pgm_p2("mandel_4096_p2.pgm", pixels)
print("wrote mandel_4096_p2.pgm")

# MAX=1000  # maximum Mandelbrot iterations
# N=4096  # width and height of image
# pixels=np.empty((N,N),dtype=int)
# 
# def scale(x,a,b):
#     return a+(x/N)*(b-a)
# 
# @par(gpu=True,collapse=2)
# for i in range(N):
#     for j in range(N):
#         c=complex(scale(j,-2.00,0.47),scale(i,-1.12,1.12))
#         z=0j
#         iteration=0
# 
#         while abs(z)<=2 and iteration<MAX:
#             z=z**2+c
#             iteration+=1
# 
#         pixels[i,j]=int(255*iteration/MAX)








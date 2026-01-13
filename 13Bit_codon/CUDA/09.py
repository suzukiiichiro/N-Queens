# t_min_int.py
import gpu
x = [0] * 32

@gpu.kernel
def k(x, n: int):
  gid = gpu.block.x * gpu.block.dim.x + gpu.thread.x
  if gid < n:
    x[gid] = gid + 100

k(x, 32, grid=1, block=32)
print(x)



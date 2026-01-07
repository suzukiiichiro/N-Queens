# https://docs.exaloop.io/parallel/gpu/
# suzuki@cudacodon$ codon run 05.py
# [0, 1, 1.41421, 1.73205, 2, 2.23607, 2.44949, 2.64575, 2.82843, 3]

import math
import gpu

x=[float(i) for i in range(10)]

@gpu.kernel
def hello(x):
  i=gpu.thread.x
  x[i]=math.sqrt(x[i])  # uses __nv_sqrt from libdevice

hello(x,grid=1,block=10)
print(x)


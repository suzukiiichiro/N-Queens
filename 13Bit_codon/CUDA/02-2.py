# https://docs.exaloop.io/parallel/gpu/
#
# Segmentation fault
#
# suzuki@cudacodon$ codon run 02-2.py
# [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
# suzuki@cudacodon$

import gpu

a=[i for i in range(16)]
b=[2*i for i in range(16)]
c=[0 for _ in range(16)]

@gpu.kernel
def execKernel(a,b,c)->None:
  i=gpu.thread.x
  c[i]=a[i]+b[i]

execKernel(gpu.raw(a),gpu.raw(b),gpu.raw(c),grid=1,block=16)
print(c)

# a=[i for i in range(16)]
# b=[2*i for i in range(16)]
# c=[0 for _ in range(16)]
#
# @par(gpu=True)
# for i in range(16):
#     c[i]=a[i]+b[i]


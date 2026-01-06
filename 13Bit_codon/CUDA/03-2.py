#
# Segmentation fault
#
# suzuki@cudacodon$ codon run 03-2.py
# 16 array of int
# lqqqqqwqqqqqwqqqqqwqqqqqwqqqqqwqqqqqwqqqqqk
# x   0 x   3 x   6 x ... x  39 x  42 x  45 x
# mqqqqqvqqqqqvqqqqqvqqqqqvqqqqqvqqqqqvqqqqqj
# suzuki@cudacodon$

import numpy as np
import gpu

a=np.arange(16)
b=np.arange(16)*2
c=np.empty(16,dtype=int)

@gpu.kernel
def execKernel(a,b,c)->None:
  i=gpu.thread.x
  c[i]=a[i]+b[i]

execKernel(gpu.raw(a),gpu.raw(b),gpu.raw(c),grid=1,block=16)
print(c)

# a=np.arange(16)
# b=np.arange(16)*2
# c=np.empty(16,dtype=int)
# 
# @par(gpu=True)
# for i in range(16):
#     c[i]=a[i]+b[i]
# 

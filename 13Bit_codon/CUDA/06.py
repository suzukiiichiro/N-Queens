# suzuki@cudacodon$ codon run 06.py
# [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]

import gpu

a=[i for i in range(16)]
b=[2*i for i in range(16)]
c=[0 for _ in range(16)]

@gpu.kernel
def hello(a,b,c):
  i=gpu.thread.x
  c[i]=a[i]+b[i]

# call the kernel with three int-pointer arguments:
hello(gpu.raw(a),gpu.raw(b),gpu.raw(c),grid=1,block=16)
print(c)  # output same as first snippet's


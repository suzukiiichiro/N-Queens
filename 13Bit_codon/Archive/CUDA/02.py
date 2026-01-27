# https://docs.exaloop.io/parallel/gpu/
#
# Segmentation fault
#

import gpu

a=[i for i in range(16)]
b=[2*i for i in range(16)]
c=[0 for _ in range(16)]

@par(gpu=True)
for i in range(16):
    c[i]=a[i]+b[i]


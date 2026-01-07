# https://docs.exaloop.io/parallel/gpu/
#
# Segmentation fault
#

import numpy as np

a=np.arange(16)
b=np.arange(16)*2
c=np.empty(16,dtype=int)

@par(gpu=True)
for i in range(16):
    c[i]=a[i]+b[i]


#
# Segmentation fault
#

import numpy as np

MAX=1000  # maximum Mandelbrot iterations
N=4096  # width and height of image
pixels=np.empty((N,N),dtype=int)

def scale(x,a,b):
    return a+(x/N)*(b-a)

@par(gpu=True,collapse=2)
for i in range(N):
    for j in range(N):
        c=complex(scale(j,-2.00,0.47),scale(i,-1.12,1.12))
        z=0j
        iteration=0

        while abs(z)<=2 and iteration<MAX:
            z=z**2+c
            iteration+=1

        pixels[i,j]=int(255*iteration/MAX)


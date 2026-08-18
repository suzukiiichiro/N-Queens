import gpu

@gpu.kernel
def hello(a, b, c):
    i = gpu.thread.x
    c[i] = a[i] + b[i]

a = [i for i in range(16)]
b = [2*i for i in range(16)]
c = [0 for _ in range(16)]

hello(a, b, c, grid=1, block=1)
print(c)

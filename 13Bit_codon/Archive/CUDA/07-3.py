import gpu

MAX = 200
N = 256
pixels = [0 for _ in range(N * N)]

@gpu.kernel
def mandel(pixels, N: int, MAX: int):
    idx = gpu.block.x * gpu.block.dim.x + gpu.thread.x
    if idx >= N * N:
        return

    i = idx // N
    j = idx - i * N

    # scale(j, -2.00, 0.47)
    cr = -2.00 + (float(j) / float(N)) * (0.47 - (-2.00))
    # scale(i, -1.12, 1.12)
    ci = -1.12 + (float(i) / float(N)) * (1.12 - (-1.12))

    zr = 0.0
    zi = 0.0
    it = 0

    while (zr*zr + zi*zi) <= 4.0 and it < MAX:
        zr2 = zr*zr - zi*zi + cr
        zi2 = 2.0*zr*zi + ci
        zr = zr2
        zi = zi2
        it += 1

    pixels[idx] = (255 * it) // MAX

block = 256
grid = (N * N + block - 1) // block

mandel(gpu.raw(pixels), N, MAX, grid=grid, block=block)

print(pixels[:64])
print("min/max:", min(pixels), max(pixels))

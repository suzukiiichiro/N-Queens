import gpu

MAX = 1000
N = 4096

pixels = [0 for _ in range(N * N)]

@gpu.kernel
def mandel(pixels, N: int, MAX: int):
    idx = gpu.block.x * gpu.block.dim.x + gpu.thread.x
    if idx >= N * N:
        return

    i = idx // N
    j = idx - i * N

    cr = -2.00 + (float(j) / float(N)) * (0.47 - (-2.00))
    ci = -1.12 + (float(i) / float(N)) * (1.12 - (-1.12))

    zr = 0.0
    zi = 0.0
    it = 0

    while (zr * zr + zi * zi) <= 4.0 and it < MAX:
        zr2 = zr * zr - zi * zi + cr
        zi2 = 2.0 * zr * zi + ci
        zr = zr2
        zi = zi2
        it += 1

    pixels[idx] = (255 * it) // MAX

block = 256
grid = (N * N + block - 1) // block
mandel(gpu.raw(pixels), N, MAX, grid=grid, block=block)

print("min/max:", min(pixels), max(pixels))

mid = (N // 2) * N + (N // 2)
print("center slice:", pixels[mid - 10: mid + 10])

row = N // 2
start = row * N + (N // 2 - 32)
print("mid row slice:", pixels[start:start + 64])

# ---- ASCII PGM (P2) writer: bytes/bytearray/encode 不要 ----
# 出力は大きいので、1行を小さなチャンクに分けて書きます
with open("mandel_4096_p2.pgm", "w") as f:
    f.write("P2\n")
    f.write(str(N))
    f.write(" ")
    f.write(str(N))
    f.write("\n255\n")

    for i in range(N):
        base = i * N
        chunk = ""
        for j in range(N):
            chunk += str(pixels[base + j])
            chunk += " "
            if len(chunk) >= 8192:
                f.write(chunk)
                chunk = ""
        if len(chunk) > 0:
            f.write(chunk)
        f.write("\n")

print("wrote mandel_4096_p2.pgm")

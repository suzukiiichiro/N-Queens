#numpy配列を渡す場合はオプションをつけて実行する必要がある
#codon run -release -disable-exceptions -ptx out.ptx 07-6.py
import numpy as np
import gpu

MAX = 1000
N = 4096

def write_pgm_p2(path: str, img: np.ndarray):
    h, w = img.shape
    with open(path, "w") as f:
        f.write("P2\n")
        f.write(f"{w} {h}\n")
        f.write("255\n")
        for i in range(h):
            row = img[i]
            chunk = ""
            for j in range(w):
                chunk += str(int(row[j])) + " "
                if len(chunk) >= 8192:
                    f.write(chunk)
                    chunk = ""
            if chunk:
                f.write(chunk)
            f.write("\n")

pixels = np.empty((N, N), dtype=np.int32)

@gpu.kernel
def mandel(pixels) -> None:
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

    while (zr*zr + zi*zi) <= 4.0 and it < MAX:
        zr2 = zr*zr - zi*zi + cr
        zi2 = 2.0*zr*zi + ci
        zr = zr2
        zi = zi2
        it += 1

    pixels[i, j] = (255 * it) // MAX

block = 256
grid = (N * N + block - 1) // block
mandel(pixels, grid=grid, block=block)

pixels_u8 = pixels.astype(np.uint8)

print(pixels_u8[0, :64])
print(int(pixels_u8.min()), int(pixels_u8.max()))
mid = N // 2
print(pixels_u8[mid, mid-10:mid+10])

write_pgm_p2("mandel_4096_p2.pgm", pixels_u8)
print("wrote mandel_4096_p2.pgm")
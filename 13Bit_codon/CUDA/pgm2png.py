#!/usr/bin/env python3
import zlib
import struct

def read_pgm_p2(path):
    with open(path, "r") as f:
        def next_token():
            while True:
                tok = f.readline()
                if tok == "":
                    return None
                tok = tok.strip()
                if not tok or tok.startswith("#"):
                    continue
                # 1行に複数トークンあり得るのでキュー化
                for t in tok.split():
                    yield t

        it = next_token()
        magic = next(it)
        if magic != "P2":
            raise ValueError(f"Only P2 supported, got {magic}")

        w = int(next(it))
        h = int(next(it))
        maxv = int(next(it))

        vals = []
        # 以降は数値が続く
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.extend(int(x) for x in line.split())

    if len(vals) < w * h:
        raise ValueError(f"Not enough pixels: got {len(vals)}, need {w*h}")

    # 0..255 にスケール
    if maxv <= 0:
        raise ValueError("Invalid max value")
    if maxv == 255:
        pix = bytes(vals[:w*h])
    else:
        pix = bytes((v * 255) // maxv for v in vals[:w*h])

    return w, h, pix

def write_png_gray8(path, w, h, pix_bytes):
    # PNG: 8-bit grayscale (color type 0)
    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff))

    # 各行の先頭に filter=0 を付ける
    raw = bytearray()
    row_bytes = w
    for y in range(h):
        raw.append(0)
        start = y * row_bytes
        raw.extend(pix_bytes[start:start+row_bytes])

    comp = zlib.compress(bytes(raw), level=9)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)

    with open(path, "wb") as f:
        f.write(sig)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", comp))
        f.write(chunk(b"IEND", b""))

if __name__ == "__main__":
    in_pgm = "mandel_4096_p2.pgm"
    out_png = "mandel_4096.png"
    w, h, pix = read_pgm_p2(in_pgm)
    write_png_gray8(out_png, w, h, pix)
    print(f"wrote {out_png} ({w}x{h})")

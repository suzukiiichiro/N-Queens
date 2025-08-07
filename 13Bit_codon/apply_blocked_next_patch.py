import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
code = SRC.read_text(encoding="utf-8")

# while からブロック全体を取る（インデントレベル維持）
pattern = re.compile(
    r'^(\s*)while\s+\w+[^\n]*:\s*\n'   # while <var>:
    r'((?:\1\s+.+\n)+)',               # 同じインデント+1以上の行
    re.MULTILINE
)

def inject_blocked_next(block_body: str) -> str:
    return re.sub(
        r'^(\s*)if\s+next_free\s*:\s*\n',
        lambda m: (
            m.group(0) +
            f"{m.group(1)}if row + 1 < endmark:\n"
            f"{m.group(1)}    blocked_next:int = (next_ld << 1) | (next_rd >> 1) | next_col\n"
            f"{m.group(1)}    if row + 1 == mark1:\n"
            f"{m.group(1)}        blocked_next &= ~(1 << (N - 1 - mark1))\n"
            f"{m.group(1)}    if row + 1 == mark2:\n"
            f"{m.group(1)}        blocked_next &= ~(1 << (N - 1 - mark2))\n"
            f"{m.group(1)}    if (board_mask & ~blocked_next) == 0:\n"
            f"{m.group(1)}        continue\n"
        ),
        block_body,
        flags=re.MULTILINE
    )

def repl(m):
    indent = m.group(1)
    body = m.group(2)
    body_new = inject_blocked_next(body)
    return f"{indent}while ...:\n{body_new}"  # while行は戻すときにそのまま使うべきなら m.group(0)[:...] で

# 実際には while ... の行そのまま残す
def repl_keep_while(m):
    while_line = f"{m.group(1)}while" + m.group(0).split("while",1)[1].split("\n",1)[0] + "\n"
    body = m.group(2)
    return while_line + inject_blocked_next(body)

code_new = pattern.sub(repl_keep_while, code)

SRC.write_text(code_new, encoding="utf-8")
print(f"Patched: {SRC}")


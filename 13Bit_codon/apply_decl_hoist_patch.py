import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
code = SRC.read_text(encoding="utf-8")

# 対象の一時変数
temps = ["bit","next_ld","next_rd","next_col","ld1","rd1","col1","blocked","blocked_next","next_free"]
temps_regex = "|".join(temps)

# 1) SQ* 関数ブロックを走査
func_pattern = re.compile(
    r'(def\s+SQ\w+\s*\([^)]*\)\s*->\s*int\s*:\s*\n)'  # ヘッダ
    r'(?P<body>(?:\s+.+\n)+?)'                        # 本文
    r'(?=\n\s*def\s+|\Z)',                            # 次のdef か EOF まで
    re.MULTILINE
)

def hoist_in_func(mfunc):
    head = mfunc.group(1)
    body = mfunc.group('body')

    # 2) while ブロックを見つける（最初の while だけを対象にするが、必要なら複数対応にも拡張可）
    while_pattern = re.compile(r'^(\s*)while\s+[^\n:]+:\s*\n', re.MULTILINE)
    mwhile = while_pattern.search(body)
    if not mwhile:
        return mfunc.group(0)

    indent = mwhile.group(1)
    # 3) ループ先頭直後の位置を求める（while 行の直後）
    start = mwhile.end()

    # 4) その while ブロック内での「型付き宣言」を「型なし代入」に置換
    #    例: "next_ld:int = ..." → "next_ld = ..."
    typed_decl_pattern = re.compile(
        rf'^({indent}\s*)\b({temps_regex})\s*:\s*int\s*=\s*',
        re.MULTILINE
    )
    body2 = typed_decl_pattern.sub(r'\1\2 = ', body)

    # 5) ループ直前（while 行の直前）に一度だけ「宣言」を挿入
    #    すでに同名の宣言が直前にある場合は挿入しない簡易チェック
    pre_while_region = body[:mwhile.start()]
    if not re.search(rf'^{indent}\b(?:{temps_regex})\s*:\s*int\s*=\s*0\s*$', pre_while_region, re.MULTILINE):
        decls = "".join([f"{indent}{v}:int = 0\n" for v in temps])
        body2 = body2[:mwhile.start()] + decls + body2[mwhile.start():]

    return head + body2

code = func_pattern.sub(hoist_in_func, code)

# 6) 余分な空行の簡易整理（任意）
code = re.sub(r'\n{3,}', '\n\n', code)

SRC.write_text(code, encoding="utf-8")
print("Patched:", SRC)


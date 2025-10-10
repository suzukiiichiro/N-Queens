#!/usr/bin/env bash

# codon build -release を実行する
# 使い方： 
# $ bash MAIN.SH <python.py>

set -Eeuo pipefail

if [ $# -lt 1 ]; then
  echo "使い方: $0 <ソースファイル（例: test.py）>"
  exit 1
fi

targetFile=$1
# echo "targetFile: $targetFile"

# パスの最後の要素だけ取り出し（例: src/foo.bar.py -> foo.bar.py）
base=${targetFile##*/}

# .env のような「拡張子なしの隠しファイル」はそのまま扱う
if [[ $base == .* && $base != *.* ]]; then
  execFile="$base"
else
  # 最後の拡張子だけ除去（foo.bar.py -> foo.bar）
  execFile=${base%.*}
fi

# echo "execFile: $execFile"

# ソースのあるディレクトリでビルド＆実行（出力名の混乱を避ける）
srcdir=${targetFile%/*}
[[ $srcdir == "$targetFile" ]] && srcdir="."

pushd "$srcdir" >/dev/null
codon build -release "$base" && "./$execFile"
popd >/dev/null


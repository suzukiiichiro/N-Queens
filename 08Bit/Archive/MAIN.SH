#!/bin/bash

# Usage
#
# debug mode
# bash MAIN.SH <sourceFile.c> -d
#
# plain mode
# bash MAIN.SH <sourceFile.c>


sourceFile="$1";
debugFlag="$2";
# ピリオド以降を切り落として実行ファイル名とする
executeFile=$( echo "$sourceFile" | sed -e "s/\..*$//g" );

# ファイル名にピリオドがない場合は無効なファイル名としてexitする
if echo "$sourceFile" | grep "\." >/dev/null; then
  echo "fileName validate success";
else 
  echo "fileName validte VOID....";
  exit ;
fi

# -d オプションが二つ目のパラメータでデバッグモードとして実行
if [ "$2" == "-d" ]; then
  echo "debug mode";
  gcc -Wall -W -g -O0 -std=c99 -o "$executeFile" "$sourceFile";
else
  echo "plain mode"
  gcc -Wall -W -O3 -std=c99 -o "$executeFile" "$sourceFile";
fi

echo "sourceFile: $sourceFile"
echo "executeFile: $executeFile";

exit;

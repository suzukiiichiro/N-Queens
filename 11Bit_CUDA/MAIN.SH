#!/bin/bash

# $ bash MAIN.SH Main4.c gcc


: '
# Java Main.java
17:         95815104               0            00:12.79
# コンパイルオプションなしの Main.c
17:         95815104                      00:00:00:23.04
# コンパイルオプション付きの Main.c
17:         95815104                      00:00:00:13.74
# GCC Main_ani1.c
17:         95815104               0     000:00:00:13.77
# GCC Main3.c
17:         95815104               0     000:00:00:12.80
# GCC Main4.c
17:         95815104               0     000:00:00:12.74
# 05CUDA_CarryChain.c 20240829
17:         95815104               0     000:00:00:12.71

# GCC Symmetry
17:         95815104        11977939               24.08
# GCC Pthread
17:         95815104        11979381      00:00:00:10.25
# GPU BitBoard
17:         95815104        11977939     000:00:00:00.26
';

# ソースファイルの指定
JavaSourceFile="$1";
gccSourceFile="$1";
#
# 実行方法
#
# Java
# $ bash MAIN.SH sourcefile java
#
# gcc
# $ bash MAIN.SH sourcefile gcc
#
function func_Java(){
  # コンパイル
  javac -cp .:commons-lang3-3.4.jar $JavaSourceFile;  
  # 実行
  java  -cp .:commons-lang3-3.4.jar: -server -Xms4G -Xmx8G -XX:-HeapDumpOnOutOfMemoryError -XX:NewSize=256m -XX:MaxNewSize=256m -XX:-UseAdaptiveSizePolicy  Main  ;
}
#
function func_gcc(){
  # コンパイル
  # gcc $gccSourceFile ;
  # gcc -Wall -W -O3 -ftrapv -std=c99 -Wno-sign-compare $gccSourceFile&& ./a.out ;
  # g++ -O3 -mtune=native -march=native -mfpmath=both $gccSourceFile && ./a.out  
  g++ -O3 -mtune=native -march=native -falign-loops -fgcse-after-reload -finline-functions -finline-functions-called-once -fipa-cp-clone -fpredictive-commoning -ftree-vectorize -funswitch-loops -fmove-loop-invariants $gccSourceFile && ./a.out  

}
if [ "java" == "$2" ];then
  func_Java ;
elif [ "gcc" == "$2" ];then
  func_gcc;
else 
  echo ""
  echo "usage"
  echo "$ bash MAIN.SH sourcefile java";
  echo "$ bash MAIN.SH sourcefile gcc";
fi


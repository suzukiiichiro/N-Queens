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
# 06CUDA_CarryChain.c 20240924
17:         95815104               0     000:00:00:12.11


# 04CUDA_Symmetry_BitBoard.cu
17:         95815104        11977939     000:00:00:00.26
# 07CUDA_CarryChain.cu 20240924
17:         95815104               0     000:00:00:03.41

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
nvccSourceFile="$1";
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
function func_gmon(){
  # 「実行が完了したら」 gmon.out が生成される
  > GMON.OUT;
  gcc -pg -std=c99 $gccSourceFile && gprof a.out gmon.out | tee -a  GMON.OUT ;
}
function func_cuda(){
  # POCL_DEBUG=all デバッグ出力を全開に
  nvcc -O3 -arch=sm_61 -m64 $nvccSourceFile && POCL_DEBUG=all ./a.out -n ;
  #nvcc -O3 -arch=sm_61 -m64 $nvccSourceFile && ./a.out -n
  #nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false $nvccSourceFile && POCL_DEBUG=all ./a.out -n ;
}
if [ "java" == "$2" ];then
  func_Java ;
elif [ "gcc" == "$2" ];then
  func_gcc;
elif [ "gmon" == "$2" ];then
  func_gmon;
elif [ "nvcc" == "$2" ];then
  func_cuda;
else 
  echo ""
  echo "usage"
  echo "$ bash MAIN.SH sourceFile java";
  echo "$ bash MAIN.SH sourceFile gcc";
  echo "$ bash MAIN.SH sourceFile gmon";
  echo "$ bash MAIN.SH sourceFile nvcc";
fi


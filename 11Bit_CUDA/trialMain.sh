#!/usr/bin/bash

# 使い方
# $ bash trialMain.sh
#
# 

# 結局 screen を起動して
# https://qiita.com/hnishi/items/3190f2901f88e2594a5f
# #
# デタッチする（離れる）
# Ctrl z + d
#
# アタッチする
# screen -r 
clear;
rm -fr a.out;
date;
#nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false $sourceFile && ./a.out -n
#nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu && ./a.out -n
nvcc -O3 -arch=sm_61 -m64 04CUDA_Symmetry_BitBoard.cu && ./a.out -n
#./a.out -n 2>&1 | tee -a OUT
# gcc -O3 -W -Wall -mtune=native -march=native -arch=sm_61 04CUDA_Symmetry_BitBoard.cu && POCL_DEBUG=all ./a.out -n 

# デタッチする（離れる）
#  + d
#
# アタッチする
# screen -r 


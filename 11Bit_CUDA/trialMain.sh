#!/usr/bin/bash

# 使い方
# $ bash trialMain.sh
#
# 

# 結局 screen を起動して
# https://qiita.com/hnishi/items/3190f2901f88e2594a5f
# #
# デタッチする（離れる）
#  + d
#
# アタッチする
# screen -r 

sourceFile="04CUDA_Symmetry_BitBoard.cu";  # ソースファイル
nvcc -O3 -arch=sm_61 $sourceFile && ./a.out -n ;

# デタッチする（離れる）
#  + d
#
# アタッチする
# screen -r 


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

# 実行
# bash-5.2$ nohup bash trialMain.sh &
rm -fr a.out;
:>OUT
clear;
date;
nvcc -O3 -arch=sm_61 -m64 04CUDA_Symmetry_BitBoard.cu && ./a.out -n|tee -a OUT

# デタッチする（離れる）
#  + d
#
# アタッチする
# screen -r 


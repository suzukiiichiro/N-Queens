#!/usr/bin/bash

# 使い方
# $ bash trialMain.sh sourceFile
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

#nvcc -O3 -arch=sm_61 -m64 $sourceFile && ./a.out -n|tee -a OUT

# nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false $sourceFile && POCL_DEBUG=all ./a.out -n ;
codon build -release 15Py_constellations_optimize_codon.py
./15Py_constellations_optimize_codon | tee -a OUT


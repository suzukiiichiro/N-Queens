#!/usr/bin/bash

# $B;H$$J}(B
# $ bash trialMain.sh sourceFile
#
# 

# $B7k6I(B screen $B$r5/F0$7$F(B
# https://qiita.com/hnishi/items/3190f2901f88e2594a5f
# #
# $B%G%?%C%A$9$k!JN%$l$k!K(B
# Ctrl z + d
#
# $B%"%?%C%A$9$k(B
# screen -r 

# $B<B9T(B
# bash-5.2$ nohup bash trialMain.sh &
rm -fr a.out;
:>OUT
clear;
date;

#nvcc -O3 -arch=sm_61 -m64 $sourceFile && ./a.out -n|tee -a OUT

# nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false $sourceFile && POCL_DEBUG=all ./a.out -n ;
codon build -release 15Py_constellations_optimize_codon.py
./15Py_constellations_optimize_codon | tee -a OUT


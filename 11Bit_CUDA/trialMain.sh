#!/usr/bin/bash

# $B;H$$J}(B
# $ bash trialMain.sh
#
# 

# $B7k6I(B screen $B$r5/F0$7$F(B
# https://qiita.com/hnishi/items/3190f2901f88e2594a5f
# #
# $B%G%?%C%A$9$k!JN%$l$k!K(B
#  + d
#
# $B%"%?%C%A$9$k(B
# screen -r 

sourceFile="04CUDA_Symmetry_BitBoard.cu";  # $B%=!<%9%U%!%$%k(B
nvcc -O3 -arch=sm_61 $sourceFile && ./a.out -n ;

# $B%G%?%C%A$9$k!JN%$l$k!K(B
#  + d
#
# $B%"%?%C%A$9$k(B
# screen -r 


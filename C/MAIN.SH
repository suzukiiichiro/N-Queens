#!/bin/bash

# 07_01
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_01N-Queen.c -o 01N-Queen && ./07_01

# 07_02
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_02N-Queen.c -o 02N-Queen && ./07_02

# 07_03
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_03N-Queen.c -o 03N-Queen && ./07_03

# 07_04
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_04N-Queen.c -o 04N-Queen && ./07_04

# 07_05
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_05N-Queen.c -o 05N-Queen && ./07_05

# 07_06
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_06N-Queen.c -o 06N-Queen && ./07_06

# 07_07
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_07N-Queen.c -o 07N-Queen && ./07_07

# 07_08
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_08N-Queen.c -o 08N-Queen && ./07_08

# 07_09
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_09N-Queen.c -o 09N-Queen && ./07_09

# 07_10
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_10N-Queen.c -o 10N-Queen && ./07_10

# 07_11
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm 07_11N-Queen.c -o 11N-Queen && ./07_11

# 07_12
#gcc -Wall -W -O3 -g -ftrapv -std=c99 -lm -pthread 07_12N-Queen.c -o 12N-Queen && ./07_12

# gpu version
#nvcc gpuNQueen.cu -v -o gpuNQueen -l pthread && ./gpuNQueen ;
#nvcc gpuNQueen.cu -v  -Xcompiler="-pthread" -o gpuNQueen -l pthread && ./gpuNQueen ;
#nvcc gpuNQueen.cu -v  -o gpuNQueen -I/usr/local/NVIDIA_CUDA_SDK/common/inc -l pthread && ./gpuNQueen ;
#nvcc gpuNQueen.cu -v -o gpuNQueen -l pthread && ./gpuNQueen ;
#nvcc gpuNQueen.cu -v -o gpuNQueen && ./gpuNQueen ;
nvcc 07_13N-Queen.cu -v -o 07_13NQueen && ./07_13NQueen ;

exit ;
